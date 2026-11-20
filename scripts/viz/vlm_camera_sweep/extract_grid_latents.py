"""Run the Psi0 checkpoint on the training start frames and on the grid renders; record the latents (stage 2 of
the in-distribution experiment).

Runs in Psi0's venv. Inference mirrors ``serve_psi0_simple.Server.predict_action`` on the first request of an
episode: PIL image -> the run's resize + center-crop -> Qwen3-VL chat template [image, instruction] -> the frozen
VLM (all 29 hidden states) -> the action expert denoises one chunk in --num-inference-steps Euler steps from fixed
noise, conditioned on the last VLM layer and the normalised proprio state.

Observations:
  train   frame --ref-frames of every training episode, decoded from the train pack's video like the training
          dataloader does, with that frame's recorded state: the reference set ("what the VLA started from")
  grid    every render of render_base_grid.py, with the state of the grid scene's first training frame (the robot
          keeps that posture at every base pose)

The action expert is a joint-attention transformer: in each of its 6 blocks the 30 action tokens and the context
tokens (the VLM tokens + 1 proprio token) attend to each other. A recording attention processor (set on the blocks
here, the model code is untouched and its output unchanged) keeps the rows of the action queries:
  attn        (n, steps, blocks, 30 + N + 1)  attention of the action tokens (mean over the 24 heads and the 30
              action tokens) on every token: [30 action tokens | N VLM tokens | proprio token], each row sums to 1
  attn_heads  (n, blocks, heads, 30 + N + 1)  the same per head, averaged over the denoising steps
The default weights w of an "action-attention-weighted" latent are that attention on the image tokens, averaged over
steps and blocks and renormalised to sum to 1 over the image tokens.

Saved to <out>/latents/ (float16 unless noted; L = 29 VLM layers, 0 = the embeddings, where the image tokens are the
ViT/merger output; D = 2048):
  pool_mean_img   (n, L, D)   mean of the image tokens                                  (step 3)
  pool_attn_img   (n, L, D)   sum_k w_k h_k over the image tokens                       (steps 1 + 3)
  pool_attn_all   (n, L, D)   the same over all VLM tokens (weights renormalised over them)
  pool_mean_txt   (n, L, D)   mean of the tokens after the image (instruction + template)
  pool_last       (n, L, D)   the last token
  tokens_l<l>     (n, N_img, D) every image token of layer l, for l in --spatial-layers  (step 4)
  ctx_mean, ctx_attn (n, blocks, 1536)  the action expert's own context stream entering each block (block 0: the
                  projected last VLM layer), image positions pooled by mean / by w, averaged over the denoising steps
  act             (n, 30, 1536)  the action tokens after the last block, averaged over the denoising steps: the
                  fine-tuned part's own latent, from which the action is read
  attn, attn_heads  float32, see above
  pred            (n, seeds, 30, Da) float32  the predicted chunk, denormalised, one per noise seed (--action-seeds)
and <out>/latents.json (the observation table, token layout, massive channels per layer: mean |h| over the
reference image tokens > 8x the layer's median channel).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision.transforms import v2

from psi.models.psi0 import Psi0Model
from psi.utils import apply_legacy_model_config_defaults, pad_to_len, parse_args_to_tyro_config, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renders", required=True, help="output directory of render_base_grid.py")
    p.add_argument("--prep", required=True, help="directory of prepare_base_grid.py")
    p.add_argument("--out", default=None, help="default: --renders")
    p.add_argument("--run-dir", default=None, help="default: the one prepare_base_grid.py resolved")
    p.add_argument("--ckpt-step", default=None)
    p.add_argument("--ref-frames", default="0", help="comma list: frames of every training episode in the reference set")
    p.add_argument("--spatial-layers", default="0,4,8,12,16,20,24,28",
                   help="VLM layers whose image tokens are all saved, or 'all'")
    p.add_argument("--instruction", default="", help="default: the train pack's task, as the dataloader gives it")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--action-seeds", type=int, default=4, help="chunks predicted per observation, from different noise")
    p.add_argument("--limit", type=int, default=0, help="only the first N of each group (0 = all), for a quick look")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


class RecordingProcessor:
    """Wraps a block's JointVLAAttnProcessor: same output, and the attention rows of the action queries are kept."""

    def __init__(self, inner, sink: list):
        self.inner, self.sink, self.on = inner, sink, False

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        if self.on:
            b, heads = hidden_states.shape[0], attn.heads
            split = lambda x: x.view(b, -1, heads, x.shape[-1] // heads).transpose(1, 2)
            q, k, ek = split(attn.to_q(hidden_states)), split(attn.to_k(hidden_states)), \
                split(attn.add_k_proj(encoder_hidden_states))
            if attn.norm_q is not None:
                q = attn.norm_q(q)
            if attn.norm_k is not None:
                k = attn.norm_k(k)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            keys = torch.cat([k, ek], dim=2).float()
            scores = q.float() @ keys.transpose(-1, -2) / math.sqrt(q.shape[-1])  # (b, heads, actions, all tokens)
            if attention_mask is not None:
                keep = torch.cat([torch.ones(b, hidden_states.shape[1], dtype=torch.bool, device=scores.device),
                                  attention_mask.to(torch.bool)], dim=1)
                scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))
            self.sink.append(scores.softmax(-1).mean(dim=2)[0])  # (heads, all tokens), b = 1
        return self.inner(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                          attention_mask=attention_mask, *args, **kwargs)


def video_frame(index: dict, episode: int, frame: int) -> np.ndarray:
    path = Path(index["train_dir"]) / index["video_path"].format(
        episode_chunk=episode // index["chunks_size"], video_key=index["video_key"], episode_index=episode)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})",
                          "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    h, w = index["image_shape"][:2]
    if len(raw) != h * w * 3:
        raise RuntimeError(f"{path}: frame {frame} could not be decoded")
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def main() -> None:
    args = parse_args()
    renders, prep = Path(args.renders), Path(args.prep)
    out = Path(args.out) if args.out else renders
    (out / "latents").mkdir(parents=True, exist_ok=True)
    index = json.loads((prep / "train_index.json").read_text())
    plan = json.loads((prep / "plan.json").read_text())
    meta = json.loads((renders / "meta.json").read_text())
    rows = [json.loads(line) for line in open(renders / "index.jsonl")]
    run = Path(args.run_dir or index["run_dir"])
    step = args.ckpt_step or index["ckpt_step"]
    dev = args.device
    started = time.perf_counter()

    cfg_ = parse_args_to_tyro_config(run / "argv.txt")
    lc = cfg_.model_validate(apply_legacy_model_config_defaults(json.loads((run / "run_config.json").read_text())))
    seed_everything(0)
    model = Psi0Model.from_pretrained(run, step, lc, device=dev).to(dev).eval()
    field, mt, repack = lc.data.transform.field, lc.data.transform.model, lc.data.transform.repack
    img_t = v2.Compose([mt.resize(), mt.center_crop()])
    Tp, Da = lc.model.action_chunk_size, lc.model.action_dim
    pad_state = getattr(repack, "pad_state_dim", None) or getattr(field, "pad_state_dim", None)
    instruction = (args.instruction or index["instruction"]).lower()  # the server lowercases too
    image_pad_id = model.vlm_processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    blocks = model.action_header.transformer_blocks
    n_blocks = len(blocks)

    # ---- recorders: attention rows of the action queries, and the context stream entering each block
    attn_sink, ctx_sink, act_sink, recorders = [], [], [], []
    for blk in blocks:
        rec = RecordingProcessor(blk.attn.processor, attn_sink)
        blk.attn.processor = rec
        recorders.append(rec)
        blk.register_forward_pre_hook(
            lambda _m, _a, kw: ctx_sink.append(kw["obs_hidden_states"][0].float()) if recorders[0].on else None,
            with_kwargs=True)
    blocks[-1].register_forward_hook(
        lambda _m, _a, result: act_sink.append(result[0][0].float()) if recorders[0].on else None)

    @torch.inference_mode()
    def run_vlm(pil: Image.Image):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": instruction}]}]]
        text = [model.vlm_processor.apply_chat_template(m_, tokenize=False, add_generation_prompt=True) for m_ in msgs]
        imgs, vids = process_vision_info(msgs, image_patch_size=16)
        inp = model.vlm_processor(text=text, images=imgs, videos=vids, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model.vlm_model(**inp, output_hidden_states=True, return_dict=True).hidden_states
        return hs, inp

    @torch.inference_mode()
    def run_action(hs, inp, state_raw: np.ndarray, seed: int, record: bool) -> np.ndarray:
        st = state_raw[None].astype(np.float32)
        if pad_state is not None:
            st, _ = pad_to_len(st, pad_state, dim=1)
        if getattr(field, "normalize_state", True):
            st = field.normalize_state_func(st)
        st = torch.as_tensor(np.asarray(st, dtype=np.float32)).to(dev)[None]  # (1, 1, Ds)
        a = torch.randn(1, Tp, Da, device=dev, generator=torch.Generator(dev).manual_seed(seed))
        model.noise_scheduler.set_timesteps(args.num_inference_steps)
        for r in recorders:
            r.on = record
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in model.noise_scheduler.timesteps:
                v = model.action_header(
                    hidden_states=None, timestep=t.expand(1).to(dev), pooled_projections=None,
                    joint_attention_kwargs=dict(action_hidden_embeds=a, views=model._select_vlm_views(hs),
                                                obs=st, traj2ds=None),
                    vlm_attn_mask=inp["attention_mask"], return_dict=True).action
                a = model.noise_scheduler.step(model_output=v, timestep=t, sample=a).prev_sample
        for r in recorders:
            r.on = False
        return field.denormalize(a.float().reshape(Tp, Da).cpu().numpy()).astype(np.float32)

    # ---- the observations: the reference set first
    obs = []
    ref_frames = [int(v) for v in args.ref_frames.split(",")]
    packs = {}
    for e in index["episodes"][:args.limit or None]:
        for f in ref_frames:
            if f >= e["length"]:
                continue
            if f == 0:
                state = np.asarray(e["state"], dtype=np.float32)
            else:
                import pandas as pd
                if e["episode"] not in packs:
                    packs[e["episode"]] = pd.read_parquet(next(Path(index["train_dir"]).glob(
                        f"data/chunk-*/episode_{e['episode']:06d}.parquet")), columns=["states"])
                state = np.asarray(packs[e["episode"]]["states"].iloc[f], dtype=np.float32)
            obs.append(dict(group="train", episode=e["episode"], frame=f, state=state,
                            load=lambda e=e, f=f: Image.fromarray(video_frame(index, e["episode"], f))))
    anchor_state = np.asarray(meta["anchor"]["state"], dtype=np.float32)
    by_id = {q["id"]: q for q in plan["poses"]}
    grid_rows = [r for r in rows if r["kind"] != "recorded"]
    for r in grid_rows[:args.limit or None]:
        obs.append(dict(group="grid", kind=r["kind"], id=r["id"], file=r["file"], state=anchor_state,
                        pose=by_id[r["id"]], load=lambda r=r: Image.open(renders / r["file"]).convert("RGB")))
    n = len(obs)
    n_ref = sum(o["group"] == "train" for o in obs)

    names = ("pool_mean_img", "pool_attn_img", "pool_attn_all", "pool_mean_txt", "pool_last")
    store: dict[str, list] = {k: [] for k in (*names, "ctx_mean", "ctx_attn", "act", "attn", "attn_heads", "pred")}
    ids0 = None
    for count, o in enumerate(obs):
        hs, inp = run_vlm(img_t(o["load"]()))
        H = torch.stack([h[0] for h in hs]).float()  # (L, N, D)
        ids = inp["input_ids"][0].cpu().numpy()
        if ids0 is None:
            ids0, grid_thw = ids, inp["image_grid_thw"][0].tolist()
            L, N, D = H.shape
            is_img = ids == image_pad_id
            is_txt = np.arange(N) > np.nonzero(is_img)[0].max()
            img_m, txt_m = torch.as_tensor(is_img, device=H.device), torch.as_tensor(is_txt, device=H.device)
            spatial = list(range(L)) if args.spatial_layers == "all" else [int(v) for v in args.spatial_layers.split(",")]
            tokens = {l: [] for l in spatial}
            absmean = torch.zeros(L, D, device=H.device)
            print(f"[tokens] {N} VLM tokens: {int(is_img.sum())} image ({grid_thw[1] // 2} x {grid_thw[2] // 2}), "
                  f"{int(is_txt.sum())} after the image; {L} layers of {D}; instruction '{instruction}'", flush=True)
        if not np.array_equal(ids, ids0):
            raise RuntimeError(f"observation {count}: its token layout differs from the first one's")
        attn_sink.clear(), ctx_sink.clear(), act_sink.clear()
        preds = [run_action(hs, inp, o["state"], seed, record=seed == 0) for seed in range(args.action_seeds)]
        steps = len(attn_sink) // n_blocks
        AH = torch.stack(attn_sink).reshape(steps, n_blocks, -1, Tp + N + 1)  # (steps, blocks, heads, Tp + N + 1)
        A = AH.mean(2)
        C = torch.stack(ctx_sink).reshape(steps, n_blocks, N + 1, -1).mean(0)  # (blocks, N + 1, 1536)
        w_vlm = A[:, :, Tp:Tp + N].mean(dim=(0, 1))  # attention of the action tokens on each VLM token
        w_img = w_vlm[img_m] / w_vlm[img_m].sum()
        w_all = w_vlm / w_vlm.sum()
        Himg = H[:, img_m]
        f16 = lambda x: x.cpu().numpy().astype(np.float16)
        store["pool_mean_img"].append(f16(Himg.mean(1)))
        store["pool_attn_img"].append(f16((Himg * w_img[None, :, None]).sum(1)))
        store["pool_attn_all"].append(f16((H * w_all[None, :, None]).sum(1)))
        store["pool_mean_txt"].append(f16(H[:, txt_m].mean(1)))
        store["pool_last"].append(f16(H[:, -1]))
        store["ctx_mean"].append(f16(C[:, :N][:, img_m].mean(1)))
        store["ctx_attn"].append(f16((C[:, :N][:, img_m] * w_img[None, :, None]).sum(1)))
        store["act"].append(f16(torch.stack(act_sink).mean(0)))
        store["attn"].append(A.cpu().numpy().astype(np.float32))
        store["attn_heads"].append(AH.mean(0).cpu().numpy().astype(np.float32))
        store["pred"].append(np.stack(preds))
        for l in spatial:
            tokens[l].append(f16(Himg[l]))
        if o["group"] == "train":
            absmean += Himg.abs().mean(1) / n_ref
        if count % 50 == 0 or count == n - 1:
            share = float(A[:, :, Tp:Tp + N][:, :, img_m].sum(-1).mean())
            print(f"[{count + 1}/{n}] {o['group']} {o.get('file') or 'episode %d frame %d' % (o['episode'], o['frame'])}"
                  f"  action attention on the image tokens {share:.2f}", flush=True)

    dest = out / "latents"
    for k, v in store.items():
        np.save(dest / f"{k}.npy", np.stack(v))
    for l, v in tokens.items():
        np.save(dest / f"tokens_l{l}.npy", np.stack(v))
    massive = absmean > 8 * absmean.median(dim=1, keepdim=True).values
    np.save(dest / "massive.npy", massive.cpu().numpy())
    np.save(dest / "ref_absmean.npy", absmean.cpu().numpy())
    table = [{k: v for k, v in o.items() if k not in ("load", "state", "pose")} for o in obs]
    (out / "latents.json").write_text(json.dumps(dict(
        obs=table, n=n, n_ref=n_ref, layers=L, dim=D, blocks=n_blocks, steps=steps, action_tokens=Tp, action_dim=Da,
        heads=int(AH.shape[2]),
        n_tokens=N, is_img=is_img.tolist(), is_txt=is_txt.tolist(), img_grid=[grid_thw[1] // 2, grid_thw[2] // 2],
        tokens=model.vlm_processor.tokenizer.convert_ids_to_tokens(ids0.tolist()), spatial_layers=spatial,
        instruction=instruction, run_dir=str(run), ckpt_step=str(step), ref_frames=ref_frames,
        action_seeds=args.action_seeds, num_inference_steps=args.num_inference_steps,
        vlm_input_size=list(img_t(obs[0]["load"]()).size[::-1]),
        massive_channels={str(l): torch.nonzero(massive[l]).squeeze(-1).tolist() for l in range(L)}), indent=1))
    print(f"saved {dest}  ({n_ref} training frames + {n - n_ref} renders, {int(is_img.sum())} image tokens of {N}, "
          f"image grid {grid_thw[1] // 2}x{grid_thw[2] // 2}) in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
