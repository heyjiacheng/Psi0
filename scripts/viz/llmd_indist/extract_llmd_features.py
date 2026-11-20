"""Stage 2 of the LLMD in-distribution experiment: the action expert's last-layer features (Psi0 venv, GPU).

LLMD (VLA-FAIL, arXiv 2606.21386, section 3.1) scores an observation by the Mahalanobis distance of the action
expert's last-layer features f*(o) = f(o, t=0, a0*) to the finetuning data: ONE forward pass of the action expert at
the noise end of the flow, with ONE fixed noise sample a0*. On Psi0:

  feature      the input of ``action_header.action_proj_out.linear`` (the final linear layer, 1536 -> 36 per action
               token): 30 action tokens x 1536. This legacy checkpoint has no final LayerNorm, so that input is
               f = h * scale(t) + shift(t), with h the action stream leaving the last transformer block and
               (scale, shift) a function of the timestep only. h is what is stored (float32); the fit stage forms f
               in float64, so no precision is lost in channels whose scale is tiny.
  noise end    scheduler timestep 1000 = sigma 1 = 100 % prior noise (the paper's t = 0; training builds
               x_t = (1 - sigma) * action + sigma * noise with timestep = 1000 * sigma). It is also exactly the first
               Euler step of the served policy on the first request of an episode.
  fixed noise  a0* = torch.randn(1, 30, 36) from a seeded CPU generator, saved to feats/noise_seed<k>.npy.

The stage starts with the hook check (checks/hook_check.json) and stops if a hard assertion fails:
  - timestep 1000 is the pure-noise end: scheduler shift 1, sigma(1000) = 1, scale_noise(x, 1000, n) = n, and the
    served ``predict_action`` feeds the untouched prior sample with timestep 1000 into its first header call;
  - the hooks: linear(f) reproduces the header output, f = h * scale + shift, and our single pass reproduces the
    served first step's f for the same noise;
  - numerics: fp32 vs bf16 header, batch composition, repeatability.

Two phases per observation set, so that any header variant can be re-run from the cached VLM output:
  A  image -> the run's resize + centre crop -> Qwen3-VL chat template [image, instruction] -> the frozen VLM under
     bf16 autocast, exactly as served. Kept: the last hidden layer (the action expert's context) as its bf16 bits
     (int16, exact), and for the small sets the image tokens of layer 8 (experiment 3's best signal).
  B  the action expert on the cached context + the normalised 32-D state:
       h_s<k>       fp32 header, timestep 1000, fixed noise seed k              (the LLMD feature; k = 0 is primary)
       h_s0_bf16    the same under bf16 autocast, as served
       f_fresh      timestep 1000, new noise for every observation              (ablation: is the fixed noise needed?)
       f_t500       timestep 500, fixed noise seed 0                            (ablation: is the noise end needed?)
       f_rand       random timestep and noise per observation                   (the paper's App. C.1 ablation)
     For the 35,808 training frames the three ablations are not stored: their first and second moments are
     accumulated in float64 (mom_<name>.npz) and the frames of the grid scene's episode are kept (f_<name>_ep.npy)
     so the fit stage can leave that episode out.

Observation sets (feats/<set>/obs.jsonl lists every row with its labels):
  train       every frame of every training episode, decoded like the training dataloader, with its recorded state
  trainstart  frame 0 of every training episode (also keeps layer-8 tokens)
  grid        experiment 3's 281 renders in the scene of training episode 0 (read-only input)
  start       every training start re-rendered in its own scene      \\
  unused      starts of teleop recordings that were not trained on    > render_starts.py
  extra       19 more poses in the scene of training episode 0       /
  <set>_h264  the same renders pushed through the training videos' H.264 encode/decode (h264_roundtrip.py), with
              the writer's own settings (crf 23); <set>_h264s: with the crf at which a lone frame is quantised like
              the training frames; <set>_h264c: lossless (the colour conversion alone)
  grid_b16    the grid again with 16 observations per VLM pass instead of 1 (the bf16 VLM output depends on the batch
              size at the rounding level; everything else uses 1, as served)
  traj, target  first observation of earlier closed-loop trials (index_trials.py); target_tf = target with the
              checkpoint's own prompt; <set>_ms = with the median training-start state instead of the trial's
  other       frames of other tasks' datasets with a faucet start state (must be rejected)
  swap        image and state taken from different moments of the same training episode (which input drives it?)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision.transforms import v2

from psi.models.psi0 import Psi0Model
from psi.utils import apply_legacy_model_config_defaults, pad_to_len, parse_args_to_tyro_config, seed_everything

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ABLATIONS = ("fresh", "t500", "rand")
RENDER_SETS = ("start", "unused", "extra")
T_NOISE = 1000.0  # scheduler timestep of the pure-noise end
TOKEN_LAYER = 8  # experiment 3's best layer for the token-match score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the experiment directory (prep/, renders/, trials/ are read; feats/ written)")
    p.add_argument("--exp3", required=True, help="experiment 3's directory (read-only): prep/ and exp3_indist_signal/")
    p.add_argument("--simple-dir", required=True)
    p.add_argument("--sets", default="all", help="comma list of observation sets, or 'all' (those whose inputs exist)")
    p.add_argument("--phase", default="ab", choices=("a", "b", "ab", "check"))
    p.add_argument("--seeds", default="0,1,2", help="seeds of the fixed noise a0*")
    p.add_argument("--vlm-batch", type=int, default=1, help="observations per VLM pass; 1 = as served (the bf16 VLM "
                   "output depends on the batch size at the rounding level, so it is kept the same everywhere)")
    p.add_argument("--batch", type=int, default=16, help="observations per fp32 pass of the action expert")
    p.add_argument("--limit", type=int, default=0, help="only the first N observations of each set (0 = all), for a pilot")
    p.add_argument("--suffix", default="", help="appended to feats/ (a pilot writes to feats<suffix>/)")
    p.add_argument("--redo", action="store_true", help="ignore finished phases")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--watchdog", type=int, default=0, help="print every thread's stack each N seconds (0 = off), to find a hang")
    return p.parse_args()


# ------------------------------------------------------------------------------------------------ the model
class Psi0Features:
    """The checkpoint, the hooks on its last layer, and the two passes (VLM, action expert)."""

    def __init__(self, run: Path, step: str, dev: str, batch: int, vlm_batch: int = 1):
        cfg_ = parse_args_to_tyro_config(run / "argv.txt")
        self.lc = cfg_.model_validate(apply_legacy_model_config_defaults(json.loads((run / "run_config.json").read_text())))
        seed_everything(0)
        self.model = Psi0Model.from_pretrained(run, step, self.lc, device=dev).to(dev).eval()
        self.dev, self.batch, self.vlm_batch = dev, batch, vlm_batch
        t = self.lc.data.transform
        self.field = t.field
        self.img_t = v2.Compose([t.model.resize(), t.model.center_crop()])  # as the server does
        self.pad_state = getattr(t.repack, "pad_state_dim", None) or getattr(t.field, "pad_state_dim", None)
        self.Tp, self.Da = self.lc.model.action_chunk_size, self.lc.model.action_dim
        self.hdr = self.model.action_header
        self.proc = self.model.vlm_processor
        self.image_pad_id = self.proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.sink: dict = {}
        self.calls: list | None = None  # set to a list to record every header call (the hook check)
        # h: the action stream leaving the last block; f: what the final linear layer receives
        self.hdr.transformer_blocks[-1].register_forward_hook(self._on_block)
        self.hdr.action_proj_out.linear.register_forward_pre_hook(self._on_linear)
        self.hdr.register_forward_pre_hook(self._on_header, with_kwargs=True)

    def _on_block(self, _m, _args, out) -> None:
        self.sink["h"] = out[0]

    def _on_linear(self, _m, args) -> None:
        self.sink["f"] = args[0]
        if self.calls is not None:
            self.calls[-1]["f"] = args[0].detach().float().clone()

    def _on_header(self, _m, _args, kwargs) -> None:
        if self.calls is not None:  # before forward() pops the sample from the dict
            self.calls.append(dict(timestep=kwargs["timestep"].detach().float().clone(),
                                   sample=kwargs["joint_attention_kwargs"]["action_hidden_embeds"].detach().float().clone()))

    def norm_state(self, states: np.ndarray) -> torch.Tensor:
        """(B, 32) raw states -> (B, 1, 36) normalised, as the server does."""
        st = np.asarray(states, dtype=np.float32)
        if self.pad_state is not None:
            st, _ = pad_to_len(st, self.pad_state, dim=1)
        if getattr(self.field, "normalize_state", True):
            st = self.field.normalize_state_func(st)
        return torch.as_tensor(np.asarray(st, dtype=np.float32)).to(self.dev)[:, None]

    @torch.inference_mode()
    def vlm(self, pils: list[Image.Image], instruction: str):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": p}, {"type": "text", "text": instruction}]}]
                for p in pils]
        text = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        imgs, vids = process_vision_info(msgs, image_patch_size=16)
        inp = self.proc(text=text, images=imgs, videos=vids, padding=True, return_tensors="pt").to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.model.vlm_model(**inp, output_hidden_states=True, return_dict=True).hidden_states
        return hs, inp

    @torch.inference_mode()
    def header(self, views: torch.Tensor, state: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, bf16: bool = False):
        """One pass of the action expert. views (B, 1, N, 2048) bf16 as the VLM returned it; state (B, 1, 36);
        noise (B, 30, 36); t (B,). Returns h, f, the header output (float32)."""
        mask = torch.ones(views.shape[0], views.shape[2], dtype=torch.long, device=self.dev)
        kw = dict(hidden_states=None, timestep=t, pooled_projections=None, vlm_attn_mask=mask, return_dict=True,
                  joint_attention_kwargs=dict(action_hidden_embeds=noise, views=views if bf16 else views.float(),
                                              obs=state, traj2ds=None))
        if bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.hdr(**kw).action
        else:
            out = self.hdr(**kw).action
        return self.sink["h"].float(), self.sink["f"].float(), out.float()

    def noise(self, seed: int) -> torch.Tensor:
        """The fixed prior sample a0* of a seed, (1, 30, 36); from a CPU generator so it does not depend on the GPU."""
        return torch.randn(1, self.Tp, self.Da, generator=torch.Generator("cpu").manual_seed(seed)).to(self.dev)


class Frames:
    """Client of frame_server.py: video frames and H.264 round trips from a process that holds no model. PyAV never
    runs in this process (with torch / torchvision loaded, an abandoned PyAV decoder hung one pilot and crashed
    another); the pixels are h264_roundtrip.py's, i.e. the training dataloader's."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen([sys.executable, "-u", str(HERE / "frame_server.py")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.job = 0

    def _read(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError(f"the frame server died (exit code {self.proc.poll()})")
            buf += chunk
        return bytes(buf)

    def _job(self, job: dict):
        """The frames of one job. A job that is not read to its end is skipped when the next one starts."""
        self.job += 1
        mine = self.job
        self.proc.stdin.write((json.dumps(dict(job, id=mine)) + "\n").encode())
        self.proc.stdin.flush()
        while True:
            if self.job != mine:
                raise RuntimeError("a frame stream was read after a newer one was started")
            jid, i, h, w, c = struct.unpack("<iiiii", self._read(20))
            if i == -2:
                msg = self._read(h).decode()
                if jid == mine:
                    raise RuntimeError(f"frame server: {msg} ({job})")
                continue
            data = self._read(h * w * c) if i >= 0 else b""
            if jid != mine:  # what an earlier, abandoned job still had in the pipe
                continue
            if i == -1:
                return
            yield i, np.frombuffer(data, np.uint8).reshape(h, w, c)

    def video(self, path):
        """(index, rgb) of every frame of a video, in order."""
        return self._job(dict(video=str(path), frames="all"))

    def frame(self, path, index: int) -> np.ndarray:
        got = list(self._job(dict(video=str(path), frames=[int(index)])))
        if len(got) != 1 or got[0][0] != index:
            raise RuntimeError(f"{path}: no frame {index}")
        return got[0][1]

    def roundtrip(self, image_path, **options) -> np.ndarray:
        return list(self._job(dict(image=str(image_path), options=options)))[0][1]

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


def bits(x: torch.Tensor) -> np.ndarray:
    """A bf16 tensor as its int16 bit pattern (exact, half the size of float32)."""
    assert x.dtype == torch.bfloat16, x.dtype
    return x.contiguous().view(torch.int16).cpu().numpy()


def unbits(a: np.ndarray, dev: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).view(torch.bfloat16)


# ------------------------------------------------------------------------------------------------ observation sets
def train_meta(index: dict) -> tuple[Path, callable]:
    train_dir = Path(index["train_dir"])
    video = lambda ep: train_dir / index["video_path"].format(
        episode_chunk=ep // index["chunks_size"], video_key=index["video_key"], episode_index=ep)
    return train_dir, video


def episode_states(train_dir: Path, ep: int) -> np.ndarray:
    import pandas as pd

    f = next(train_dir.glob(f"data/chunk-*/episode_{ep:06d}.parquet"))
    return np.stack(pd.read_parquet(f, columns=["states"])["states"].to_numpy()).astype(np.float32)


def build_sets(args, index: dict, instruction: str, FR: Frames) -> dict:
    """name -> dict(rows, instruction, images(start) -> iterator of PIL images from row `start` on, tokens: bool)."""
    out, exp3 = Path(args.out), Path(args.exp3)
    train_dir, video = train_meta(index)
    sets: dict = {}
    rgb = lambda a: Image.fromarray(np.ascontiguousarray(a))
    videos = [video(e["episode"]) for e in index["episodes"]]

    def exact(state: np.ndarray, rounded, what: str) -> list:
        """The full-precision state, checked against the copy an index file holds rounded to 6 decimals. Every set
        uses the unrounded state, as the training rows do: three finger joints have a normalisation range of about
        0.001 rad, where the rounding would be visible."""
        if float(np.abs(np.asarray(state, np.float64) - np.asarray(rounded, np.float64)).max()) > 2e-6:
            raise SystemExit(f"{what}: the recorded state differs from the index's copy")
        return np.asarray(state, np.float32).tolist()

    def files(rows: list[dict], root: Path, codec: dict | None = None):
        def images(start: int = 0):
            for r in rows[start:]:
                yield (rgb(FR.roundtrip(root / r["file"], **codec)) if codec is not None
                       else Image.open(root / r["file"]).convert("RGB"))
        return images

    def with_codec(name: str, rows: list[dict], root: Path, tokens: bool) -> None:
        # the same images through the training videos' H.264 (h264_roundtrip.VARIANTS): the writer's exact settings
        # (crf 23; a lone frame is quantised harder than the training frames were), the crf at which a lone frame is
        # quantised like the training frames, and lossless (the colour conversion and chroma subsampling alone)
        for suffix, codec in (("_h264", {}), ("_h264s", dict(crf="start")), ("_h264c", dict(crf=0))):
            sets[f"{name}{suffix}"] = dict(rows=rows, instruction=instruction, images=files(rows, root, codec=codec),
                                           tokens=tokens, root=root, codec=codec)

    # ---- every training frame
    rows, offsets, start_states = [], [], {}
    for e in index["episodes"]:
        st = episode_states(train_dir, e["episode"])
        if len(st) != e["length"]:
            raise SystemExit(f"training episode {e['episode']}: {len(st)} states, {e['length']} frames in the index")
        start_states[e["episode"]] = st[0]
        offsets.append(len(rows))
        rows += [dict(id=f"train_{e['episode']:03d}_{f:04d}", episode=e["episode"], frame=f, length=e["length"],
                      left_out=e["episode"], state=st[f].tolist()) for f in range(e["length"])]

    def train_images(start: int = 0):
        for e, off in zip(index["episodes"], offsets):
            if off + e["length"] <= start:
                continue
            n = 0
            for f, frame in FR.video(video(e["episode"])):
                n += 1
                if off + f >= start:
                    yield rgb(frame)
            if n != e["length"]:
                raise RuntimeError(f"{video(e['episode'])}: decoded {n} frames, expected {e['length']}")

    sets["train"] = dict(rows=rows, instruction=instruction, images=train_images, tokens=False, sources=videos)
    start_state = np.median(np.stack([start_states[e["episode"]] for e in index["episodes"]]), axis=0)

    # ---- frame 0 of every training episode
    rows0 = [dict(id=f"trainstart_{e['episode']:03d}", episode=e["episode"], frame=0, left_out=e["episode"],
                  state=exact(start_states[e["episode"]], e["state"], f"training episode {e['episode']}"),
                  faucet_rel=e["faucet_rel"]) for e in index["episodes"]]
    sets["trainstart"] = dict(rows=rows0, instruction=instruction, tokens=True, sources=videos, images=lambda start=0: (
        rgb(FR.frame(video(r["episode"]), 0)) for r in rows0[start:]))

    # ---- experiment 3's grid
    r3 = exp3 / "exp3_indist_signal"
    plan = json.loads((exp3 / "prep" / "plan.json").read_text())
    meta3 = json.loads((r3 / "meta.json").read_text())
    by_id = {q["id"]: q for q in plan["poses"]}
    grid_state = exact(start_states[plan["episode"]], meta3["anchor"]["state"], "experiment 3's anchor")
    grid = []
    for r in (json.loads(line) for line in open(r3 / "index.jsonl")):
        if r["kind"] == "recorded":
            continue
        q = by_id[r["id"]]
        grid.append(dict(id=f"grid_{r['id']:03d}", file=r["file"], kind=r["kind"], left_out=plan["episode"],
                         state=grid_state,
                         **{k: q[k] for k in ("fx", "fy", "dyaw", "faucet_rel", "pos_dist", "yaw_dist_deg", "pose_dist",
                                              "in_range", "faucet_dist")}))
    sets["grid"] = dict(rows=grid, instruction=instruction, images=files(grid, r3), tokens=True, root=r3)
    with_codec("grid", grid, r3, True)
    # the same renders with 16 observations per VLM pass: how much does a different batch size move the score?
    sets["grid_b16"] = dict(rows=grid, instruction=instruction, images=files(grid, r3), tokens=False, vlm_batch=16, root=r3)

    # ---- the new renders
    plan_file, rindex = out / "prep" / "render_plan.json", out / "renders" / "index.jsonl"
    if plan_file.exists() and rindex.exists():
        from prepare_llmd import model_state, read_raw  # the state of a raw teleop frame, as the train pack has it

        rplan = json.loads(plan_file.read_text())
        raw_dir = Path(index["raw_dir"])
        done = {r["id"]: r for r in (json.loads(line) for line in open(rindex))}
        by_set: dict = {k: [] for k in RENDER_SETS}
        for sc in rplan["scenes"]:
            for it in sc["items"]:
                if it["id"] not in done:
                    continue
                row = {k: v for k, v in it.items() if k not in ("base_pose", "joints")}
                full = (start_states[sc["episode"]] if sc["source"] == "train"
                        else model_state(read_raw(raw_dir / it["raw"]), it["raw_frame"]))
                row.update(file=done[it["id"]]["file"], scene=sc["scene"], recorded_gap=done[it["id"]].get("recorded_gap"),
                           left_out=sc["episode"] if sc["source"] == "train" else None,
                           state=exact(full, it["state"], it["id"]))
                if it["set"] == "start":
                    row["episode"] = sc["episode"]
                by_set[it["set"]].append(row)
        for k, rws in by_set.items():
            if rws:
                sets[k] = dict(rows=rws, instruction=instruction, images=files(rws, out / "renders"), tokens=True,
                               root=out / "renders")
                with_codec(k, rws, out / "renders", True)

    # ---- earlier closed-loop trials
    trials = out / "trials" / "trials.jsonl"
    if trials.exists():
        allrows = [json.loads(line) for line in open(trials)]
        for name in ("traj", "target"):
            # a trial in a training episode's scene carries that episode as left_out (index_trials.py)
            rws = [dict(r, left_out=r.get("left_out", None)) for r in allrows if r["set"] == name]
            if not rws:
                continue
            prompts = sorted({r["instruction"] for r in rws})
            if len(prompts) != 1:
                raise SystemExit(f"trial set {name}: more than one prompt {prompts}")
            sets[name] = dict(rows=rws, instruction=prompts[0], images=files(rws, out / "trials"), tokens=False,
                              root=out / "trials")
            if prompts[0] != instruction:  # the same images under the checkpoint's own prompt
                sets[f"{name}_tf"] = dict(rows=rws, instruction=instruction, images=files(rws, out / "trials"),
                                          tokens=False, root=out / "trials")
            # the same images with the median training-start state: every closed-loop first state sits off the
            # training starts' states (index_trials.py), so this separates what the image does from what the state does
            rws_ms = [dict(r, state=start_state.tolist(), state_source="median of the training starts") for r in rws]
            sets[f"{name}_ms"] = dict(rows=rws_ms, instruction=prompts[0], images=files(rws_ms, out / "trials"), tokens=False,
                                      root=out / "trials")

    # ---- other tasks' frames, with a faucet start state
    other = []
    for pack in sorted(Path(index["train_dir"]).parent.iterdir()):
        info_f = pack / "meta" / "info.json"
        if pack.name == Path(index["train_dir"]).name or not info_f.exists():
            continue
        info = json.loads(info_f.read_text())
        vkey = next((k for k, v in info["features"].items() if v.get("dtype") == "video"), None)
        eps = [json.loads(line) for line in (pack / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
        if vkey is None:
            continue
        for e in eps[:5]:
            ep, n = int(e["episode_index"]), int(e["length"])
            path = pack / info["video_path"].format(episode_chunk=ep // info.get("chunks_size", 1000),
                                                    video_key=vkey, episode_index=ep)
            if not path.exists():
                path = pack / info["video_path"].format(episode_chunk=ep // info.get("chunks_size", 1000),
                                                        video_key=vkey.split(".")[-1], episode_index=ep)
            if not path.exists():
                continue
            for f in sorted({0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1}):
                other.append(dict(id=f"other_{pack.name}_{ep:03d}_{f:04d}", pack=pack.name, video=str(path), frame=f,
                                  left_out=None, state=start_state.tolist()))
    if other:
        sets["other"] = dict(rows=other, instruction=instruction, tokens=False,
                             sources=sorted({Path(r["video"]) for r in other}), images=lambda start=0: (
            rgb(FR.frame(Path(r["video"]), r["frame"])) for r in other[start:]))

    # ---- image and state from different moments of the same training episode
    swap = []
    for e in index["episodes"]:
        late = int(0.8 * (e["length"] - 1))
        st = episode_states(train_dir, e["episode"])
        for kind, fi, fs in (("start_image_late_state", 0, late), ("late_image_start_state", late, 0)):
            swap.append(dict(id=f"swap_{e['episode']:03d}_{kind}", kind=kind, episode=e["episode"], image_frame=fi,
                             state_frame=fs, left_out=e["episode"], state=st[fs].tolist()))
    sets["swap"] = dict(rows=swap, instruction=instruction, tokens=False, sources=videos, images=lambda start=0: (
        rgb(FR.frame(video(r["episode"]), r["image_frame"])) for r in swap[start:]))
    return sets


def fingerprint(P: "Psi0Features", spec: dict, rows: list[dict], run: dict) -> str:
    """What a set's features depend on: its rows (ids, states, labels), its image files or videos (path, size,
    modification time), the prompt, the round-trip settings, the batch size of the VLM pass and the checkpoint. A
    finished set is only skipped when this still matches. A file is named by its real path, relative to this
    checkout when it lies inside it, so that neither the working directory nor the spelling of --out / --exp3
    (relative in a call by hand, absolute in run_llmd_indist.sh) changes the fingerprint."""
    repo = HERE.parents[2]

    def named(f: Path) -> str:
        p = f.resolve()
        return p.relative_to(repo).as_posix() if p.is_relative_to(repo) else str(p)

    src = spec.get("sources") or [Path(spec["root"]) / r["file"] for r in rows]
    files = [(named(f), f.stat().st_size, f.stat().st_mtime_ns) for f in map(Path, src)]
    blob = json.dumps(dict(rows=rows, files=files, instruction=spec["instruction"], codec=spec.get("codec"),
                           vlm_batch=int(spec.get("vlm_batch", P.vlm_batch)), run=run), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def batches(n: int, start: int, size: int):
    """(first row, rows in the batch): the last batch is padded with repeats of its last row so every forward pass
    has the same batch size."""
    for a in range(start, n, size):
        idx = list(range(a, min(a + size, n)))
        yield a, idx + [idx[-1]] * (size - len(idx))


# ------------------------------------------------------------------------------------------------ phase A: the VLM
def phase_a(P: Psi0Features, name: str, spec: dict, dest: Path, limit: int, redo: bool, run: dict) -> dict:
    rows = spec["rows"][:limit or None]
    n, B = len(rows), int(spec.get("vlm_batch", P.vlm_batch))
    fp = fingerprint(P, spec, rows, run)
    prog_f = dest / "progress_a.json"
    prog = json.loads(prog_f.read_text()) if prog_f.exists() and not redo else dict(done=0)
    if prog.get("fingerprint") != fp:  # other rows, images, prompt or checkpoint: start over
        prog = dict(done=0)
    if prog.get("done") == n and prog.get("n") == n:
        return prog
    dest.mkdir(parents=True, exist_ok=True)
    if prog.get("n") != n:
        # a new start: say so on disk before the set's files are overwritten, so that an interrupted pass does not
        # leave the "finished" mark of the features it replaces (phase B alone would then keep the old ones)
        prog = dict(done=0, n=n, fingerprint=fp)
        prog_f.write_text(json.dumps(prog))
    with open(dest / "obs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.save(dest / "state.npy", np.array([r["state"] for r in rows], dtype=np.float32))
    start = (prog["done"] // B) * B if prog.get("n") == n else 0
    images = spec["images"](start)
    vlm_mm = tok_mm = None
    t0, ids0 = time.perf_counter(), None
    for a, idx in batches(n, start, B):
        fresh = [next(images) for _ in range(len(set(idx)))]
        pils = [P.img_t(im) for im in fresh]
        pils += [pils[-1]] * (B - len(pils))
        hs, inp = P.vlm(pils, spec["instruction"])
        ids = inp["input_ids"].cpu().numpy()
        if not bool(inp["attention_mask"].all()):
            raise RuntimeError(f"{name}: a batch was padded; every observation of a set must have the same token layout")
        if ids0 is None:
            ids0 = ids[0]
            is_img = ids0 == P.image_pad_id
            N, D = hs[-1].shape[1:]
            mode = "r+" if start > 0 and (dest / "vlm.npy").exists() else "w+"
            vlm_mm = np.lib.format.open_memmap(dest / "vlm.npy", mode=mode, dtype=np.int16, shape=(n, N, D))
            if spec["tokens"]:
                tok_mm = np.lib.format.open_memmap(dest / "tok8.npy", mode=mode, dtype=np.float16,
                                                   shape=(n, int(is_img.sum()), D))
            grid_thw = inp["image_grid_thw"][0].tolist()
            (dest / "layout.json").write_text(json.dumps(dict(
                n=n, n_tokens=int(N), dim=int(D), is_img=is_img.tolist(), img_grid=[grid_thw[1] // 2, grid_thw[2] // 2],
                instruction=spec["instruction"], vlm_batch=B, tokens=P.proc.tokenizer.convert_ids_to_tokens(ids0.tolist()),
                vlm_input_size=list(pils[0].size[::-1]), source_image_size=list(fresh[0].size[::-1])), indent=1))
        if not (ids == ids0[None]).all():
            raise RuntimeError(f"{name}: rows {a}..: the token layout differs from the set's first observation")
        k = len(set(idx))
        vlm_mm[a:a + k] = bits(hs[-1])[:k]
        if tok_mm is not None:
            tok_mm[a:a + k] = hs[TOKEN_LAYER][:k][:, torch.as_tensor(is_img, device=P.dev)].float().cpu().numpy()
        done = a + k
        if done % 512 < B or done == n:
            vlm_mm.flush()
            if tok_mm is not None:
                tok_mm.flush()
            prog = dict(done=done, n=n, fingerprint=fp)
            prog_f.write_text(json.dumps(prog))
            rate = (time.perf_counter() - t0) / max(done - start, 1)
            print(f"[{name}] A {done}/{n}  {1000 * rate:.1f} ms per observation", flush=True)
    del vlm_mm, tok_mm
    return prog


# ------------------------------------------------------------------------------------------------ phase B: the header
def phase_b(P: Psi0Features, name: str, dest: Path, seeds: list[int], grid_episode: int, redo: bool) -> None:
    rows = [json.loads(line) for line in open(dest / "obs.jsonl")]
    n, B, dev = len(rows), P.batch, P.dev
    done_f = dest / "progress_b.json"
    # (the fingerprint of phase A covers the rows and their states; a set whose phase A was redone gets phase B again)
    prog_a = json.loads((dest / "progress_a.json").read_text())
    if prog_a.get("done") != n or prog_a.get("n") != n or "fingerprint" not in prog_a:
        raise SystemExit(f"{name}: phase A is not finished for these observations ({prog_a}); run it first (--phase ab)")
    # (the batch size of the fp32 passes is part of it: the features depend on it at the rounding level, and the
    # per-observation noise of the ablations is drawn batch by batch)
    want = dict(n=n, seeds=seeds, ablations=list(ABLATIONS), batch=B, fingerprint=prog_a["fingerprint"])
    if done_f.exists() and not redo and json.loads(done_f.read_text()) == want:
        return
    vlm_mm = np.load(dest / "vlm.npy", mmap_mode="r")
    states = np.load(dest / "state.npy")
    assert vlm_mm.shape[0] == n == len(states)
    is_train = name == "train"
    T, D = P.Tp, P.hdr.action_proj_out.linear.in_features
    mk = lambda fname: np.lib.format.open_memmap(dest / fname, mode="w+", dtype=np.float32, shape=(n, T, D))
    out = {f"h_s{k}": mk(f"h_s{k}.npy") for k in seeds}
    out["h_s0_bf16"] = mk("h_s0_bf16.npy")
    if is_train:  # the ablations of the training frames: moments only, plus the grid scene's episode itself
        mom = {a: dict(n=0, s=np.zeros((T, D)), P=np.zeros((T, D, D)), buf=[]) for a in ABLATIONS}

        def flush(m: dict) -> None:  # float64 on the CPU; a few hundred rows at a time keep the products in BLAS
            if m["buf"]:
                X = np.concatenate(m["buf"]).astype(np.float64)
                m["buf"].clear()
                m["n"] += len(X)
                m["s"] += X.sum(0)
                for tk in range(T):
                    m["P"][tk] += X[:, tk].T @ X[:, tk]

        ep_rows = np.array([i for i, r in enumerate(rows) if r["episode"] == grid_episode])
        ep_feat = {a: np.zeros((len(ep_rows), T, D), np.float32) for a in ABLATIONS}
        ep_pos = {int(i): j for j, i in enumerate(ep_rows)}
    else:
        for a in ABLATIONS:
            out[f"f_{a}"] = mk(f"f_{a}.npy")
    a0 = {k: P.noise(k) for k in seeds}
    g = torch.Generator("cpu").manual_seed(zlib.crc32(name.encode()) % (2 ** 31))  # per-observation noise / timestep
    t1000 = torch.full((B,), T_NOISE, device=dev)
    t500 = torch.full((B,), 500.0, device=dev)
    t0 = time.perf_counter()
    for a, idx in batches(n, 0, B):
        k = len(set(idx))
        views = unbits(vlm_mm[idx[0]:idx[0] + k], dev)[:, None]
        if k < B:
            views = torch.cat([views, views[-1:].expand(B - k, -1, -1, -1)])
        st = P.norm_state(states[idx])
        for s in seeds:
            out[f"h_s{s}"][a:a + k] = P.header(views, st, a0[s].expand(B, -1, -1), t1000)[0][:k].cpu().numpy()
        # as served: bf16 autocast, one observation per pass
        out["h_s0_bf16"][a:a + k] = torch.cat([P.header(views[j:j + 1], st[j:j + 1], a0[0], t1000[:1], bf16=True)[0]
                                              for j in range(k)]).cpu().numpy()
        # per-observation draws are made for the k real rows only, so the stream does not depend on the padding
        fresh = torch.randn(k, P.Tp, P.Da, generator=g).to(dev)
        rnd = torch.randn(k, P.Tp, P.Da, generator=g).to(dev)
        trand = (torch.rand(k, generator=g) * 1000.0).to(dev)
        pad = lambda x: torch.cat([x, x[-1:].expand(B - k, *x.shape[1:])]) if k < B else x
        abl = dict(fresh=P.header(views, st, pad(fresh), t1000)[1][:k],
                   t500=P.header(views, st, a0[0].expand(B, -1, -1), t500)[1][:k],
                   rand=P.header(views, st, pad(rnd), pad(trand))[1][:k])
        for nm, F in abl.items():
            if is_train:
                Fc = F.cpu().numpy()
                mom[nm]["buf"].append(Fc)
                if len(mom[nm]["buf"]) >= 32:
                    flush(mom[nm])
                for j, i in enumerate(idx[:k]):
                    if i in ep_pos:
                        ep_feat[nm][ep_pos[i]] = Fc[j]
            else:
                out[f"f_{nm}"][a:a + k] = F.cpu().numpy()
        if (a // B) % 200 == 0 or a + k == n:
            print(f"[{name}] B {a + k}/{n}  {1000 * (time.perf_counter() - t0) / (a + k):.1f} ms per observation", flush=True)
    for m in out.values():
        m.flush()
    if is_train:
        for nm in ABLATIONS:
            flush(mom[nm])
            np.savez(dest / f"mom_{nm}.npz", n=mom[nm]["n"], s=mom[nm]["s"], P=mom[nm]["P"])
            np.save(dest / f"f_{nm}_ep.npy", ep_feat[nm])
        np.save(dest / "ep_rows.npy", ep_rows)
    done_f.write_text(json.dumps(want))


# ------------------------------------------------------------------------------------------------ the hook check
def hook_check(P: Psi0Features, sets: dict, out: Path, feats: Path, seeds: list[int], exp3: Path, index: dict,
               FR: Frames) -> dict:
    """What the feature is, measured. Raises if a hard assertion fails."""
    dev, B, hdr, model = P.dev, P.batch, P.hdr, P.model
    res: dict = dict(hard=[], notes=[])

    def hard(name: str, ok: bool, detail: str) -> None:
        res["hard"].append(dict(check=name, ok=bool(ok), detail=detail))
        print(f"[check] {'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)

    # ---- the architecture this script assumes
    po = hdr.action_proj_out
    arch = dict(final_layer_norm=bool(po.final_layer_norm), state_as_action_token=bool(getattr(hdr, "state_as_action_token", False)),
                combined_temb=bool(getattr(hdr, "combined_temb", False)), vlm_layer_indices=model.vlm_layer_indices,
                blocks=len(hdr.transformer_blocks), feature_dim=po.linear.in_features, action_dim=po.linear.out_features,
                action_tokens=P.Tp, training=bool(model.training))
    res["architecture"] = arch
    hard("architecture", not arch["final_layer_norm"] and not arch["state_as_action_token"] and not arch["combined_temb"]
         and arch["vlm_layer_indices"] is None and not arch["training"],
         f"legacy head without final LayerNorm (f = h * scale(t) + shift(t)), last VLM layer only, eval mode: {arch}")

    # ---- timestep 1000 is the pure-noise end of the flow (the paper's t = 0)
    sch = model.noise_scheduler
    x = torch.randn(2, P.Tp, P.Da, device=dev)
    nz = torch.randn(2, P.Tp, P.Da, device=dev)
    sch.set_timesteps(10)
    steps10 = [float(v) for v in sch.timesteps]
    sig10 = [float(v) for v in sch.sigmas]
    mixed = sch.scale_noise(sample=x, timestep=torch.full((2,), T_NOISE, device=dev), noise=nz)
    half = sch.scale_noise(sample=x, timestep=torch.full((2,), sch.timesteps[4].item(), device=dev), noise=nz)
    s4 = sig10[4]
    res["flow"] = dict(
        scheduler=type(sch).__name__, num_train_timesteps=int(sch.config.num_train_timesteps), shift=float(sch.config.shift),
        inference_timesteps=steps10, inference_sigmas=sig10,
        scale_noise_at_1000_equals_noise=bool(torch.equal(mixed, nz)),
        scale_noise_mid_matches_formula=float((half - ((1 - s4) * x + s4 * nz)).abs().max()),
        training_rule="src/psi/trainers/finetune.py: sigmas = rand(bsz); timesteps = sigmas * train_diffusion_steps; "
                      "noisy_actions = (1 - sigma) * actions + sigma * noise  -> sigma = 1 (timestep 1000) is 100 % noise, "
                      "sigma = 0 is the clean action",
        paper_convention="VLA-FAIL f*(o) = f(o, t=0, a0*): t = 0 is the prior-noise end of the flow (pi0 convention, "
                         "A^tau = tau * A + (1 - tau) * eps)")
    hard("timestep 1000 = pure prior noise",
         sch.config.shift == 1.0 and steps10[0] == T_NOISE and sig10[0] == 1.0 and res["flow"]["scale_noise_at_1000_equals_noise"]
         and res["flow"]["scale_noise_mid_matches_formula"] < 1e-5,
         f"shift {sch.config.shift}, first inference timestep {steps10[0]} with sigma {sig10[0]}, "
         f"scale_noise(x, 1000, n) == n: {res['flow']['scale_noise_at_1000_equals_noise']}")

    # ---- check observations: two training frames and two renders
    instr = sets["grid"]["instruction"]
    grid_rows = sets["grid"]["rows"]
    far_i = int(np.argmax([r["pose_dist"] for r in grid_rows]))
    r3 = exp3 / "exp3_indist_signal"
    train_dir, video = train_meta(index)
    read_video_frame = FR.frame
    ep0 = index["episodes"][0]
    st0 = episode_states(train_dir, ep0["episode"])
    mid = ep0["length"] // 2
    obs = [("train ep0 frame 0", Image.fromarray(read_video_frame(video(ep0["episode"]), 0)), st0[0]),
           (f"train ep0 frame {mid}", Image.fromarray(read_video_frame(video(ep0["episode"]), mid)), st0[mid]),
           ("grid render at the start", Image.open(r3 / grid_rows[0]["file"]).convert("RGB"), np.asarray(grid_rows[0]["state"], np.float32)),
           ("grid render far away", Image.open(r3 / grid_rows[far_i]["file"]).convert("RGB"), np.asarray(grid_rows[far_i]["state"], np.float32))]
    names = [o[0] for o in obs]
    pils = [P.img_t(o[1]) for o in obs]
    states = np.stack([o[2] for o in obs])
    reps = B // len(obs)
    order = [i for i in range(len(obs)) for _ in range(reps)] + [len(obs) - 1] * (B - reps * len(obs))
    hs, inp = P.vlm([pils[i] for i in order], instr)
    first = [order.index(i) for i in range(len(obs))]
    views = hs[-1][:, None]
    st = P.norm_state(states[order])
    res["vlm"] = dict(dtype=str(hs[-1].dtype), tokens=int(hs[-1].shape[1]), dim=int(hs[-1].shape[2]),
                      max_abs=float(hs[-1].float().abs().max()))
    hard("the VLM context is bf16 and its bit cache is exact", hs[-1].dtype == torch.bfloat16
         and torch.equal(unbits(bits(hs[-1]), dev), hs[-1]), f"{res['vlm']}")
    one = P.norm_state(states[:1])
    hard("state normalisation is row-wise", float((P.norm_state(states)[:1] - one).abs().max()) == 0.0,
         "normalising a batch equals normalising each row")

    a0 = P.noise(seeds[0]).expand(B, -1, -1)
    t = torch.full((B,), T_NOISE, device=dev)
    h, f, o = P.header(views, st, a0, t)
    lin = po.linear
    with torch.inference_mode():
        relin = lin(f)
        temb = hdr.time_ins_embed(t)
        shift, scale = po.adaLN_modulation(temb[:, None] if temb.dim() == 2 else temb).chunk(2, dim=-1)
        temb5 = hdr.time_ins_embed(torch.full((B,), 500.0, device=dev))
        shift5, scale5 = po.adaLN_modulation(temb5[:, None] if temb5.dim() == 2 else temb5).chunk(2, dim=-1)
    recon = h * scale + shift
    sc, sh = scale[0, 0].double().cpu().numpy(), shift[0, 0].double().cpu().numpy()
    fa = f[first].double().cpu().numpy()  # (4, 30, 1536)
    res["hook"] = dict(
        h_shape=list(h.shape), f_shape=list(f.shape), out_shape=list(o.shape),
        linear_of_f_vs_output_max_abs=float((relin - o).abs().max()),
        f_vs_h_scale_shift_max_abs=float((recon - f).abs().max()), f_abs_max=float(f.abs().max()),
        scale_abs=dict(median=float(np.median(np.abs(sc))), min=float(np.abs(sc).min()), max=float(np.abs(sc).max()),
                       below_1e_2=int((np.abs(sc) < 1e-2).sum()), below_1e_3=int((np.abs(sc) < 1e-3).sum())),
        shift_abs=dict(median=float(np.median(np.abs(sh))), max=float(np.abs(sh).max())),
        # how much of f is the constant shift: a float32 file of f would lose this many digits of its variation
        shift_over_spread_median=float(np.median(np.abs(sh) / (fa.reshape(-1, fa.shape[-1]).std(0) + 1e-30))),
        f_channel_std=dict(median=float(np.median(fa.reshape(-1, fa.shape[-1]).std(0))),
                           min=float(fa.reshape(-1, fa.shape[-1]).std(0).min())))
    hard("linear(f) is the header output", res["hook"]["linear_of_f_vs_output_max_abs"] <= 1e-5,
         f"max abs difference {res['hook']['linear_of_f_vs_output_max_abs']:.2e} (the hooked tensor is the final layer's input)")
    hard("f = h * scale(t) + shift(t)", res["hook"]["f_vs_h_scale_shift_max_abs"] <= 1e-4 * max(res["hook"]["f_abs_max"], 1.0),
         f"max abs difference {res['hook']['f_vs_h_scale_shift_max_abs']:.2e} against |f| up to {res['hook']['f_abs_max']:.2f}")
    feats.mkdir(parents=True, exist_ok=True)
    np.savez(feats / "adaln.npz", scale_1000=sc, shift_1000=sh, scale_500=scale5[0, 0].double().cpu().numpy(),
             shift_500=shift5[0, 0].double().cpu().numpy())
    for k in seeds:
        np.save(feats / f"noise_seed{k}.npy", P.noise(k).cpu().numpy())

    # ---- the served policy: what does its first header call receive, and do we reproduce its feature?
    served, solo = [], []
    t1 = t[:1]
    for i, nm in enumerate(names):
        P.calls = []
        torch.manual_seed(1234 + i)
        model.predict_action(observations=[[pils[i]]], states=P.norm_state(states[i:i + 1]), instructions=[instr],
                             num_inference_steps=10, traj2ds=None)
        calls, P.calls = P.calls, None
        torch.manual_seed(1234 + i)
        prior = torch.randn(1, P.Tp, P.Da, device=dev)
        hs1, _ = P.vlm([pils[i]], instr)  # one observation per pass, as served
        st1 = P.norm_state(states[i:i + 1])
        mine = P.header(hs1[-1][:, None], st1, calls[0]["sample"], t1, bf16=True)[1]
        solo.append((hs1[-1], st1))
        served.append(dict(obs=nm, header_calls=len(calls), timesteps=[float(c["timestep"][0]) for c in calls],
                           first_sample_is_the_seeded_randn=bool(torch.equal(calls[0]["sample"], prior)),
                           first_sample_mean=float(calls[0]["sample"].mean()), first_sample_std=float(calls[0]["sample"].std()),
                           ours_vs_served_f_max_abs=float((mine - calls[0]["f"]).abs().max()),
                           f_abs_max=float(calls[0]["f"].abs().max())))
    res["served"] = served
    hard("the served first step runs at timestep 1000 on the untouched prior sample",
         all(s["header_calls"] == 10 and s["timesteps"] == steps10 and s["first_sample_is_the_seeded_randn"] for s in served),
         f"header calls {[s['header_calls'] for s in served]}, first timestep {[s['timesteps'][0] for s in served]}, "
         f"first sample == torch.randn of the same seed: {[s['first_sample_is_the_seeded_randn'] for s in served]}")
    worst = max(s["ours_vs_served_f_max_abs"] for s in served)
    hard("our single pass reproduces the served first step's feature", worst <= 1e-6,
         f"max abs difference {worst:.2e} (one observation per pass, bf16 autocast, the served noise)")

    # ---- numerics: precision and batch size
    rel = lambda x, y: float((x.float() - y.float()).pow(2).mean().sqrt() / y.float().pow(2).mean().sqrt())
    v1, st1 = solo[0][0][:, None], solo[0][1]
    h1 = P.header(v1, st1, a0[:1], t1)[0]
    h1_again = P.header(v1, st1, a0[:1], t1)[0]
    h1_in16 = P.header(v1.expand(B, -1, -1, -1), st1.expand(B, -1, -1), a0, t)[0][:1]
    h1_bf = P.header(v1, st1, a0[:1], t1, bf16=True)[0]
    h_all1 = torch.cat([P.header(v[:, None], s_, a0[:1], t1)[0] for v, s_ in solo])
    spread = h_all1.std(0).mean().item()  # typical difference between observations
    same_obs = max(float((h[j] - h[first[i]]).abs().max()) for i in range(len(obs)) for j in range(B) if order[j] == i)
    res["numerics"] = dict(
        between_observation_std=spread,
        repeat_max_abs=float((h1 - h1_again).abs().max()),
        vlm_batch1_vs_batch16_rel_rms=[rel(hs[-1][first[i]], solo[i][0][0]) for i in range(len(obs))],
        h_from_vlm_batch1_vs_batch16_rel_rms=[rel(h[first[i]], h_all1[i]) for i in range(len(obs))],
        fp32_header_batch1_vs_batch16_max_abs=float((h1 - h1_in16).abs().max()),
        fp32_vs_bf16_header_rel_rms=rel(h1_bf, h1),
        same_batch_other_slot_max_abs=same_obs)
    hard("the header is repeatable", res["numerics"]["repeat_max_abs"] == 0.0,
         f"two passes differ by {res['numerics']['repeat_max_abs']:.2e}")
    res["notes"].append(
        f"h differs between observations by about {spread:.3f} (std). The bf16 VLM output depends on the batch size: "
        f"batch 1 vs 16 differs by {max(res['numerics']['vlm_batch1_vs_batch16_rel_rms']):.1e} (relative rms), which moves h by "
        f"{max(res['numerics']['h_from_vlm_batch1_vs_batch16_rel_rms']):.1e}; so every VLM pass here has the served batch size 1. "
        f"The fp32 action expert at batch 1 vs 16: {res['numerics']['fp32_header_batch1_vs_batch16_max_abs']:.1e} (max abs); "
        f"fp32 vs bf16 action expert: {res['numerics']['fp32_vs_bf16_header_rel_rms']:.1e} (relative rms).")

    # ---- decoding: our frame 0 of episode 0 against experiment 3's (ffmpeg CLI) copy
    rec = np.asarray(Image.open(r3 / "img" / "00000_recorded.png").convert("RGB")).astype(np.int16)
    ours = np.asarray(obs[0][1]).astype(np.int16)
    res["decode"] = dict(vs_exp3_recorded_max_abs=int(np.abs(rec - ours).max()), vs_exp3_recorded_mean_abs=float(np.abs(rec - ours).mean()))
    res["notes"].append(f"training frame 0 of episode 0, our decoder vs experiment 3's ffmpeg decode: max abs "
                        f"{res['decode']['vs_exp3_recorded_max_abs']}, mean abs {res['decode']['vs_exp3_recorded_mean_abs']:.3f} (0-255)")
    res["ok"] = all(c["ok"] for c in res["hard"])
    (out / "checks").mkdir(parents=True, exist_ok=True)
    (out / "checks" / "hook_check.json").write_text(json.dumps(res, indent=1))
    if not res["ok"]:
        raise SystemExit("the hook check failed, see checks/hook_check.json")
    return res


def main() -> None:
    args = parse_args()
    if args.watchdog > 0:
        import faulthandler

        faulthandler.dump_traceback_later(args.watchdog, repeat=True)
    out, exp3 = Path(args.out), Path(args.exp3)
    feats = out / f"feats{args.suffix}"
    index = json.loads((exp3 / "prep" / "train_index.json").read_text())
    plan = json.loads((exp3 / "prep" / "plan.json").read_text())
    seeds = [int(v) for v in args.seeds.split(",")]
    instruction = index["instruction"].lower()  # the server lowercases too
    FR = Frames()
    sets = build_sets(args, index, instruction, FR)
    names = list(sets) if args.sets == "all" else [s for s in args.sets.split(",") if s]
    missing = [s for s in names if s not in sets]
    if missing:
        raise SystemExit(f"observation sets without inputs: {missing} (have {list(sets)})")
    started = time.perf_counter()
    P = Psi0Features(Path(index["run_dir"]), index["ckpt_step"], args.device, args.batch, args.vlm_batch)
    hook_check(P, sets, out, feats, seeds, exp3, index, FR)
    if args.phase == "check":
        return
    run = dict(run_dir=index["run_dir"], ckpt_step=str(index["ckpt_step"]))
    (feats / "run.json").write_text(json.dumps(dict(
        run_dir=index["run_dir"], ckpt_step=index["ckpt_step"], instruction=instruction, seeds=seeds, batch=args.batch,
        vlm_batch=args.vlm_batch, task=index["task"],
        timestep=T_NOISE, token_layer=TOKEN_LAYER, grid_episode=plan["episode"], ablations=list(ABLATIONS)), indent=1))
    for name in names:
        dest = feats / name
        t0 = time.perf_counter()
        if "a" in args.phase:
            phase_a(P, name, sets[name], dest, args.limit, args.redo, run)
        if "b" in args.phase:
            phase_b(P, name, dest, seeds, plan["episode"], args.redo)
        print(f"[{name}] {len(sets[name]['rows'][:args.limit or None])} observations in {time.perf_counter() - t0:.0f} s", flush=True)
    FR.close()
    print(f"saved {feats} in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
