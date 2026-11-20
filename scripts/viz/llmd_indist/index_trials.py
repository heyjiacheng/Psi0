"""Index the earlier closed-loop trials whose outcome is known (the "trials" stage of the LLMD experiment).

Runs in Psi0's venv, CPU only, no simulator and no policy. Two earlier experiments ran the open-faucet checkpoint
closed-loop from many start conditions and kept the outcome; this stage collects, for every trial, the FIRST
observation the policy was asked about (image + 32-D state + prompt), so the detector can score it without a new
rollout. Only the two sweeps' report folders survive (the runners delete the rest), so everything comes from them:

  traj    trajectory sweep (report/results.json, report/img/t###.jpg, report/videos/t###.mp4): training scenes,
          prompt "turn the faucet". Standing starts (the spawn as recorded, moved or turned) and mid-demo takeovers
          (the episode's recorded actions replayed up to a path point, then the policy's first request).
            image  the eval saved the first-request image shrunk to 320x180 (cv2.INTER_AREA); the report kept a JPEG
                   of that (quality 82). Copied unchanged. The same instant is also frame <policy_from> of the trial
                   video (640x360, with the report's "policy" label burnt in); kept as a second form (alt_file).
            state  the 32-D state the agent sent, recorded by the eval (5 decimals).
  target  target pose sweep (report/results.json, report/videos/t###.mp4): the faucet put on a 3 cm grid around
          the robot, 10 base scenes, prompt "move forward to the faucet and open it". Always a standing start.
            image  frame 0 of the trial video: SIMPLE's recorder restarts the video with the observation the policy
                   gets first, so frame 0 is that image after three lossy encodes (OpenCV mp4v, libx264, libx264
                   crf 30). Decoded (PyAV, else ffmpeg) and saved as PNG.
            state  not saved anywhere (the base scenes' parquet rows are all zero: scene-only episodes). The eval's
                   stand-up is repeatable: the trajectory sweep recorded the identical state in 77 of its 105
                   standing starts (the rest were moved 10 cm or turned; 15 of them touched the faucet at the first
                   request). That state is the proxy. A trial whose pelvis stood up exactly where those 77 did very
                   likely had it; the others are flagged (state_standup_nominal false).

Ground truth per trial, in experiment 3's convention (prepare_base_grid.py): where the faucet is in the pelvis frame
at the first request (faucet_rel), and how far that is outside the training starts' range (pose_dist, 0 inside).
These keys, and pelvis_world / faucet_world (x, y in m, yaw in rad, world frame), mean the same in both sets; the
other pose keys are each source's own log and differ (the trajectory sweep's "pelvis" is in the recorded spawn's frame
with the yaw in degrees, the target sweep's is the world pose in radians). For a mid-demo takeover pose_dist says how
far the robot has walked out of the range of training STARTS, not that the pose is unseen in training.

Neither image is the policy's exact input (first_obs_exact is false everywhere, image_note says why): the model takes
every second pixel of the 640x360 camera image (its nearest resize to 180x320); the JPEG is a 2x2 box mean of that
image, the video frame went through three encodes. The trajectory sweep has both forms of the same instant, which
measures what the form does to a score.

Written to --out:
  trials.jsonl     one line per trial, the trajectory sweep first, each in the source's trial order:
                     id (<set>_<the source's trial number>: traj_0007 is the trajectory sweep's t007), set,
                     source_label, file, image_source, first_obs_exact, image_note, image_size,
                     [alt_file, alt_image_source, alt_image_note, alt_frame, alt_frame_type, alt_label_box_xyxy,
                      alt_vs_file_mad: mean |file - alt_file| per 255 at the model's input size, label left out] (traj)
                     [frame_type]                                                                              (target)
                     state, state_source, [state_note, state_standup_nominal]                                  (target)
                     instruction, success, outcome, standing_start, condition, scene,
                     left_out (the training episode whose scene the trial ran in, else null), pose, source_dir
  img/<id>.jpg     traj: the report's takeover JPEG, byte for byte
  img/<id>.png     target: frame 0 of the trial video
  img_alt/<id>.png traj: the takeover frame of the trial video (unless --no-traj-video-frames)
  meta.json        the sources, the counts per condition, the state proxy and how much start states vary, the checks
trials.jsonl is written last and in one step (the runner takes its presence as "this stage is done"); a re-run
overwrites everything and gives the same bytes.
The run ends with a summary table and its own checks (both sweeps ran experiment 3's checkpoint, counts against the
sources' reports, every image opened, every state finite, both image forms of a trajectory-sweep trial show the same
picture); a failed check is printed and the exit status is 1.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

try:  # PyAV is in Psi0's venv; the ffmpeg binary is the fallback (same frames, checked on this data)
    import av
except ImportError:
    av = None

TRAJ_RUN = ".runs/trajectory_sweep/G1WholebodyOpenFaucetTeleop-v0_x5_seed0"
TARGET_RUN = ".runs/target_pose_sweep/G1WholebodyOpenFaucetTeleop-v0_grid3cm_x5_seed0"
CAM_SIZE = (640, 360)        # the head camera, and the trial videos
TAKEOVER_SIZE = (320, 180)   # the trajectory sweep's takeover image
STATE_DIM = 32
YAW_LEVER = 0.5              # m per rad in pose_dist, as in prepare_base_grid.py
LABEL_BOX = [6, 6, 78, 36]   # x0, y0, x1, y1 (px, 640x360) of the "policy" label in the trajectory sweep's videos
FORMS_MAD_MAX = 10.0         # per 255. Measured: a trial's own two image forms differ by 3.0 to 6.4, its JPEG and the
                             # next trial's video frame by 46 on average
NOMINAL_M, NOMINAL_RAD = 2e-4, 3.5e-4  # a stand-up counts as the usual one within 0.2 mm and 0.02 deg (4-decimal logs)
PICT_TYPES = {1: "I", 2: "P", 3: "B"}  # ffmpeg's AVPictureType
# the 32-D state, in the agent's order (simple.baselines.psi0_decoupled_wbc.STATE_SLICES + waist rpy + height command)
STATE_NAMES = ([f"left_hand_{j}" for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
               + [f"right_hand_{j}" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0",
                                               "middle_1")]
               + [f"{side}_{j}" for side in ("left", "right") for j in
                  ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
               + ["waist_roll", "waist_pitch", "waist_yaw", "base_height_cmd"])
TRAJ_GROUPS = ("training start", "moved 5 cm", "moved 10 cm", "turned 20°", "turned 40°",
               "1/3 of the way", "2/3 of the way", "reach point")
TARGET_GROUPS = ("inside the training range", "closer than every demo", "farther than every demo",
                 "demo distance, off to the side")

TRAJ_IMAGE = ("takeover JPEG 320x180 as saved by the trajectory sweep's report: the first-request image (640x360) "
              "shrunk with cv2.INTER_AREA, then JPEG quality 82 (4:2:0)")
TRAJ_NOTE = ("not the exact pixels: a 2x2 box mean and a JPEG of the 640x360 image the policy got; the model itself "
             "takes every second pixel of that image (nearest), and its resize is a no-op on this file")
TRAJ_ALT_IMAGE = ("frame {frame} ({kind}-frame) of the trial video 640x360: the same first-request image after three "
                  "lossy encodes (OpenCV mp4v, libx264 default, libx264 crf 30)")
TRAJ_ALT_NOTE = ("not the exact pixels: three lossy encodes, and the report burnt a 'policy' label into the top-left "
                 "corner (x 6..77, y 6..35 px) before the last one")
TARGET_IMAGE = ("frame 0 (I-frame) of the trial video 640x360: the first-request image after three lossy encodes "
                "(OpenCV mp4v, libx264 default, libx264 crf 30), decoded with {decoder}")
TARGET_NOTE = ("not the exact pixels: the right frame (the recorder restarts the video with the observation the policy "
               "gets first), but three lossy encodes blur fine texture")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--psi0-dir", required=True)
    p.add_argument("--simple-dir", required=True, help="to check the target sweep's base scenes (their states are zero)")
    p.add_argument("--exp3", required=True, help="experiment 3's directory (read-only): prep/train_index.json")
    p.add_argument("--out", required=True)
    p.add_argument("--traj-run", default="", help=f"trajectory sweep directory (default: <psi0-dir>/{TRAJ_RUN})")
    p.add_argument("--target-run", default="", help=f"target pose sweep directory (default: <psi0-dir>/{TARGET_RUN})")
    p.add_argument("--target-state", choices=("standup", "train_median"), default="standup",
                   help="the target sweep's state proxy: the eval's stand-up state as the trajectory sweep recorded it "
                        "(default), or the median of the training start states")
    p.add_argument("--traj-video-frames", action=argparse.BooleanOptionalAction, default=True,
                   help="also keep the takeover frame of each trajectory-sweep video (img_alt/)")
    p.add_argument("--limit", type=int, default=0, help="only the first N trials of each sweep (0 = all), for a quick look")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- geometry
def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_frame(origin, xy) -> list[float]:
    """xy in the frame of origin = (x, y, yaw): x ahead, y to the left."""
    c, s = math.cos(origin[2]), math.sin(origin[2])
    dx, dy = xy[0] - origin[0], xy[1] - origin[1]
    return [c * dx + s * dy, -s * dx + c * dy]


def from_frame(origin, rel) -> list[float]:
    c, s = math.cos(origin[2]), math.sin(origin[2])
    return [origin[0] + c * rel[0] - s * rel[1], origin[1] + s * rel[0] + c * rel[1]]


def box_dist(v: float, lo: float, hi: float) -> float:
    return max(lo - v, 0.0, v - hi)


def faucet_truth(pelvis, faucet, train_range: dict) -> dict:
    """Experiment 3's ground truth of a base pose: the faucet (x, y, yaw) seen from the pelvis (x, y, yaw), and how
    far that is outside the range of the 99 training starts (prepare_base_grid.py's faucet_rel, pose_dist)."""
    rel = to_frame(pelvis, faucet[:2])
    ryaw = wrap(faucet[2] - pelvis[2])
    d_pos = math.hypot(box_dist(rel[0], *train_range["dx"]), box_dist(rel[1], *train_range["dy"]))
    d_yaw = box_dist(ryaw, *train_range["dyaw"])
    world = lambda p: [round(p[0], 4), round(p[1], 4), round(wrap(p[2]), 5)]  # the same keys and units in both sets
    return dict(pelvis_world=world(pelvis), faucet_world=world(faucet),
                faucet_rel=[round(rel[0], 4), round(rel[1], 4), round(ryaw, 5)], pos_dist=round(d_pos, 4),
                yaw_dist_deg=round(math.degrees(d_yaw), 2), pose_dist=round(math.hypot(d_pos, YAW_LEVER * d_yaw), 4),
                in_train_start_range=bool(d_pos < 1e-9 and d_yaw < 1e-9))


# ---------------------------------------------------------------------------------------------- images
def decoder_name() -> str:
    return f"PyAV {av.__version__}" if av is not None else "ffmpeg"


def video_frame(path: Path, index: int) -> tuple[np.ndarray, str, int]:
    """Frame `index` (presentation order) of a video as RGB uint8, its type (I, P, B) and the video's frame count."""
    if av is not None:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if i == index:
                    kind = frame.pict_type  # an enum in older PyAV, ffmpeg's AVPictureType number in newer ones
                    kind = getattr(kind, "name", None) or PICT_TYPES.get(int(kind), str(kind))
                    return frame.to_ndarray(format="rgb24"), kind, int(stream.frames)
        raise RuntimeError(f"{path}: no frame {index}")
    # every frame is listed (presentation order): a listing cut short at `index` ends out of order where B-frames are
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_frames:frame=pict_type", "-of", "json", str(path)],
                           capture_output=True, check=True, text=True)
    info = json.loads(probe.stdout)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})", "-vsync",
                          "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    w, h = CAM_SIZE
    if len(raw) != h * w * 3:
        raise RuntimeError(f"{path}: frame {index} could not be decoded")
    return (np.frombuffer(raw, np.uint8).reshape(h, w, 3), info["frames"][index]["pict_type"],
            int(info["streams"][0]["nb_frames"]))


def save_png(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(np.ascontiguousarray(rgb)).save(path)  # PNG: lossless


def forms_mad(jpeg: Path, frame: np.ndarray) -> float:
    """Mean absolute difference (per 255) between a trial's two image forms as the model would get them: the 320x180
    JPEG as it is, the 640x360 video frame at every second pixel (the model's nearest resize); the label left out."""
    with Image.open(jpeg) as im:
        small = np.asarray(im.convert("RGB"), dtype=np.float64)
    keep = np.ones(frame.shape[:2], dtype=bool)
    keep[LABEL_BOX[1]:LABEL_BOX[3], LABEL_BOX[0]:LABEL_BOX[2]] = False
    return float(np.abs(small - frame[1::2, 1::2])[keep[1::2, 1::2]].mean())


# ---------------------------------------------------------------------------------------------- states
def standup_state(traj_trials: list[dict]) -> dict:
    """The eval's stand-up, from the trajectory sweep's standing starts: the first-request state most of them share
    (the stand-up is deterministic unless the robot touches something on the way up) and where the pelvis ends up
    relative to the spawn in those trials. Also how the others differ, to judge the proxy."""
    standing = [t for t in traj_trials if t["kind"] == "start"]
    states = np.array([t["takeover"]["state"] for t in standing], dtype=float)
    modal, n_modal = Counter(map(tuple, states.tolist())).most_common(1)[0]
    modal = np.array(modal)
    offsets = np.array([own_spawn_offset(t) for t in standing])
    same = np.all(states == modal, axis=1)
    offset = np.median(offsets[same], axis=0)
    nominal = np.array([is_nominal(o, offset) for o in offsets])
    dev = np.abs(states - modal).max(axis=1)
    return dict(state=modal.tolist(), n_identical=int(n_modal), n_standing=len(standing),
                pelvis_offset=[round(float(v), 6) for v in offset],
                n_nominal_standup=int(nominal.sum()), n_nominal_with_this_state=int((nominal & same).sum()),
                max_dev_nominal_standup=round(float(dev[nominal].max()), 5),
                n_other_standup=int((~nominal).sum()),
                median_dev_other_standup=round(float(np.median(dev[~nominal])), 5) if (~nominal).any() else 0.0,
                max_dev_other_standup=round(float(dev[~nominal].max()), 5) if (~nominal).any() else 0.0,
                range_all_standing=np.ptp(states, axis=0).round(5).tolist())


def own_spawn_offset(t: dict) -> list[float]:
    """Trajectory sweep, standing start: where the pelvis stood (x, y in m, yaw in rad) in the frame of the trial's
    own (moved or turned) spawn. The sweep logs it in the recorded spawn's frame, yaw in degrees."""
    x, y, yaw = t["pelvis_start"]
    a = math.radians(t["dyaw"])
    ex, ey = x - t["dx"], y - t["dy"]
    return [math.cos(a) * ex + math.sin(a) * ey, -math.sin(a) * ex + math.cos(a) * ey,
            wrap(math.radians(yaw - t["dyaw"]))]


def is_nominal(offset, nominal) -> bool:
    return bool(math.hypot(offset[0] - nominal[0], offset[1] - nominal[1]) <= NOMINAL_M
                and abs(wrap(offset[2] - nominal[2])) <= NOMINAL_RAD)


def train_start_states(index: dict) -> dict:
    s = np.array([e["state"] for e in index["episodes"]], dtype=float)
    return dict(n=len(s), median=np.median(s, axis=0).round(6).tolist(), min=s.min(axis=0).round(6).tolist(),
                max=s.max(axis=0).round(6).tolist(), range=np.ptp(s, axis=0).round(6).tolist())


def base_scene_states_zero(scenes: Path) -> bool | None:
    """The target sweep's base scenes are scene-only episodes: is every state in their parquet rows zero?"""
    files = sorted(scenes.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        return None
    import pandas as pd

    return bool(all(float(np.abs(np.stack(pd.read_parquet(f, columns=["observation.state"])["observation.state"]
                                          .values)).max()) == 0.0 for f in files))


# ---------------------------------------------------------------------------------------------- the two sweeps
def same_checkpoint(name: str, res: dict, index: dict) -> list[str]:
    """The trials are only worth re-scoring if the policy that ran them is the checkpoint the detector is fitted
    for (experiment 3's): the same task, run and step, as each sweep's report names them."""
    theirs = (res["task"], Path(res["checkpoint"]).name, str(res["ckpt_step"]))
    ours = (index["task"], Path(index["run_dir"]).name, str(index["ckpt_step"]))
    return [] if theirs == ours else [f"{name}: the sweep ran {theirs}, experiment 3 is about {ours}"]


def index_traj(report: Path, out: Path, index: dict, args) -> tuple[list[dict], dict, list[str]]:
    res = json.loads((report / "results.json").read_text())
    trials = sorted(res["trials_detail"], key=lambda t: t["trial"])
    eps = {e["episode"]: e for e in index["episodes"]}
    rows, problems = [], same_checkpoint("traj", res, index)
    for t in trials[:args.limit or None]:
        tid = f"traj_{t['trial']:04d}"
        to, ep = t["takeover"], eps[t["episode"]]
        if max(abs(a - b) for a, b in zip(t["spawn_rec"], ep["spawn"])) > 1e-3:
            problems.append(f"{tid}: its scene's spawn {t['spawn_rec']} is not training episode {t['episode']}'s "
                            f"{ep['spawn']}")
        src = report / t["img"]
        dst = out / "img" / f"{tid}.jpg"
        shutil.copyfile(src, dst)  # unchanged: a re-encode would add a second JPEG generation
        if not filecmp.cmp(src, dst, shallow=False):
            problems.append(f"{tid}: the copied JPEG differs from {src}")
        # the pelvis at the first request: logged in the recorded spawn's frame (yaw in deg); to the world frame
        spawn = t["spawn_rec"]
        px, py, pyaw_deg = to["pose"]
        pelvis = [*from_frame(spawn, [px, py]), spawn[2] + math.radians(pyaw_deg)]
        base = to_frame(ep["pelvis"], pelvis[:2])  # against this episode's training start pose, as experiment 3's fx, fy
        pose = dict(kind=t["kind"], cond=t["cond"], name=t["name"], spawn_offset=[t["dx"], t["dy"], t["dyaw"]],
                    point=t.get("point"), frame=t["frame"], pelvis=to["pose"],
                    base_offset=[round(base[0], 4), round(base[1], 4),
                                 round(math.degrees(wrap(pelvis[2] - ep["pelvis"][2])), 2)],
                    train_dpos_cm=t.get("train_dpos_cm"), train_dyaw_deg=t.get("train_dyaw_deg"),
                    **faucet_truth(pelvis, ep["faucet"], index["train_range"]))
        row = dict(id=tid, set="traj", source_label=t["label"], file=f"img/{tid}.jpg", image_source=TRAJ_IMAGE,
                   first_obs_exact=False, image_note=TRAJ_NOTE, image_size=list(TAKEOVER_SIZE))
        if args.traj_video_frames:
            rgb, kind, n_frames = video_frame(report / t["video"], to["call"])
            save_png(out / "img_alt" / f"{tid}.png", rgb)
            mad = forms_mad(dst, rgb)
            row.update(alt_file=f"img_alt/{tid}.png", alt_image_source=TRAJ_ALT_IMAGE.format(frame=to["call"], kind=kind),
                       alt_image_note=TRAJ_ALT_NOTE, alt_frame=to["call"], alt_frame_type=kind,
                       alt_label_box_xyxy=LABEL_BOX, alt_vs_file_mad=round(mad, 2))
            # frame 0 is the first observation and every step adds one: a video that started earlier would be longer
            if n_frames != t["steps"] + 1:
                problems.append(f"{tid}: video has {n_frames} frames, {t['steps']} steps")
            # a JPEG and a video of different trials would not show the same picture (a one-frame slip would still
            # pass: neighbouring frames differ by less than 1)
            if mad > FORMS_MAD_MAX:
                problems.append(f"{tid}: its JPEG and frame {to['call']} of its video differ by {mad:.1f}/255")
        row.update(state=to["state"], state_source="recorded", instruction=str(t["instruction"]).lower(),
                   success=bool(t["success"]), outcome=t["outcome"], standing_start=t["kind"] == "start",
                   condition=t["group"], scene=f"train_ep{t['episode']:03d}", left_out=t["episode"], pose=pose,
                   source_dir=str(report))
        rows.append(row)
    # the report's own numbers
    if not args.limit:
        if sum(r["success"] for r in rows) != res["successes"]:
            problems.append(f"traj: {sum(r['success'] for r in rows)} successes, the report says {res['successes']}")
        for g in res["groups"]:
            mine = [r for r in rows if r["condition"] == g["group"]]
            if (len(mine), sum(r["success"] for r in mine)) != (g["n"], g["k"]):
                problems.append(f"traj {g['group']}: {sum(r['success'] for r in mine)}/{len(mine)}, the report says "
                                f"{g['k']}/{g['n']}")
    info = dict(dir=str(report), task=res["task"], checkpoint=res["checkpoint"], ckpt_step=res["ckpt_step"],
                instruction=res["instruction"], trials=len(rows), successes=sum(r["success"] for r in rows),
                episodes=sorted({r["left_out"] for r in rows}))
    return rows, info, problems


def target_group(t: dict, box: dict) -> str:
    """The target sweep's own split (plot_target_pose.py): by the grid point against the training demos' box."""
    if t["inside"]:
        return TARGET_GROUPS[0]
    dx = t["plan"]["dx"]
    return TARGET_GROUPS[1] if dx < box["dx"][0] else TARGET_GROUPS[2] if dx > box["dx"][1] else TARGET_GROUPS[3]


def index_target(report: Path, out: Path, index: dict, standup: dict, train: dict,
                 args) -> tuple[list[dict], dict, list[str]]:
    res = json.loads((report / "results.json").read_text())
    trials = sorted(res["trials_detail"], key=lambda t: t["trial"])
    box = res["training_range"]
    rows, problems = [], same_checkpoint("target", res, index)
    if args.target_state == "standup":
        state = standup["state"]
        source = (f"proxy: the eval's stand-up state as the trajectory sweep recorded it (identical in "
                  f"{standup['n_identical']} of its {standup['n_standing']} standing starts); this sweep saved no state")
    else:
        state = train["median"]
        source = (f"proxy: median of the {train['n']} training episodes' first-frame states; this sweep saved no state")
    for t in trials[:args.limit or None]:
        tid = f"target_{t['trial']:04d}"
        pl = t["placed"]
        rgb, kind, n_frames = video_frame(report / t["video"], 0)
        if rgb.shape != (CAM_SIZE[1], CAM_SIZE[0], 3):
            problems.append(f"{tid}: frame 0 is {rgb.shape}")
        if n_frames != t["steps"] + 1:
            problems.append(f"{tid}: video has {n_frames} frames, {t['steps']} steps")
        save_png(out / "img" / f"{tid}.png", rgb)
        # where the pelvis stood up, against the spawn: the usual stand-up or one that went differently
        off = [*to_frame(pl["robot"], pl["pelvis_at_start"][:2]), wrap(pl["pelvis_at_start"][2] - pl["robot"][2])]
        nominal = is_nominal(off, standup["pelvis_offset"])
        dev_mm = 1000 * math.hypot(off[0] - standup["pelvis_offset"][0], off[1] - standup["pelvis_offset"][1])
        dev_deg = math.degrees(wrap(off[2] - standup["pelvis_offset"][2]))
        if nominal:
            note = ("the pelvis stood up where the trajectory sweep's did (within 0.2 mm, 0.02 deg): the true state is "
                    "very likely the eval's stand-up state")
        else:
            note = (f"the stand-up went differently (pelvis {dev_mm:.1f} mm, {dev_deg:+.2f} deg off the usual one): "
                    f"the true state may be off the eval's stand-up state by up to "
                    f"{standup['max_dev_other_standup']:.2f} rad in a joint (median "
                    f"{standup['median_dev_other_standup']:.3f}), as in the trajectory sweep's such starts")
        faucet = [*pl["world"][:2], pl["robot"][2] + math.radians(pl["dyaw"])]
        # the prompt is the scene's language, which the eval sends unless the runner was given INSTRUCTION (its
        # default is empty; the report does not keep the run's settings, so an override would not show)
        pose = dict(faucet_spawn=[t["dx"], t["dy"], t["dyaw"]], plan=[t["plan"]["dx"], t["plan"]["dy"], t["plan"]["dyaw"]],
                    grid=t.get("grid"), point=t.get("point"), rep=t.get("rep"), seed=t.get("seed"),
                    inside_training_range=bool(t["inside"]), out_by=round(t["out_by"], 4), to_train=round(t["to_train"], 4),
                    robot_spawn=pl["robot"], pelvis=pl["pelvis_at_start"],
                    standup_offset=[round(off[0], 4), round(off[1], 4), round(off[2], 4)],
                    standup_dev=[round(dev_mm, 2), round(dev_deg, 3)], standup_nominal=nominal,
                    **faucet_truth(pl["pelvis_at_start"], faucet, index["train_range"]))
        rows.append(dict(
            id=tid, set="target", source_label=t["label"], file=f"img/{tid}.png",
            image_source=TARGET_IMAGE.format(decoder=decoder_name()), first_obs_exact=False, image_note=TARGET_NOTE,
            image_size=list(CAM_SIZE), frame_type=kind, state=state, state_source=source, state_note=note,
            state_standup_nominal=nominal, instruction=str(pl["instruction"]).lower(), success=bool(t["success"]),
            outcome=t["outcome"], standing_start=True, condition=target_group(t, box),
            scene=f"dr2_ep{t['base_episode']}", left_out=None, pose=pose, source_dir=str(report)))
    if not args.limit:
        ins = [r for r in rows if r["pose"]["inside_training_range"]]
        outs = [r for r in rows if not r["pose"]["inside_training_range"]]
        mine = dict(all=(len(rows), sum(r["success"] for r in rows)), inside=(len(ins), sum(r["success"] for r in ins)),
                    outside=(len(outs), sum(r["success"] for r in outs)))
        theirs = dict(all=(res["trials"], res["successes"]),
                      inside=(res["inside_training_range"]["n"], res["inside_training_range"]["successes"]),
                      outside=(res["outside_training_range"]["n"], res["outside_training_range"]["successes"]))
        for k, v in res["by_direction"].items():
            g = [r for r in rows if r["condition"] == k]
            mine[k], theirs[k] = (len(g), sum(r["success"] for r in g)), (v["n"], v["successes"])
        problems += [f"target {k}: {mine[k][1]}/{mine[k][0]}, the report says {theirs[k][1]}/{theirs[k][0]}"
                     for k in mine if mine[k] != theirs[k]]
    scenes = Path(res["settings"]["base_scenes"])
    if not scenes.is_dir():  # the report keeps the path of the machine it ran on
        scenes = Path(args.simple_dir) / "data" / "simple" / res["task"] / f"dr-level-{res['settings']['dr_level']}"
    info = dict(dir=str(report), task=res["task"], checkpoint=res["checkpoint"], ckpt_step=res["ckpt_step"],
                instruction=res["instruction"], trials=len(rows), successes=sum(r["success"] for r in rows),
                training_box_spawn_frame=box, base_scenes=str(scenes),
                base_scene_states_all_zero=base_scene_states_zero(scenes),
                state_proxy=args.target_state, n_standup_nominal=sum(r["state_standup_nominal"] for r in rows),
                n_standup_other=sum(not r["state_standup_nominal"] for r in rows))
    return rows, info, problems


# ---------------------------------------------------------------------------------------------- checks, summary
def check_rows(rows: list[dict], out: Path) -> list[str]:
    """Every image opens and has its size, every state is 32 finite numbers, every id is used once."""
    problems = []
    if len({r["id"] for r in rows}) != len(rows):
        problems.append("ids are not unique")
    for r in rows:
        for key, size in (("file", tuple(r["image_size"])), ("alt_file", CAM_SIZE)):
            if key not in r:
                continue
            try:
                with Image.open(out / r[key]) as im:
                    im.load()
                    if im.size != size or im.mode != "RGB":
                        problems.append(f"{r['id']}: {r[key]} is {im.size} {im.mode}, expected {size} RGB")
            except Exception as e:  # a truncated or missing file
                problems.append(f"{r['id']}: {r[key]} does not open ({e})")
        s = np.asarray(r["state"], dtype=float)
        if s.shape != (STATE_DIM,) or not np.isfinite(s).all():
            problems.append(f"{r['id']}: state is not {STATE_DIM} finite numbers")
        if r["instruction"] != r["instruction"].lower():
            problems.append(f"{r['id']}: the instruction is not lower-case")
    return problems


def counts(rows: list[dict]) -> dict:
    table: dict = {}
    for name, groups in (("traj", TRAJ_GROUPS), ("target", TARGET_GROUPS)):
        mine = [r for r in rows if r["set"] == name]
        table[name] = {g: dict(n=sum(r["condition"] == g for r in mine),
                               successes=sum(r["success"] for r in mine if r["condition"] == g),
                               standing_start=sum(r["standing_start"] for r in mine if r["condition"] == g),
                               in_train_start_range=sum(r["pose"]["in_train_start_range"] for r in mine
                                                        if r["condition"] == g))
                       for g in groups if any(r["condition"] == g for r in mine)}
        other = sorted({r["condition"] for r in mine} - set(groups))
        if other:
            raise SystemExit(f"{name}: conditions the sweep's report does not name: {other}")
    return table


def print_summary(rows: list[dict], table: dict, standup: dict, train: dict) -> None:
    print(f"{'set':<7} {'condition':<32} {'n':>4} {'success':>8} {'rate':>6} {'standing':>9} {'in start range':>15}")
    for name, groups in table.items():
        for g, c in groups.items():
            print(f"{name:<7} {g:<32} {c['n']:>4} {c['successes']:>8} {c['successes'] / c['n']:>6.0%} "
                  f"{c['standing_start']:>9} {c['in_train_start_range']:>15}")
        mine = [r for r in rows if r["set"] == name]
        k, st = sum(r["success"] for r in mine), [r for r in mine if r["standing_start"]]
        print(f"{name:<7} {'all':<32} {len(mine):>4} {k:>8} {k / max(len(mine), 1):>6.0%} {len(st):>9} "
              f"{sum(r['pose']['in_train_start_range'] for r in mine):>15}")
        if name == "traj" and mine:
            mid = [r for r in mine if not r["standing_start"]]
            print(f"        standing starts {sum(r['success'] for r in st)}/{len(st)}, mid-demo takeovers "
                  f"{sum(r['success'] for r in mid)}/{len(mid)}")
            mads = [r["alt_vs_file_mad"] for r in mine if "alt_vs_file_mad" in r]
            if mads:
                print(f"        JPEG vs video frame of the same instant, at the model's input size: mean "
                      f"{np.mean(mads):.2f} / 255 ({min(mads):.2f} to {max(mads):.2f})")
        if name == "target" and mine:
            n = sum(r["state_standup_nominal"] for r in mine)
            print(f"        state proxy: {n} trials stood up as usual, {len(mine) - n} differently")
    print(f"prompts: " + ", ".join(f"{name}: {sorted({r['instruction'] for r in rows if r['set'] == name})}"
                                    for name in table))
    d = np.array(standup["state"]) - np.array(train["median"])
    worst = np.argsort(-np.abs(d))[:4]
    print(f"stand-up state (identical in {standup['n_identical']}/{standup['n_standing']} trajectory-sweep standing "
          f"starts) minus the training starts' median: " + ", ".join(f"{STATE_NAMES[i]} {d[i]:+.4f}" for i in worst)
          + " rad")
    widest = STATE_NAMES[int(np.argmax(train["range"]))]
    print(f"training start states vary by up to {max(train['range']):.3f} rad per dim ({widest}); "
          f"the standing starts' by up to {max(standup['range_all_standing']):.3f}")


def main() -> int:
    args = parse_args()
    psi0, out = Path(args.psi0_dir), Path(args.out)
    traj_report = Path(args.traj_run or psi0 / TRAJ_RUN) / "report"
    target_report = Path(args.target_run or psi0 / TARGET_RUN) / "report"
    index = json.loads((Path(args.exp3) / "prep" / "train_index.json").read_text())
    for d in ("img", "img_alt") if args.traj_video_frames else ("img",):
        (out / d).mkdir(parents=True, exist_ok=True)

    traj_all = json.loads((traj_report / "results.json").read_text())["trials_detail"]
    standup, train = standup_state(traj_all), train_start_states(index)
    print(f"[trials] decoding with {decoder_name()}; trajectory sweep {traj_report.parent.name}, "
          f"target sweep {target_report.parent.name}", flush=True)
    traj, traj_info, problems = index_traj(traj_report, out, index, args)
    print(f"[trials] traj: {len(traj)} trials", flush=True)
    target, target_info, p2 = index_target(target_report, out, index, standup, train, args)
    print(f"[trials] target: {len(target)} trials", flush=True)
    rows = traj + target
    problems += p2 + check_rows(rows, out)
    table = counts(rows)  # before anything is written: it stops the run on a condition the reports do not name

    (out / "meta.json").write_text(json.dumps(dict(
        args=vars(args), decoder=decoder_name(), sources=dict(traj=traj_info, target=target_info), counts=table,
        train_range=index["train_range"],
        state=dict(names=STATE_NAMES, standup=standup, train_starts=train,
                   standup_minus_train_median=(np.array(standup["state"]) - np.array(train["median"])).round(6).tolist()),
        problems=problems), indent=1))
    # last, and in one step: the runner skips this stage once trials.jsonl exists, so it must never be half-written
    # or stand next to images and a meta.json of an unfinished run
    tmp = out / "trials.jsonl.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out / "trials.jsonl")
    print_summary(rows, table, standup, train)
    for p in problems:
        print(f"[trials] CHECK FAILED: {p}")
    print(f"[trials] {len(rows)} trials -> {out / 'trials.jsonl'}" + ("" if problems else "; all checks passed"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
