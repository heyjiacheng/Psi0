"""Render the head camera for every item of the LLMD render plan (stage 1 of experiment 4).

Runs in SIMPLE's venv (Isaac Sim + MuJoCo). No controller and no policy server: for each scene of
prepare_llmd.py's render_plan.json the environment is reset exactly as recorded (room, table material, lighting,
clutter, faucet; like simple.cli.render_decoupled_wbc, which rendered the training images), then the robot is put at
each item's pelvis pose + 43 joints and the free objects where they were recorded at that frame, and the head camera
is rendered. Only the renderer runs; physics is never stepped.

Written to --out (native 360x640 ``head_stereo_left``, lossless PNG):
  img/<id>.png   one render per plan item (start_000, unused_000, extra_00, ...)
  index.jsonl    one line per image, appended as soon as the image is on disk: id, set, kind, scene, file, the
                 camera's world pose (cam_pos, cam_quat wxyz), subframes (rendered at that pose before it was read),
                 warmup, settle, and for "start" items recorded_gap: the mean abs difference (0-255) to that
                 episode's training frame 0, decoded from the train pack's video as the training dataloader does
  meta.json      written at the end of every process: the arguments, the camera, the image shape, how many items of
                 each set are done, min / median / max recorded_gap so far, seconds per scene of this process

Resumable: items that are in index.jsonl and whose PNG exists are skipped (a line without its image is dropped
from the file; lines are in the order rendered, so look items up by id). One process resets at most --max-scenes
scenes (Isaac's memory grows with every reset) and then stops; call it in a loop until its last line says
"wrote all". The last line of the output of a process that got through is "done: <k>/<n> items" (+ " - wrote all"
when nothing is left), exit code 0. When nothing is left Isaac is not started at all. With --only, <n> and meta.json
count the selected scenes only. A process that fails prints "Error: ..." or the traceback and exits 1: with Isaac
running Python's own exit code is lost (a sys.exit("message") came out as exit code 0), so the script leaves with
os._exit itself; what was rendered before the failure stays valid and is skipped by the next call.

meta.json also carries "camera_in_pelvis": where the head camera sits in the frame of the plan's pelvis pose (x
ahead, y left, z up, heading), the median over the images and the largest deviation from it. The camera is fixed
to the torso and the start postures are alike, so every image should give the same numbers (0.2 mm and 0.1 deg
over 17 scenes when measured); a robot that was not moved to its plan pose (it would stand at the scene's spawn
pose, 5 to 6 cm ahead of the recorded start) shows up there, and a "WARNING:" line is printed above 3 cm or 3 deg.
"warmup_settle" lists the distinct (--warmup, --settle) the images were rendered with; more than one (a resume
with other arguments) also gives a "WARNING:" line.

Renderer notes (measured on this task, RTX 5080):
  * A reset builds a new MuJoCo model and data: a pose written through handles kept from before a reset does not
    move the robot (0.000 m against 0.500 m), so sim.mjModel / sim.mjData are read again at every use. The GI
    setting and the switched-off right camera survived a reset when checked; both are set again anyway.
  * The free objects go where the item's frame recorded them, as the training renderer did: in 19 of the 155
    scenes one had moved more than 5 mm or 3 deg from where the reset puts it by the time the recording started,
    in 7 of them the knife by 3 cm and 17 deg (--objects reset leaves them there; the training frame shows the
    recorded pose).
  * RTX indirect diffuse accumulates over frames and is carried over a reset (the scene before leaves a colour
    cast), so the first pose of a scene is rendered --warmup subframes before it is read (25 ms each). Gap to the
    training frame, /255, by subframes (--probe-warmup, 5 scenes): 200: 4.4-7.4, 300: 4.0-6.1, 400: 3.7-5.2,
    600: 3.5-5.4. One scene is still falling at 400 (3.7 at 1600); another, rendered first in a process, is
    closest at 200 (4.6) and drifts away after (5.8 at 1600): the training frames are not all settled renders, so
    no warm-up matches every scene. 400 is what experiment 3 used; on 6 scenes in a row 400 gave 4.3-5.7
    (median 4.9) and 800 gave 3.7-5.3 (median 4.6) at +10 s per scene.
  * Further poses of the same scene get --settle subframes, more after a long move or turn (the rule of
    render_base_grid.py). What was rendered before still shows. The same pose rendered again in another process
    differs by 0.1-0.2/255 after the same renders in the same order, by 0.7-0.9/255 from experiment 3's renders
    of the same start, by 1.9-2.4/255 for an extra pose reached from another pose of its scene, and by
    2.2-3.6/255 for a start whose process rendered a different scene before it (or none): mostly a colour cast,
    up to 6 levels in one channel, and as much as half of what separates a re-render from its training frame.
    The plan's scene order and --max-scenes therefore belong to the result; keep both when rendering again.
  * One scene takes 13 s (reset 3, warm-up 10), the first of a process 20 s (+9 s to start Isaac); the process
    grows by 0.11 GB of host memory and 0.07 GB of GPU memory per reset (9 GB and 5.7 GB after the first).
  * --probe-warmup 1,2,4,... renders nothing to img/: for each selected scene it resets, and reads the first item
    after each of those cumulative subframe counts, writing the gap to the training frame to warmup_probe.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

CAM = "head_stereo_left"
GI_SETTING = "/rtx/indirectDiffuse/enabled"
PLACED_M, PLACED_DEG = 0.03, 3.0  # the camera further than this from where the other images have it: warn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, help="render_plan.json of prepare_llmd.py")
    p.add_argument("--out", required=True, help="the renders directory")
    p.add_argument("--env-id", default="", help="default: simple/<the plan's task>")
    p.add_argument("--max-scenes", type=int, default=40, help="scenes this process resets before it stops")
    p.add_argument("--settle", type=int, default=12, help="RTX subframes rendered per pose before reading it")
    p.add_argument("--warmup", type=int, default=400, help="subframes after each reset (materials load, GI settles)")
    p.add_argument("--gi", choices=("on", "off"), default="on", help="RTX indirect diffuse GI (on: as trained on)")
    p.add_argument("--only", default="", help="comma list of scene names: render only these")
    p.add_argument("--objects", choices=("recorded", "reset"), default="recorded",
                   help="the free objects on the table: where the item's frame recorded them (as the training "
                        "renderer put them), or left where the scene reset drops them")
    p.add_argument("--probe-warmup", default="", help="comma list of cumulative subframe counts: measure the gap to "
                                                      "the training frame after each (writes warmup_probe.jsonl only)")
    p.add_argument("--headless", type=int, default=int(os.environ.get("HEADLESS", "1")))
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- files
def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def read_index(out: Path) -> dict[str, dict]:
    """The lines of index.jsonl whose PNG exists, by item id. A line without its image, or one cut off by a crash,
    is dropped from the file, so the file always has exactly one line per image."""
    path = out / "index.jsonl"
    if not path.is_file():
        return {}
    rows, stale = {}, 0
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            known = (out / row["file"]).is_file() and row["id"] not in rows
        except (json.JSONDecodeError, KeyError, TypeError):  # cut off, or not a row of this script
            stale += 1
            continue
        if known:
            rows[row["id"]] = row
        else:
            stale += 1
    if stale:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
        os.replace(tmp, path)
        print(f"index.jsonl: dropped {stale} line(s) without an image", flush=True)
    return rows


def scene_conf(scene: dict) -> dict:
    """The scene as recorded: the environment_config of its line of meta/episodes.jsonl."""
    line = Path(scene["episodes_jsonl"]).read_text().splitlines()[scene["line"]]
    return json.loads(json.loads(line)["environment_config"])


def recorded_frame(scene: dict) -> np.ndarray:
    """Training frame 0 of a training scene's episode as the train pack stores it (decoded from its video)."""
    train_dir = Path(scene["episodes_jsonl"]).parent.parent
    info = json.loads((train_dir / "meta" / "info.json").read_text())
    key = next(k for k, v in info["features"].items() if v.get("dtype") == "video")
    path = train_dir / info["video_path"].format(episode_chunk=scene["episode"] // info.get("chunks_size", 1000),
                                                 video_key=key.split(".")[-1], episode_index=scene["episode"])
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", "select=eq(n\\,0)", "-vsync", "0",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    h, w = info["features"][key]["shape"][:2]
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def camera_in_pelvis(items: list[dict], rows: dict) -> dict | None:
    """Where the rendered camera is in the frame of each item's plan pelvis pose. It is the same for every image
    when the robot stood where the plan put it, whatever the scene: the check that no pose was lost on the way
    from the plan through MuJoCo to Isaac (after a reset, say)."""
    ids, local = [], []
    for it in items:
        if it["id"] in rows:
            base, row = it["base_pose"], rows[it["id"]]
            yaw = yaw_of(base[3:7])
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy, dz = (row["cam_pos"][k] - base[k] for k in range(3))
            turn = (yaw_of(row["cam_quat"]) - yaw + math.pi) % (2 * math.pi) - math.pi
            ids.append(it["id"])
            local.append([c * dx + s * dy, -s * dx + c * dy, dz, math.degrees(turn)])
    if not ids:
        return None
    local = np.array(local)
    median = np.median(local, axis=0)
    off_m, off_deg = np.linalg.norm(local[:, :3] - median[:3], axis=1), np.abs(local[:, 3] - median[3])
    return dict(n=len(ids), median_xyz_m=median[:3].round(4).tolist(), median_heading_deg=round(float(median[3]), 3),
                max_dev_m=round(float(off_m.max()), 4), max_dev_m_id=ids[int(off_m.argmax())],
                max_dev_deg=round(float(off_deg.max()), 3), max_dev_deg_id=ids[int(off_deg.argmax())],
                beyond_limits=[i for i, m, a in zip(ids, off_m, off_deg) if m > PLACED_M or a > PLACED_DEG])


def write_meta(out: Path, args, plan: dict, scenes: list[dict], rows: dict, process: dict) -> dict:
    items = [it for s in scenes for it in s["items"]]
    gaps = [r["recorded_gap"] for r in rows.values() if r.get("recorded_gap") is not None]
    shape = None
    if rows:
        from PIL import Image

        with Image.open(out / next(iter(rows.values()))["file"]) as im:
            shape = [im.height, im.width, len(im.getbands())]
    meta = dict(
        plan=str(Path(args.plan).resolve()), task=plan["task"], args=vars(args), cam=CAM, image_shape=shape,
        items=dict(done=sum(it["id"] in rows for it in items), total=len(items)),
        counts={k: dict(done=sum(it["id"] in rows for it in items if it["set"] == k),
                        total=sum(it["set"] == k for it in items)) for k in ("start", "unused", "extra")},
        complete=all(it["id"] in rows for it in items),
        recorded_gap=dict(n=len(gaps), min=min(gaps), median=float(np.median(gaps)), max=max(gaps)) if gaps else None,
        # the distinct [warm-up, settle] the images were rendered with: more than one after a resume with other
        # arguments, and then the images are not comparable (the warm-up moves an image by several /255)
        warmup_settle=sorted({(r.get("warmup"), r.get("settle")) for r in rows.values()}, key=str),
        camera_in_pelvis=camera_in_pelvis(items, rows), this_process=process)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------------------------------------- sim
def make_env(env_id: str, headless: bool):
    import gymnasium as gym
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    import simple.envs  # noqa: F401  (registers the env ids)

    sonic_config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[]).load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    env = gym.make(env_id, sim_mode="mujoco_isaac", render_hz=50, physics_dt=float(sonic_config["SIMULATE_DT"]),
                   headless=headless, webrtc=False, max_episode_steps=30000, sonic_config=sonic_config)
    return env, env.unwrapped


def reset_scene(env, sim, scene: dict, gi: bool):
    """Reset to the scene as recorded; returns the head camera. Everything a reset undoes is done again here."""
    import carb

    conf = scene_conf(scene)
    conf["uid"] = sim.task.uid  # same fix-up as the eval and experiment 3
    env.reset(options={"state_dict": conf, "task_id": scene["scene"], "dr_level": None})
    carb.settings.get_settings().set(GI_SETTING, gi)
    try:  # the policy only sees the left image; not rendering the right one saves time
        sim.isaac.cameras[CAM.replace("left", "right")]._render_product.hydra_texture.set_updates_enabled(False)
    except (AttributeError, KeyError):
        pass
    return sim.isaac.cameras[CAM]


def place(sim, item: dict, objects: bool) -> None:
    """The robot at the item's pelvis pose and joints, the free objects where that frame recorded them."""
    import mujoco

    m, d = sim.mjModel, sim.mjData  # read here, never kept: a reset replaces both
    d.qpos[:7] = np.asarray(item["base_pose"], dtype=np.float64)
    for name, value in zip(sim.task.robot.joint_names, item["joints"]):
        sim.mujoco.joints[name].qpos = value
    for label, pose in (item.get("object_poses") or {}).items() if objects else ():
        joint = m.body_jntadr[sim.mujoco.mj_objects[label].id]
        if joint < 0 or m.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"object {label!r} has no free joint to put it at its recorded pose")
        adr = int(m.jnt_qposadr[joint])
        d.qpos[adr:adr + 7] = np.asarray(pose, dtype=np.float64)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    sim.isaac.step(sim.mujoco)  # push MuJoCo -> Isaac


def grab(cam, subframes: int) -> np.ndarray:
    import omni.replicator.core as rep

    rep.orchestrator.step(rt_subframes=subframes, pause_timeline=False)
    return np.asarray(cam.get_rgba()[..., :3], dtype=np.uint8)


def settle_factor(a: dict, b: dict) -> int:
    """A long move or turn shows the renderer a new view: let its accumulated lighting settle longer
    (render_base_grid.py's rule)."""
    move = math.hypot(a["base_pose"][0] - b["base_pose"][0], a["base_pose"][1] - b["base_pose"][1])
    turn = abs((yaw_of(a["base_pose"][3:7]) - yaw_of(b["base_pose"][3:7]) + math.pi) % (2 * math.pi) - math.pi)
    return 4 if move > 0.45 or turn > 0.6 else 2 if move > 0.12 or turn > 0.2 else 1


def close_quietly(env) -> None:
    """Close Isaac with its shutdown messages thrown away: closing ends the process, and the line printed before
    must stay the last one of the output."""
    sys.stdout.flush()
    sys.stderr.flush()
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        env.close()
    except BaseException:  # noqa: BLE001  everything is written and reported: a bad shutdown is not a failed run
        os._exit(0)


def fail(err: BaseException) -> None:
    """Leave with a failure code. With Isaac running, the process ends with exit code 0 whatever Python raised
    (measured: the joint-order sys.exit("...") came out as 0), and the runner would take a crashed pass for one
    that got through and start Isaac again. The message goes straight to the output's file descriptor, so it is
    there before the process is gone."""
    if isinstance(err, SystemExit) and err.code in (None, 0):
        code, text = 0, ""
    elif isinstance(err, SystemExit):
        code, text = (err.code, f"exit {err.code}") if isinstance(err.code, int) else (1, str(err.code))
    else:
        code = 130 if isinstance(err, KeyboardInterrupt) else 1
        text = "".join(traceback.format_exception(type(err), err, err.__traceback__)).rstrip()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    try:
        if code:
            os.write(1, (text if text.startswith("Traceback") else f"Error: {text}").encode() + b"\n")
    finally:
        os._exit(code)


def memory() -> dict:
    """This process's memory, host (GB) and GPU (GB, None if nvidia-smi cannot say): it grows with every reset,
    which is what --max-scenes is for."""
    with open("/proc/self/statm") as f:
        rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    gpu = None
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout
        gpu = next((round(float(mb) / 1024, 2) for pid, mb in (line.split(",") for line in apps.splitlines() if "," in line)
                    if int(pid) == os.getpid()), None)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return dict(rss_gb=round(rss, 2), gpu_gb=gpu)


# ---------------------------------------------------------------------------------------------- the two jobs
def render_scene(env, sim, scene: dict, done: dict, args, out: Path, index) -> dict:
    """Reset one scene and render its items that are not done yet; returns its timing."""
    from PIL import Image

    t0 = time.perf_counter()
    cam = reset_scene(env, sim, scene, args.gi == "on")
    t_reset = time.perf_counter() - t0
    items = scene["items"]
    rec = recorded_frame(scene) if any(it["set"] == "start" for it in items) else None
    place(sim, items[0], args.objects == "recorded")  # the warm-up is always at the scene's first pose
    grab(cam, args.warmup)
    t_warm = time.perf_counter() - t0 - t_reset
    last, warm, n_new = items[0], args.warmup, 0
    for item in items:
        if item["id"] in done:
            continue
        place(sim, item, args.objects == "recorded")
        subframes = args.settle * settle_factor(item, last)
        img = grab(cam, subframes)
        if img.shape != (360, 640, 3):
            raise RuntimeError(f"{item['id']}: the camera gave {img.shape}, the model was trained on (360, 640, 3)")
        pos, quat = cam.get_world_pose(camera_axes="world")
        row = dict(id=item["id"], set=item["set"], kind=item.get("kind"), scene=scene["scene"],
                   file=f"img/{item['id']}.png", cam_pos=np.asarray(pos, dtype=float).round(4).tolist(),
                   cam_quat=np.asarray(quat, dtype=float).round(5).tolist(),
                   # the pose the warm-up was done at has seen those subframes too
                   subframes=subframes + (warm if item is last else 0), warmup=args.warmup, settle=args.settle)
        if item["set"] == "start":
            row["recorded_gap"] = gap(img, rec)
        tmp = out / "img" / f".{item['id']}.png.tmp"
        Image.fromarray(img).save(tmp, format="PNG")
        os.replace(tmp, out / row["file"])  # the image is complete before its line exists
        index.write(json.dumps(row) + "\n")
        index.flush()
        os.fsync(index.fileno())
        done[item["id"]] = row
        last, warm, n_new = item, 0, n_new + 1
    table = sim.task.layout.actors.get("table")
    return dict(scene=scene["scene"], items=n_new, seconds=round(time.perf_counter() - t0, 2),
                reset_s=round(t_reset, 2), warmup_s=round(t_warm, 2), **memory(),
                table_material=(getattr(table, "material", None) or {}).get("name"),
                recorded_gap=next((done[it["id"]].get("recorded_gap") for it in items if it["set"] == "start"), None))


def probe_scene(env, sim, scene: dict, counts: list[int], args, log) -> None:
    """After a reset: the gap between the first item's render and its training frame, as the subframes add up."""
    t0 = time.perf_counter()
    cam = reset_scene(env, sim, scene, args.gi == "on")
    t_reset = time.perf_counter() - t0
    rec = recorded_frame(scene) if scene["source"] == "train" else None
    place(sim, scene["items"][0], args.objects == "recorded")
    total, prev, t1 = 0, None, time.perf_counter()
    for c in counts:
        img = grab(cam, c - total)
        total = c
        row = dict(scene=scene["scene"], subframes=c, seconds=round(time.perf_counter() - t1, 2), reset_s=round(t_reset, 2),
                   objects=args.objects, recorded_gap=None if rec is None else round(gap(img, rec), 3),
                   step_gap=None if prev is None else round(gap(img, prev), 3), mean=round(float(img.mean()), 2))
        prev = img
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"probe {scene['scene']}: {c:4d} subframes ({row['seconds']:5.1f} s)  gap to the training frame "
              f"{row['recorded_gap']}  to the read before {row['step_gap']}  mean {row['mean']}", flush=True)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan = json.loads(Path(args.plan).read_text())
    only = {s for s in args.only.split(",") if s}
    unknown = only - {s["scene"] for s in plan["scenes"]}
    if unknown:
        raise SystemExit(f"--only: not in the plan: {sorted(unknown)}")
    scenes = [s for s in plan["scenes"] if not only or s["scene"] in only]
    items = [it for s in scenes for it in s["items"]]
    probe = [int(v) for v in args.probe_warmup.split(",") if v.strip()]
    done = read_index(out)
    todo = scenes if probe else [s for s in scenes if any(it["id"] not in done for it in s["items"])]
    batch, timing = todo[:max(args.max_scenes, 0)], []
    process = dict(scenes=timing, startup_s=None, seconds_per_scene=None, total_s=None)

    if batch:  # Isaac is only started when there is something to render
        env, sim = make_env(args.env_id or f"simple/{plan['task']}", bool(args.headless))
        names = plan.get("joint_names")
        if names and list(sim.task.robot.joint_names) != names:
            raise SystemExit("the plan's joints are not in the order of the robot's joint_names")
        process["startup_s"] = round(time.perf_counter() - started, 1)
        if probe:
            with open(out / "warmup_probe.jsonl", "a") as log:
                for scene in batch:
                    probe_scene(env, sim, scene, probe, args, log)
        else:
            with open(out / "index.jsonl", "a") as index:
                for n, scene in enumerate(batch):
                    t = render_scene(env, sim, scene, done, args, out, index)
                    timing.append(t)
                    print(f"scene {n + 1}/{len(batch)} {t['scene']}: {t['items']} image(s) in {t['seconds']:.1f} s "
                          f"(reset {t['reset_s']:.1f} s, warm-up {t['warmup_s']:.1f} s), "
                          + (f"{t['recorded_gap']:.2f}/255 from the training frame, " if t["recorded_gap"] is not None else "")
                          + f"process memory {t['rss_gb']:.1f} GB"
                          + (f", GPU {t['gpu_gb']:.1f} GB" if t["gpu_gb"] is not None else ""), flush=True)
    if timing:
        process["seconds_per_scene"] = round(float(np.median([t["seconds"] for t in timing])), 2)
    process["total_s"] = round(time.perf_counter() - started, 1)
    if probe:  # a measurement, not a render run: no meta.json, no "done" line
        print(f"probe: {len(batch)} scene(s) x {len(probe)} reads -> {out / 'warmup_probe.jsonl'}", flush=True)
    else:
        rows = {it["id"]: done[it["id"]] for it in items if it["id"] in done}
        meta = write_meta(out, args, plan, scenes, rows, process)
        placed = meta["camera_in_pelvis"]
        if len(meta["warmup_settle"]) > 1:
            print(f"WARNING: the images were rendered with different (warm-up, settle) subframes {meta['warmup_settle']}: "
                  "an earlier call had other arguments; they are not comparable (start again in an empty --out)", flush=True)
        if placed:
            print(f"camera in the frame of the plan's pelvis pose, {placed['n']} images: at most "
                  f"{placed['max_dev_m'] * 1000:.1f} mm and {placed['max_dev_deg']:.2f} deg from their median", flush=True)
            if placed["beyond_limits"]:
                print(f"WARNING: {len(placed['beyond_limits'])} image(s) have the camera more than {PLACED_M * 100:.0f} cm "
                      f"or {PLACED_DEG:.0f} deg from where the others have it; the robot was probably not at its plan "
                      f"pose there (delete their PNGs to render them again): {', '.join(placed['beyond_limits'][:20])}",
                      flush=True)
        gaps = [r["recorded_gap"] for r in rows.values() if "recorded_gap" in r]
        if gaps:
            print(f"start renders vs their training frame: {min(gaps):.2f} / {float(np.median(gaps)):.2f} / "
                  f"{max(gaps):.2f} per 255 (min / median / max of {len(gaps)})", flush=True)
        print(f"done: {len(rows)}/{len(items)} items" + (" - wrote all" if len(rows) == len(items) else ""), flush=True)
    if batch:
        close_quietly(env)


if __name__ == "__main__":
    try:
        main()
    except BaseException as err:  # noqa: BLE001  sys.exit and Ctrl-C too: see fail()
        fail(err)
