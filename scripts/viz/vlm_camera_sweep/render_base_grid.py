"""Render the head camera at every base pose of the plan (stage 1 of the in-distribution experiment).

Runs in SIMPLE's venv (Isaac Sim + MuJoCo). No controller and no policy server: the scene of one training episode
is reset exactly as recorded (table material, lighting, clutter, faucet; like simple.cli.render_decoupled_wbc, which
rendered the training images), the robot is put into that episode's first training frame (pelvis pose + 43 joints,
from the raw teleop recording), and for every pose of prep/plan.json the pelvis is moved over the floor and turned
about the vertical, the posture kept. Only the renderer runs; physics is never stepped.

Written to --out (native 360x640 ``head_stereo_left``, lossless PNG):
  img/00000_recorded.png  the first training frame as the train pack stores it (decoded from its video, like the
                          training dataloader does): a training image of the checkpoint
  img/<n>_<kind>.png      one render per plan pose, in plan order (kind: anchor, fine, medium, coarse, yaw, facing)
  index.jsonl             one line per image: the plan pose, the camera's world pose
  meta.json               the anchor (its state, how far its re-render is from the training image), the scene

Renderer notes (measured on this scene): RTX indirect diffuse GI is on, as in the training renders, and accumulates
over frames, so a render depends a little on where the camera was before. The plan walks from pose to pose in short
steps, each pose gets --settle subframes and more after a long move; the anchor is rendered again after every block
of the plan, which measures what is left of that: two renders of the anchor pose differ by 1-4/255 mean abs, against
about 5/255 between a render and the training image (video compression included). meta.json keeps both.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

CAM = "head_stereo_left"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default="simple/G1WholebodyOpenFaucetTeleop-v0")
    p.add_argument("--prep", required=True, help="directory of prepare_base_grid.py (train_index.json, plan.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--settle", type=int, default=12, help="RTX subframes rendered per pose before reading it")
    p.add_argument("--warmup", type=int, default=400, help="subframes after the reset (materials load, GI settles)")
    p.add_argument("--gi", choices=("on", "off"), default="on", help="RTX indirect diffuse GI (on: as trained on)")
    p.add_argument("--limit", type=int, default=0, help="render only the first N plan poses (0 = all), for a quick look")
    p.add_argument("--headless", type=int, default=int(os.environ.get("HEADLESS", "1")))
    return p.parse_args()


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> None:
    args = parse_args()
    prep, out = Path(args.prep), Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    index = json.loads((prep / "train_index.json").read_text())
    plan = json.loads((prep / "plan.json").read_text())
    poses = plan["poses"][:args.limit] if args.limit > 0 else plan["poses"]
    ep = index["episodes"][plan["episode"]]
    train_dir = Path(index["train_dir"])
    conf = json.loads(json.loads((train_dir / "meta" / "episodes.jsonl").read_text().splitlines()[plan["episode"]])
                      ["environment_config"])

    import gymnasium as gym
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    import simple.envs  # noqa: F401  (registers the env ids)

    sonic_config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[]).load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    env = gym.make(args.env_id, sim_mode="mujoco_isaac", render_hz=50, physics_dt=float(sonic_config["SIMULATE_DT"]),
                   headless=bool(args.headless), webrtc=False, max_episode_steps=30000, sonic_config=sonic_config)
    sim = env.unwrapped

    import carb
    import mujoco
    import omni.replicator.core as rep
    import transforms3d as t3d
    from PIL import Image

    conf["uid"] = sim.task.uid
    env.reset(options={"state_dict": conf, "task_id": f"episode_{plan['episode']}", "dr_level": None})
    carb.settings.get_settings().set("/rtx/indirectDiffuse/enabled", args.gi == "on")
    cam = sim.isaac.cameras[CAM]
    try:  # the policy only sees the left image; not rendering the right one saves time
        sim.isaac.cameras[CAM.replace("left", "right")]._render_product.hydra_texture.set_updates_enabled(False)
    except (AttributeError, KeyError):
        pass
    robot, m, d = sim.task.robot, sim.mjModel, sim.mjData
    base0 = np.asarray(ep["base_pose"], dtype=np.float64)
    joints0 = np.asarray(ep["joints"], dtype=np.float64)
    R0, yaw0 = t3d.quaternions.quat2mat(base0[3:7]), yaw_of(base0[3:7])

    def place(x: float, y: float, yaw: float) -> None:
        """The pelvis at (x, y) on the floor, turned to `yaw` about the vertical; height, tilt and posture kept."""
        d.qpos[:3] = (x, y, base0[2])
        d.qpos[3:7] = t3d.quaternions.mat2quat(t3d.axangles.axangle2mat([0, 0, 1], yaw - yaw0) @ R0)
        for name, value in zip(robot.joint_names, joints0):
            sim.mujoco.joints[name].qpos = value
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        sim.isaac.step(sim.mujoco)  # push MuJoCo -> Isaac

    def grab(subframes: int) -> np.ndarray:
        rep.orchestrator.step(rt_subframes=subframes, pause_timeline=False)
        return np.asarray(cam.get_rgba()[..., :3], dtype=np.uint8)

    def recorded_frame() -> np.ndarray:
        path = train_dir / index["video_path"].format(episode_chunk=plan["episode"] // index["chunks_size"],
                                                      video_key=index["video_key"], episode_index=plan["episode"])
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", "select=eq(n\\,0)", "-vsync", "0",
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        h, w = index["image_shape"][:2]
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3)

    rows = []

    def save(img: np.ndarray, kind: str, **meta) -> None:
        name = f"img/{len(rows):05d}_{kind}.png"
        Image.fromarray(img).save(out / name)
        rows.append({"file": name, "kind": kind, **meta})

    x0, y0 = base0[:2]
    place(x0, y0, yaw0)
    grab(args.warmup)
    rec = recorded_frame()
    gap = lambda img: float(np.abs(img.astype(np.int16) - rec.astype(np.int16)).mean())
    save(rec, "recorded", id=-1)
    first = grab(args.settle)
    print(f"anchor: training episode {plan['episode']} frame 0, pelvis ({x0:+.3f}, {y0:+.3f}, "
          f"{math.degrees(yaw0):+.1f} deg); re-render vs the training image {gap(first):.2f}/255 mean abs", flush=True)

    last, gaps, repeats, t_grid = (x0, y0, yaw0), [], [], time.perf_counter()
    for n, q in enumerate(poses):
        move = math.hypot(q["x"] - last[0], q["y"] - last[1])
        turn = abs((q["yaw"] - last[2] + math.pi) % (2 * math.pi) - math.pi)
        place(q["x"], q["y"], q["yaw"])
        # a long move or turn shows the renderer a new view: let its accumulated lighting settle longer
        img = grab(args.settle * (4 if move > 0.45 or turn > 0.6 else 2 if move > 0.12 or turn > 0.2 else 1))
        p, quat = cam.get_world_pose(camera_axes="world")
        meta = {k: q[k] for k in ("id", "fx", "fy", "dyaw", "x", "y", "yaw")}
        if q["kind"] == "anchor":
            gaps.append(gap(img))
            repeats.append(img.astype(np.int16))
            meta["recorded_gap"] = gaps[-1]
        save(img, q["kind"], **meta, cam_pos=np.asarray(p, dtype=float).round(4).tolist(),
             cam_quat=np.asarray(quat, dtype=float).round(5).tolist())
        last = (q["x"], q["y"], q["yaw"])
        if (n + 1) % 50 == 0 or n + 1 == len(poses):
            print(f"grid: {n + 1}/{len(poses)} poses ({(time.perf_counter() - t_grid) / (n + 1):.2f} s each)", flush=True)

    with open(out / "index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lights = sim.task.layout.lights
    table = sim.task.layout.actors.get("table")
    meta = dict(
        anchor=dict(episode=plan["episode"], frame=0, base_pose=base0.tolist(), state=ep["state"],
                    recorded_gap_first=gap(first), recorded_gap_repeats=gaps,
                    # each render of the anchor pose against the first one: what the renderer's history changes
                    repeat_gap_to_first=[float(np.abs(r - repeats[0]).mean()) for r in repeats[1:]]),
        scene=dict(table_material=(getattr(table, "material", None) or {}).get("name"), n_lights=len(lights),
                   light_color_temperature_K=[round(float(lt.light_color_temperature), 1) for lt in lights
                                              if getattr(lt, "light_color_temperature", None) is not None][:1]),
        args=vars(args), cam=CAM, instruction=index["instruction"], task_instruction=sim.task.instruction,
        image_shape=list(first.shape), n_poses=len(poses))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if gaps:
        between = meta["anchor"]["repeat_gap_to_first"]
        print(f"anchor: the {len(gaps)} renders of the anchor pose are {min(gaps):.2f}..{max(gaps):.2f}/255 from the "
              "training image" + (f", {min(between):.2f}..{max(between):.2f}/255 from the first of them" if between else ""),
              flush=True)
    print(f"wrote {len(rows) - 1} renders to {out} in {time.perf_counter() - started:.0f} s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
