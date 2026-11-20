"""Closed-loop success of the served Psi0 checkpoint from training states. Runs in SIMPLE's venv, from the SIMPLE root.

The episode loop is SIMPLE's own Teleop eval, ``simple.cli.eval_decoupled_wbc._run_eval_worker`` with the
``psi0_decoupled_wbc`` agent, unchanged: the same reset, the same stabilization, the same step budget (the task's
max_episode_steps unless --budget-steps) and the same success test (``task.check_success``: the faucet handle turned
past 0.7 rad and held). Every episode is one training episode's own recorded scene (robot spawn, faucet, table,
distractors, materials, lighting; nothing re-drawn). What a trial changes:

  start trials  the robot's spawn is moved (dx ahead, dy to its left) or turned (dyaw) before the reset; it stands up
                there through SIMPLE's stabilization, and the policy runs from the first step, as in any eval
  path trials   the episode's recorded actions for frames 0..t-1 go into the agent's own action queue, so they run
                exactly as a policy chunk would (the agent's joint mapping, the same walking controller). When the
                queue runs empty at frame t the agent asks the policy server (its first request, so the server's RTC
                state is reset, as at any episode start) and the policy takes over. Nothing is set by hand: the robot
                walked there, and its joints and camera are where the replay put them
  --calibrate   the chosen episodes replayed whole, recorded actions only (no policy), to find their path points:
                the reach point is REACH_LEAD frames before the faucet handle starts to turn (TURN_RAD); 1/3 and 2/3
                are where the pelvis has covered that share of its (smoothed) path to the reach point. (The first
                contact is no anchor: an arm brushes the faucet body during the walk.) A replay is kept only if it
                opens the faucet and its pelvis stays within MAX_DRIFT_CM / MAX_DRIFT_DEG of the teleop recording at
                the path points (the walk is physics: a replay drifts a little, and for some episodes enough that the
                hand misses the handle, every time); then the slot moves on to its next backup. One process
                calibrates every slot
  --calib-status  print which slots are calibrated (no simulator) and exit 0 (all), 3 (some still to do) or 4 (a
                slot ran out of episodes)

The prompt is the training one (plan settings, "turn the faucet"), not the scene's.

Output (--out):
  results.jsonl          one line per finished trial, written as it finishes; re-running skips the trials in it
  calib.json             per slot: the replayed episode, its path points and the replay's pelvis path
  calib/                 the replay's head-camera frames (320x180) at the start and the path points, +-1 frame
  traces/<label>.json    10 Hz trace in the recorded spawn's frame: pelvis, hands, faucet travel, walking command
  takeover/<label>.png   what the camera saw when the policy took over (320x180, the policy's input size)
  videos/<label>/        SIMPLE's head-camera videos
  conditions.json        the run's settings and the policy server's /info
The replayed commands are the teleop recording's (simple-teleop pack), from its first frame: the train pack holds the
same commands one frame later (its action k is raw command k+1), so the camera at agent call t is training frame t's
image, rendered after the same commands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

import simple.cli.eval_decoupled_wbc as E
import simple.datasets.lerobot as SL
import simple.dr.manager as M
from simple.baselines.psi0_decoupled_wbc import STATE_SLICES, Psi0DecoupledWbcAgent, from_psi0_upper_joints
from simple.core.action import ActionCmd
from simple.dr.types import Box
from simple.envs.sonic_loco_manip import SonicLocoManipEnv

TRACE_EVERY = 5       # control steps (50 Hz) between trace rows
REACH_DIST = 0.10     # m, a hand this close to a faucet body counts as having reached it
MOVED_RAD = 0.10      # faucet joint travel that counts as having moved it
TURN_RAD = 0.05       # calibration: faucet joint travel that counts as the turn having started
FALL_Z = 0.40         # pelvis height below this = fell
HOLD = 50             # calibration: frames the last recorded command is held after the recording ends
TRIES = 1             # calibration: replays of one episode before its slot moves on to a backup (failures repeat)
MAX_DRIFT_CM = 4.0    # calibration: the replay's pelvis may be this far from the teleop recording at a path point
MAX_DRIFT_DEG = 6.0   # ... and turned this much from it
IMG_W, IMG_H = 320, 180  # the policy's input size
HAND_BODIES = ("hand_index_1_link", "hand_middle_1_link", "hand_thumb_2_link")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the sweep directory, holding plan.json")
    p.add_argument("--calibrate", action="store_true", help="replay the chosen episodes to find their path points")
    p.add_argument("--calib-status", action="store_true", help="report the calibration status and exit (no simulator)")
    p.add_argument("--trials", type=int, default=0, help="run only the first N trials of the plan (0 = all)")
    p.add_argument("--max-new", type=int, default=0,
                   help="stop after this many trials (0 = none); the runner restarts the process for the rest")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--budget-steps", type=int, default=0, help="override the task's step budget (0 = the task's)")
    p.add_argument("--instruction", default="", help="override the plan's prompt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8015)
    return p.parse_args()


def yaw_of(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_z(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def to_frame(origin, xy) -> list[float]:
    """xy in the frame of origin = (x, y, yaw): x ahead, y to the left."""
    c, s = math.cos(origin[2]), math.sin(origin[2])
    dx, dy = xy[0] - origin[0], xy[1] - origin[1]
    return [c * dx + s * dy, -s * dx + c * dy]


def read_episodes(meta: Path) -> list[dict]:
    return [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines() if line.strip()]


def subtree(m, root: int) -> set[int]:
    out, todo = set(), [root]
    while todo:
        b = todo.pop()
        out.add(b)
        todo += [i for i in range(m.nbody) if m.body_parentid[i] == b and i != b]
    return out


def agent_state(agent, obs) -> np.ndarray:
    """The 32-D state the agent sends the server (psi0_decoupled_wbc.get_action)."""
    proprio = np.asarray(obs["joint_qpos"])[None]
    return np.concatenate([proprio[:, s:e] for _, s, e in STATE_SLICES]
                          + [proprio[:, [13, 14, 12]], np.array([[agent._last_base_height_cmd]])], axis=1)[0]


def queue_recorded(agent, a: np.ndarray) -> None:
    """One recorded 36-D action into the agent's queue, exactly as the agent queues a policy action."""
    a = np.asarray(a, dtype=np.float32)
    target_qpos = dict(zip(agent.robot.joint_names[15:], from_psi0_upper_joints(a[:28])))
    waist = {"waist_yaw_joint": a[30], "waist_roll_joint": a[28], "waist_pitch_joint": a[29]}
    agent.queue_action(ActionCmd("vla_cmd", target_upper_body_pose={**target_qpos, **waist},
                                 navigate_cmd=a[32:36], base_height_command=a[31:32]))


def small(img: np.ndarray) -> np.ndarray:
    return cv2.resize(np.asarray(img)[..., :3], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def outcome(r: dict) -> str:
    if r["success"]:
        return "success"
    if r["fell"]:
        return "fell"
    if r["moved"]:
        return "moved_not_done"
    if r["reached"]:
        return "reached_not_moved"
    return "not_reached"


@dataclass
class Running:
    item: dict
    conf: dict
    prefix: list
    spawn_rec: list          # x, y, yaw of the recorded spawn: the frame everything is reported in
    spawn: list              # this trial's spawn (moved / turned for start trials)
    started: float = field(default_factory=time.perf_counter)
    begun: bool = False
    calls: int = 0           # agent calls so far = recorded actions + policy actions executed
    steps: int = 0
    takeover: dict | None = None
    robot_bodies: set = field(default_factory=set)
    faucet_bodies: set = field(default_factory=set)
    hand_bodies: dict = field(default_factory=dict)
    art_joints: list = field(default_factory=list)
    art_body_ids: list = field(default_factory=list)
    q0: np.ndarray | None = None
    min_hand: float = np.inf
    max_dq: float = 0.0
    min_pelvis_z: float = np.inf
    first_turn: int | None = None
    pelvis_start: list | None = None
    trace: list = field(default_factory=list)
    frames: list = field(default_factory=list)  # calibration: one row per agent call
    images: list = field(default_factory=list)  # calibration: the camera at every agent call (320x180)
    states: list = field(default_factory=list)  # calibration: the agent's 32-D state at every agent call


def main() -> int:
    args = parse_args()
    out = Path(args.out).resolve()
    plan = json.loads((out / "plan.json").read_text())
    st, slots = plan["settings"], plan["slots"]
    prompt = args.instruction or st["instruction"]
    train_dir = Path(st["train_dir"])
    train_eps = {e["episode_index"]: e for e in read_episodes(train_dir / "meta")}
    fps = json.loads((train_dir / "meta" / "info.json").read_text()).get("fps", 50)
    for d in ("traces", "videos", "takeover", "calib"):
        (out / d).mkdir(exist_ok=True)
    calib_path = out / "calib.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {"slots": {}}

    raw_files = {}
    if (out / "train_paths.json").exists():
        raw_files = {e["episode"]: Path(st["raw_dir"]) / e["raw"]
                     for e in json.loads((out / "train_paths.json").read_text())["episodes"]}

    def actions_of(ep: int) -> np.ndarray:
        """The teleop recording's commands, every frame, in the policy's 36-D layout. The train pack holds the same
        commands one frame later (its action k is raw command k+1) and stops 4 frames after the faucet opened in the
        replay it was rendered from; the recording starts at the first command and goes on a little longer."""
        src = raw_files[ep]
        names = json.loads((src.parents[2] / "meta" / "info.json").read_text())["features"]["action"]["names"]
        df = pd.read_parquet(src, columns=["action", "teleop.navigate_command", "teleop.base_height_command"])
        a = np.stack(df["action"].values)
        ix = {n: i for i, n in enumerate(names)}
        joints = ([f"left_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0",
                                                     "index_1")]
                  + [f"right_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0",
                                                        "middle_1")]
                  + [f"{side}_{j}_joint" for side in ("left", "right") for j in
                     ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
                  + ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"])
        height = np.asarray([np.atleast_1d(v)[0] for v in df["teleop.base_height_command"].values], dtype=float)
        acts = np.concatenate([a[:, [ix[n] for n in joints]], height[:, None],
                               np.stack(df["teleop.navigate_command"].values)], axis=1).astype(np.float32)
        tr = np.stack(pd.read_parquet(train_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet",
                                      columns=["action"])["action"].values)
        n = min(len(tr), len(acts) - 1)
        err = float(np.abs(acts[1:n + 1] - tr[:n]).max())
        if err > 1e-4:
            print(f"[sweep] WARNING: episode {ep}: recording and train pack commands differ by {err:.4f}", flush=True)
        return acts

    def slot_status() -> tuple[list, list]:
        """(slots still to calibrate as (slot, next episode), slots with no episode left)"""
        pending, dead = [], []
        # an episode is used once, and a failed replay counts against it in every slot (neighbouring slots share
        # backups)
        used = {c["episode"] for c in calib["slots"].values() if c.get("ok")}
        tries = [a["episode"] for c in calib["slots"].values() for a in c.get("attempts", []) if a.get("reason")]
        for s in slots:
            c = calib["slots"].get(str(s["slot"]), {})
            if c.get("ok"):
                continue
            left = [e for e in [s["episode"], *s["backups"]] if e not in used and tries.count(e) < TRIES]
            (pending if left else dead).append((s["slot"], left[0] if left else None))
        return pending, dead

    # ------------------------------------------------------------------ what this process runs
    raw_paths = {}
    if (out / "train_paths.json").exists():
        for e in json.loads((out / "train_paths.json").read_text())["episodes"]:
            raw_paths[e["episode"]] = {q[0]: q[1:] for q in e["path"]}

    if args.calib_status or args.calibrate:
        pending, dead = slot_status()
        ready = [s["slot"] for s in slots if calib["slots"].get(str(s["slot"]), {}).get("ok")]
        print(f"[sweep] calibration: slots ready {ready}" + (f", to do {[s for s, _ in pending]}" if pending else "")
              + (f", out of episodes {[s for s, _ in dead]}" if dead else ""), flush=True)
        if args.calib_status or not pending:
            return 4 if dead else 3 if pending else 0
        todo = [dict(kind="replay", slot=s, episode=e, label=f"cal_s{s}_ep{e}") for s, e in pending]
        n_before, n_all = 0, len(todo)
    else:
        pending, dead = slot_status()
        if pending or dead:
            print(f"[sweep] slots {[s for s, _ in pending + dead]} are not calibrated; run with --calibrate first",
                  flush=True)
            return 2
        trials = plan["trials"][:args.trials] if args.trials > 0 else plan["trials"]
        results_path = out / "results.jsonl"
        done = set()
        if results_path.exists():
            for line in results_path.read_text().splitlines():
                try:
                    done.add(json.loads(line)["trial"])
                except (json.JSONDecodeError, KeyError):
                    pass
        todo = [t for t in trials if t["trial"] not in done]
        n_before, n_all = len(trials) - len(todo), len(trials)
        print(f"[sweep] {len(trials)} trials, {n_before} already in {results_path.name}, {len(todo)} to go"
              + (f", {min(args.max_new, len(todo))} in this process" if 0 < args.max_new < len(todo) else ""),
              flush=True)
        if args.max_new > 0:
            todo = todo[:args.max_new]
    if not todo:
        return 0

    def setup(item: dict) -> Running:
        slot = str(item["slot"])
        ep = item["episode"] if item["kind"] == "replay" else calib["slots"][slot]["episode"]
        item = dict(item, episode=ep)
        conf = json.loads(train_eps[ep]["environment_config"])  # the recorded scene, every part of it
        robot = conf["dr_state_dict"]["spatial"][conf["robot_cfg"]["uid"]]
        rx, ry, rz = robot["position"]
        ryaw = yaw_of(robot["quaternion"])
        spawn_rec = [rx, ry, ryaw]
        spawn = list(spawn_rec)
        if item["kind"] == "start":
            c, s = math.cos(ryaw), math.sin(ryaw)
            spawn = [rx + c * item["dx"] - s * item["dy"], ry + s * item["dx"] + c * item["dy"],
                     ryaw + math.radians(item["dyaw"])]
            robot["position"] = [spawn[0], spawn[1], rz]
        conf["_traj"] = dict(yaw=spawn[2], label=item["label"])
        acts = actions_of(ep)
        if item["kind"] == "replay":
            hold = acts[-1].copy()
            hold[32:35] = 0.0  # stand still once the recording ends
            prefix = list(acts) + [hold] * HOLD
        elif item["kind"] == "path":
            t = calib["slots"][slot]["frames"][item["point"]]
            item["frame"] = t
            prefix = list(acts[:t])
        else:
            item["frame"] = 0
            prefix = []
        return Running(item=item, conf=conf, prefix=prefix, spawn_rec=spawn_rec, spawn=spawn)

    if args.calibrate:  # built as the process goes: each next episode is the next slot still to calibrate
        runs: list[Running] = []
        n_runs = sum(TRIES * (1 + len(s["backups"])) for s in slots)
        longest = max(train_eps[e]["length"] for s in slots for e in [s["episode"], *s["backups"]])
    else:
        runs = [setup(t) for t in todo]
        n_runs = len(runs)
    server_info = {} if args.calibrate else E._fetch_policy_info(args.host, args.port)
    if not args.calibrate:
        (out / "conditions.json").write_text(json.dumps(dict(
            settings=st, args=vars(args), instruction=prompt, server_info=server_info,
            thresholds=dict(reach=REACH_DIST, moved_rad=MOVED_RAD, fall=FALL_Z)), indent=1))

    # ------------------------------------------------------------------ hooks into SIMPLE's eval loop
    state: dict = {"cur": None, "env": None, "n_done": 0, "t0": time.perf_counter(), "idx": 0}

    class _Plan:  # stands in for the LeRobotDataset the loop opens: one "episode" per run
        def __init__(self, *a, **kw):
            self.num_episodes = n_runs
            self.meta = SimpleNamespace(fps=fps)

    import lerobot.datasets.lerobot_dataset as LD
    LD.LeRobotDataset = _Plan

    def get_episode(_ds, idx, data_format=None):
        if args.calibrate:
            pending, dead = slot_status()
            if not pending:  # every slot calibrated (or out of episodes): end here, without another episode
                ready = [s["slot"] for s in slots if calib["slots"].get(str(s["slot"]), {}).get("ok")]
                print(f"[sweep] calibration finished: slots ready {ready}"
                      + (f", out of episodes {[s for s, _ in dead]}" if dead else ""), flush=True)
                sys.stderr.flush()
                os._exit(0)
            s, e = pending[0]
            n = [a["episode"] for a in calib["slots"].get(str(s), {}).get("attempts", [])].count(e)
            runs.append(setup(dict(kind="replay", slot=s, episode=e, label=f"cal_s{s}_ep{e}_{n}")))
            idx = len(runs) - 1
        cur = runs[idx]
        cur.started = time.perf_counter()
        state["cur"] = cur
        return cur.conf, None

    SL.get_episode_lerobot = get_episode

    orig_load = M.DRManager.load_state_dict

    def load_state_dict(self, state_dict, dr_level=None):
        tr = state_dict.get("_traj")
        if tr is not None:  # the spawn heading: SIMPLE's spatial DR keeps a recorded robot position, not its heading
            q = quat_z(tr["yaw"])
            self.get_randomizer("spatial").cfg.robot_orientation_region = Box(low=q, high=q)
        return orig_load(self, state_dict, dr_level=dr_level)

    M.DRManager.load_state_dict = load_state_dict

    orig_get_action = Psi0DecoupledWbcAgent.get_action

    def get_action(self, observation, instruction=None, **kw):
        cur = state["cur"]
        if cur is not None:
            if not cur.begun:
                cur.begun = True
                try:
                    _begin(cur)
                except Exception:
                    traceback.print_exc()
                for a in cur.prefix:
                    queue_recorded(self, a)
            if cur.item["kind"] == "replay":
                try:
                    _calib_frame(cur, self, observation)
                except Exception:
                    traceback.print_exc()
                if len(self._action_queue) == 0:
                    raise StopIteration  # the recording and the hold are over
            elif cur.takeover is None and len(self._action_queue) == 0:
                try:
                    _takeover(cur, self, observation)
                except Exception:
                    traceback.print_exc()
            cur.calls += 1
        return orig_get_action(self, observation, instruction=prompt, **kw)

    Psi0DecoupledWbcAgent.get_action = get_action

    def _begin(cur: Running) -> None:
        env = state["env"]
        m, d = env.mjModel, env.mjData
        cur.robot_bodies = subtree(m, m.body("pelvis").id)
        try:
            cur.faucet_bodies = subtree(m, m.body("articulate_base").id)
        except KeyError:
            cur.faucet_bodies = set()
        for side in ("left", "right"):
            ids = []
            for b in HAND_BODIES:
                try:
                    ids.append(m.body(f"{side}_{b}").id)
                except KeyError:
                    pass
            cur.hand_bodies[side] = ids or [m.body(f"{side}_wrist_yaw_link").id]
        cur.art_joints = [i for i in range(m.njnt) if m.joint(i).name.startswith("articulate_joint")]
        cur.art_body_ids = sorted({int(m.jnt_bodyid[i]) for i in cur.art_joints}) or sorted(cur.faucet_bodies)
        cur.q0 = np.array([d.qpos[m.jnt_qposadr[i]] for i in cur.art_joints])
        pelvis = d.body("pelvis")
        cur.pelvis_start = [*to_frame(cur.spawn_rec, pelvis.xpos[:2]),
                            math.degrees(yaw_of(pelvis.xquat) - cur.spawn_rec[2])]

    def _touching(cur: Running, m, d) -> bool:
        for i in range(d.ncon):
            b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
            if (b1 in cur.robot_bodies and b2 in cur.faucet_bodies) or (b2 in cur.robot_bodies and b1 in cur.faucet_bodies):
                return True
        return False

    def _scene_contacts(cur: Running, m, d) -> list[str]:
        """Bodies (not the floor, not the robot) the robot touches."""
        names = set()
        for i in range(d.ncon):
            b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
            for a, b in ((b1, b2), (b2, b1)):
                if a in cur.robot_bodies and b not in cur.robot_bodies and b != 0:
                    names.add(m.body(b).name)
        return sorted(names)

    def _faucet_travel(cur: Running, m, d) -> float:
        if not cur.art_joints:
            return 0.0
        q = np.array([d.qpos[m.jnt_qposadr[i]] for i in cur.art_joints])
        return float(np.abs(q - cur.q0).max())

    def _pose(cur: Running, d) -> list[float]:
        pelvis = d.body("pelvis")
        return [*to_frame(cur.spawn_rec, pelvis.xpos[:2]), math.degrees(yaw_of(pelvis.xquat) - cur.spawn_rec[2])]

    def _calib_frame(cur: Running, agent, obs) -> None:
        env = state["env"]
        m, d = env.mjModel, env.mjData
        cur.frames.append([cur.calls, *_pose(cur, d), _faucet_travel(cur, m, d), int(_touching(cur, m, d))])
        cur.images.append(small(obs["head_stereo_left"]))
        cur.states.append(agent_state(agent, obs).round(5).tolist())

    def _takeover(cur: Running, agent, obs) -> None:
        env = state["env"]
        m, d = env.mjModel, env.mjData
        img = small(obs["head_stereo_left"])
        save_png(out / "takeover" / f"{cur.item['label']}.png", img)
        cur.takeover = dict(call=cur.calls, state=agent_state(agent, obs).round(5).tolist(),
                            pose=[round(v, 4) for v in _pose(cur, d)],
                            faucet_travel=round(_faucet_travel(cur, m, d), 4),
                            contacts=_scene_contacts(cur, m, d))

    orig_step = SonicLocoManipEnv.step

    def step(self, action):
        res = orig_step(self, action)
        state["env"] = self
        cur = state["cur"]
        if cur is not None and cur.begun:
            try:
                _record(self, cur)
            except Exception:  # never let bookkeeping kill the eval
                traceback.print_exc()
        return res

    orig_reset = SonicLocoManipEnv.reset

    def _reset_hook(self, *a, **kw):
        state["env"] = self
        return orig_reset(self, *a, **kw)

    SonicLocoManipEnv.reset = _reset_hook
    SonicLocoManipEnv.step = step

    def _record(env, cur: Running) -> None:
        m, d = env.mjModel, env.mjData
        cur.steps += 1
        hands = {s: d.xpos[ids].mean(axis=0) for s, ids in cur.hand_bodies.items()}
        pts = d.xpos[cur.art_body_ids] if cur.art_body_ids else None
        if pts is not None:
            for h in hands.values():
                cur.min_hand = min(cur.min_hand, float(np.linalg.norm(pts - h, axis=1).min()))
        cur.max_dq = max(cur.max_dq, _faucet_travel(cur, m, d))
        pelvis = d.body("pelvis")
        cur.min_pelvis_z = min(cur.min_pelvis_z, float(pelvis.xpos[2]))
        lb = getattr(getattr(state.get("agent_obj"), "_wbc_policy", None), "lower_body_policy", None)
        cmd = np.asarray(getattr(lb, "cmd", np.zeros(3)), dtype=float)[:3] if lb is not None else np.zeros(3)
        if cur.takeover is not None and cur.first_turn is None and abs(cmd[2]) > 1e-6:
            cur.first_turn = cur.steps
        if cur.steps % TRACE_EVERY == 1:
            ph = _pose(cur, d)
            rel_h = [to_frame(cur.spawn_rec, hands[s][:2]) + [float(hands[s][2])] for s in ("left", "right")]
            cur.trace.append([cur.steps, *np.round(ph[:2], 4).tolist(), round(float(pelvis.xpos[2]), 4), round(ph[2], 2),
                              *np.round(rel_h[0], 4).tolist(), *np.round(rel_h[1], 4).tolist(),
                              round(cur.max_dq, 4), *np.round(cmd, 3).tolist()])

    _orig_init = Psi0DecoupledWbcAgent.__init__

    def _init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        state["agent_obj"] = self

    Psi0DecoupledWbcAgent.__init__ = _init

    def append_stats(eval_dir, line):
        E_append(eval_dir, line)
        if line.startswith("episode_"):
            try:
                _finish()
            except Exception:
                print("[sweep] WARNING: could not record this run:", flush=True)
                traceback.print_exc()

    E_append = E._append_eval_stats_line
    E._append_eval_stats_line = append_stats

    def _finish_calibration(cur: Running, success: bool) -> dict:
        it = cur.item
        fr = np.array(cur.frames, dtype=float)  # call, x, y, yaw, travel, touching
        L = len(fr)
        turning = np.where(fr[:, 4] > TURN_RAD)[0]
        touching = np.where(fr[:, 5] > 0)[0]
        rec = dict(episode=it["episode"], success=success, frames_recorded=L, length=train_eps[it["episode"]]["length"],
                   first_touch=int(touching[0]) if len(touching) else None, max_travel=round(float(fr[:, 4].max()), 3),
                   wall_seconds=round(time.perf_counter() - cur.started, 1))
        if not len(turning):
            rec["reason"] = "the faucet handle never turned"
            return rec
        turn = int(turning[0])
        p3 = max(1, turn - int(st["reach_lead"]))
        k = 25  # smooth over half a second: the gait sways the pelvis a few cm per step
        xy = fr[:p3 + 1, 1:3]
        pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
        sm = np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in range(2)], axis=1)[:len(xy)]
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(sm, axis=0), axis=1))]
        total = float(cum[-1])
        away = np.where(np.linalg.norm(xy - xy[0], axis=1) >= 0.03)[0]
        p1 = int(np.searchsorted(cum, total / 3))
        p1 = max(p1, int(away[0]) if len(away) else p1)
        p2 = max(int(np.searchsorted(cum, 2 * total / 3)), p1 + 10)
        rec.update(turn_start=turn, frames=dict(p1=p1, p2=p2, p3=p3), path_length=round(total, 4),
                   points={p: [round(v, 4) for v in fr[f, 1:4]] for p, f in (("p0", 0), ("p1", p1), ("p2", p2),
                                                                           ("p3", p3))},
                   path=[[int(i), *np.round(fr[i, 1:4], 4).tolist()] for i in range(0, L, TRACE_EVERY)],
                   states={p: cur.states[f] for p, f in (("p0", 0), ("p1", p1), ("p2", p2), ("p3", p3))})
        raw = raw_paths.get(it["episode"], {})
        drift = {}
        for p, f in (("p1", p1), ("p2", p2), ("p3", p3)):
            if raw:  # the recording's sample nearest this frame (its frame t is raw frame t+1)
                q = raw[min(raw, key=lambda g: abs(g - (f - 1)))]
                drift[p] = [round(100 * math.hypot(fr[f, 1] - q[0], fr[f, 2] - q[1]), 1), round(fr[f, 3] - q[2], 1)]
        rec["drift"] = drift
        worst = max(drift.values(), key=lambda v: v[0] / MAX_DRIFT_CM + abs(v[1]) / MAX_DRIFT_DEG, default=None)
        if not success:
            rec["reason"] = "the replay did not open the faucet"
        elif p3 <= p2 + 5 or total < 0.06:
            rec["reason"] = f"walk too short before the handle turns ({100 * total:.0f} cm)"
        elif worst and (worst[0] > MAX_DRIFT_CM or abs(worst[1]) > MAX_DRIFT_DEG):
            rec["reason"] = f"the replay drifted {worst[0]:.1f} cm, {worst[1]:+.1f} deg from the recording"
        for p, f in (("p0", 0), ("p1", p1), ("p2", p2), ("p3", p3)):
            for j in (f - 1, f, f + 1):
                if 0 <= j < L:
                    save_png(out / "calib" / f"s{it['slot']}_ep{it['episode']}_f{j:04d}.png", cur.images[j])
        return rec

    def _finish() -> None:
        cur, state["cur"] = state["cur"], None
        env = state["env"]
        it = cur.item
        success = bool(getattr(env, "_success", False))
        state["last_label"] = it["label"]
        if it["kind"] == "replay":
            rec = _finish_calibration(cur, success)
            c = calib["slots"].setdefault(str(it["slot"]), {"attempts": []})
            c["attempts"].append({k: v for k, v in rec.items() if k not in ("path", "states")})
            ok = "reason" not in rec
            if ok:
                c.update(ok=True, **rec, video_dir=f"videos/{it['label']}")
            tmp = calib_path.with_name("calib.json.tmp")
            tmp.write_text(json.dumps(calib))
            tmp.replace(calib_path)
            fr = rec.get("frames") or {}
            print(f"[sweep] calibration slot {it['slot']} episode {it['episode']}: "
                  + (f"ok, handle turning from frame {rec['turn_start']}, path points {fr.get('p1')}, {fr.get('p2')}, "
                     f"{fr.get('p3')} ({100 * rec['path_length']:.0f} cm walked; off the recording by "
                     + ", ".join(f"{v[0]:.1f} cm {v[1]:+.1f} deg" for v in rec.get("drift", {}).values()) + ")"
                     if ok else f"rejected: {rec['reason']}")
                  + f" ({rec['wall_seconds']:.0f} s wall)", flush=True)
            return
        to = cur.takeover or {}
        last = cur.trace[-1] if cur.trace else None
        rec = dict(
            trial=it["trial"], label=it["label"], kind=it["kind"], cond=it["cond"], group=it["group"],
            name=it["name"], slot=it["slot"], episode=it["episode"], rep=it["rep"], point=it.get("point"),
            frame=it.get("frame"), dx=it.get("dx", 0.0), dy=it.get("dy", 0.0), dyaw=it.get("dyaw", 0.0),
            spawn=[round(v, 4) for v in cur.spawn], spawn_rec=[round(v, 4) for v in cur.spawn_rec],
            pelvis_start=None if cur.pelvis_start is None else [round(v, 4) for v in cur.pelvis_start],
            takeover=to, success=success, steps=cur.steps,
            policy_steps=max(0, cur.steps - int(to.get("call", 0))), budget_steps=args.budget_steps or None,
            min_hand_dist=None if not np.isfinite(cur.min_hand) else round(cur.min_hand, 4),
            reached=bool(cur.min_hand < REACH_DIST), max_joint_travel=round(cur.max_dq, 4),
            moved=bool(cur.max_dq > MOVED_RAD), fell=bool(cur.min_pelvis_z < FALL_Z),
            min_pelvis_z=None if not np.isfinite(cur.min_pelvis_z) else round(cur.min_pelvis_z, 4),
            final_pose=None if last is None else [last[1], last[2], last[4]],
            first_turn_step=cur.first_turn, instruction=prompt,
            video_dir=f"videos/{it['label']}" if args.save_video else None,
            wall_seconds=round(time.perf_counter() - cur.started, 1))
        rec["outcome"] = outcome(rec)
        (out / "traces" / f"{it['label']}.json").write_text(json.dumps(dict(
            columns=["step", "x", "y", "z", "yaw", "lh_x", "lh_y", "lh_z", "rh_x", "rh_y", "rh_z", "faucet_travel",
                     "cmd_vx", "cmd_vy", "cmd_vyaw"], rows=cur.trace)))
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        state["n_done"] += 1
        el = time.perf_counter() - state["t0"]
        n_now = n_before + state["n_done"]
        eta = el / state["n_done"] * (n_all - n_now)
        eta = f"{eta / 3600:.1f} h" if eta > 5400 else f"{eta / 60:.0f} min"
        where = f"slot {it['slot']} (ep {it['episode']}), " + (f"frame {it['frame']}" if it["kind"] == "path"
                                                                else "start")
        print(f"[sweep] {n_now}/{n_all} {it['label']} {it['name']} ({where}): {rec['outcome']} "
              f"({cur.steps} steps, policy from step {to.get('call', '?')}, closest hand "
              f"{rec['min_hand_dist'] or float('nan'):.2f} m, faucet {cur.max_dq:.2f} rad, "
              f"{rec['wall_seconds']:.0f} s wall)  eta {eta}", flush=True)

    class Recorder(E.VideoRecorder):
        def release(self):
            was = self._is_released
            super().release()
            label = state.get("last_label")
            if was or not label:
                return
            src = Path(self.work_dir) / self.name_prefix
            if src.is_dir():
                dst = out / "videos" / label
                shutil.rmtree(dst, ignore_errors=True)
                shutil.move(str(src), str(dst))

    E.VideoRecorder = Recorder

    budget = args.budget_steps or None
    if args.calibrate:  # the replay must never run out of budget: the longest episode plus the hold
        budget = max(budget or 0, longest + 1 + HOLD + 10)
    E._run_eval_worker(
        env_id=f"simple/{st['task']}", policy="psi0_decoupled_wbc", split="train", host=args.host, port=args.port,
        data_format="lerobot", sim_mode="mujoco_isaac", headless=args.headless, eval_dir=str(out / "eval"),
        max_episode_steps=budget, num_episodes=n_runs, episode_start=0, data_dir=str(out), rollout_save_dir=None,
        success_criteria=None, save_video=args.save_video)
    if args.calibrate:
        pending, dead = slot_status()
        print(f"[sweep] calibration round done: {len(slots) - len(pending) - len(dead)}/{len(slots)} slots ready",
              flush=True)
        return 4 if dead else 3 if pending else 0
    print(f"[sweep] finished {state['n_done']} trials in {(time.perf_counter() - state['t0']) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
