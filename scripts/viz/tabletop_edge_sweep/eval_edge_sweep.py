"""Stage 2 of the tabletop edge sweep: closed-loop trials of the served checkpoint, one per scene of plan.json.

Runs in SIMPLE's venv, from the SIMPLE root. The episode loop is SIMPLE's own eval for motion-planning tasks,
``simple.cli.eval._run_eval_worker`` with the ``psi0`` agent, unchanged: the same reset (with its 60 stand steps), the
same step budget (the task's max_episode_steps unless --budget-steps) and the same success test (task.check_success:
the target lifted 1.8 cm). What changes is the scene each episode starts from, how the table is drawn, and the
bookkeeping:

  - each trial loads its scene file (make_edge_trials.py) exactly as written: nothing is re-drawn, so trial i has the
    same scene in every set;
  - the table is drawn as in the training images (--look training, the default): the room's own counter (SIMPLE's
    "surface" prim, which it hides) is shown and SIMPLE's slab (its table box, with the scene's random material) is
    hidden; the slab keeps its collider, so in Isaac as in MuJoCo the robot and the objects meet the recorded table
    and nothing else (the counter has no collider). --look simple keeps SIMPLE's own drawing (for comparison);
  - a scene with edge_sweep.edge_closer != 0 has the counter's near end drawn that much closer to the robot (+) or
    farther (-): its vertices within NEAR_BAND of the near end move along the robot's heading, and their texture
    coordinates move with the texture's own gradient on each face, so the texture keeps its scale and stays where it
    was in the world; the far end, the physics, the target and the robot are unchanged;
  - its results.jsonl record holds the set, index and change, the drawn edge, the target's lift, the hands'
    approach (closest distance of the right fingers to the target; their reach: how far forward they got in the first
    REACH_WINDOW steps), which robot bodies touched the target or the table, and a 10 Hz trace.

Output (--out): results.jsonl, traces/<label>.json, videos/<label>/ (SIMPLE's camera videos), frames/<label>.png (the
policy's first head-camera image), conditions.json.

    python eval_edge_sweep.py --out OUT [--trials N] [--max-new N] [--look training|simple] [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import simple.cli.eval as E
import simple.datasets.lerobot as SL
import simple.dr.manager as M
from simple.baselines.psi0 import Psi0Agent
from simple.envs.loco_manipulation import LocoManipulationEnv

TRACE_EVERY = 5       # control steps (50 Hz) between trace rows
NEAR_BAND = 0.20      # m, counter vertices this close (along the heading) to its near end move with it
LIFT_ONSET = 0.005    # m, the target counts as rising from here
REACH_WINDOW = 150    # control steps (3 s) in which the right hand's farthest forward point is its reach
FALL_DROP = 0.30      # m, the pelvis this much below its start height counts as fallen
FINGERS = ("hand_index_1_link", "hand_middle_1_link", "hand_thumb_2_link")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the sweep directory, holding plan.json")
    p.add_argument("--trials", type=int, default=0, help="run only the first N trials of the plan (0 = all)")
    p.add_argument("--max-new", type=int, default=0, help="stop after this many trials (0 = none)")
    p.add_argument("--look", default="training", choices=["training", "simple"])
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--budget-steps", type=int, default=0, help="override the task's step budget (0 = the task's)")
    p.add_argument("--instruction", default="", help="override the instruction sent to the policy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8016)
    return p.parse_args()


def yaw_of(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_robot_frame(robot_xy, robot_yaw, xy) -> list[float]:
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    dx, dy = xy[0] - robot_xy[0], xy[1] - robot_xy[1]
    return [c * dx + s * dy, -s * dx + c * dy]


# ---------------------------------------------------------------------------------------------------- the counter
COUNTER: dict = {}  # mesh path -> its authored points, texture coordinates, faces


def counter_prim(isaac, stage):
    scene = isaac.task.layout.scene
    root = f"{isaac.SCENE_PRIM_PATH}/s_{scene.uid.replace(':', '_')}"
    return stage.GetPrimAtPath(scene.conf["surface"]["prim_path"].replace("/World", root))


def restore_counter(stage) -> None:
    """Put every counter mesh drawn so far back as authored. SIMPLE places the room from the counter's bounding box at
    each reset (IsaacSimSimulator.__update_scene: calc_surface_center), so a mesh still drawn for the last trial would
    move the whole room by half that trial's shift."""
    from pxr import Gf, UsdGeom, Vt
    for key, c in COUNTER.items():
        prim = stage.GetPrimAtPath(key)
        if not prim.IsValid():
            continue
        mesh = UsdGeom.Mesh(prim)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in c["points"]]))
        mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, c["points"].min(0))),
                                                Gf.Vec3f(*map(float, c["points"].max(0)))]))
        if c["st"] is not None:
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if pv.IsIndexed():
                pv.BlockIndices()
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, v)) for v in c["st"]]))


def draw_counter(stage, surface, closer: float, heading: np.ndarray) -> dict:
    """Show the room's counter with its near end moved `closer` m toward the robot (along `heading`, the robot's
    forward unit vector in world x/y), from its authored mesh. Returns where its near edge is drawn and its extent."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
    UsdGeom.Imageable(surface).MakeVisible()
    fwd = np.array([heading[0], heading[1], 0.0])
    report = dict(meshes=0, moved_vertices=0, collider=False)
    lo_all, hi_all = [], []
    for prim in Usd.PrimRange(surface):
        if prim.HasAPI(UsdPhysics.CollisionAPI):  # none in this room; drawn-only either way
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            report["collider"] = True
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        key = str(prim.GetPath())
        if key not in COUNTER:
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            st = None
            if pv and pv.HasValue():
                st = np.array(pv.ComputeFlattened(), dtype=float)
            COUNTER[key] = dict(points=np.array(mesh.GetPointsAttr().Get(), dtype=float), st=st,
                                interp=pv.GetInterpolation() if pv else None,
                                counts=np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=int),
                                index=np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int))
        c = COUNTER[key]
        pts = c["points"].copy()
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))  # row vectors
        lin, off = m[:3, :3], m[3, :3]
        along = (pts @ lin + off) @ fwd
        near = along < along.min() + NEAR_BAND
        gap = np.sort(along[~near])[:1]
        if gap.size and gap[0] < along.min() + NEAR_BAND + 0.05:
            print(f"[sweep] WARNING counter mesh {key}: vertices just past the near band", flush=True)
        d_local = (-closer * fwd) @ np.linalg.inv(lin)
        st = None if c["st"] is None else c["st"].copy()
        if closer != 0.0:
            pts[near] += d_local
            if st is not None and c["interp"] == UsdGeom.Tokens.faceVarying:
                k = 0
                for n in c["counts"]:
                    verts = c["index"][k:k + n]
                    moved = near[verts]
                    if moved.any() and n >= 3:
                        p0, p1, p2 = c["points"][verts[:3]]
                        s0, s1, s2 = c["st"][k:k + 3]
                        grad = np.c_[s1 - s0, s2 - s0] @ np.linalg.pinv(np.c_[p1 - p0, p2 - p0])  # 2x3, in-plane
                        st[k:k + n][moved] += grad @ d_local
                    k += n
            elif st is not None:
                print(f"[sweep] WARNING counter mesh {key}: texture coordinates are {c['interp']}, left as they are",
                      flush=True)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in pts]))
        mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
        if st is not None:
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if pv.IsIndexed():
                pv.BlockIndices()
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, v)) for v in st]))
        world = pts @ lin + off
        lo_all.append(world.min(0))
        hi_all.append(world.max(0))
        report["meshes"] += 1
        report["moved_vertices"] += int(near.sum()) if closer != 0.0 else 0
    if lo_all:
        lo, hi = np.min(lo_all, 0), np.max(hi_all, 0)
        report.update(lo=np.round(lo, 4).tolist(), hi=np.round(hi, 4).tolist())
    return report


# ---------------------------------------------------------------------------------------------------- bookkeeping
def body_front(m, d, robot0, top_z: float) -> dict:
    """How far forward (robot start frame) the robot's geometry reaches below the table top: the torso and legs
    ("body") and the arms and hands, from the corners of every robot geom's bounding box (MuJoCo's geom_aabb, in the
    geom frame; a little more than the geometry itself). A counter drawn closer than "body" would cut into the robot."""
    robot = m.body_rootid[m.geom_bodyid] == m.body("pelvis").id
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    front = {"body": -np.inf, "arms": -np.inf}
    for g in np.flatnonzero(robot):
        name = m.body(int(m.geom_bodyid[g])).name
        part = "arms" if any(k in name for k in ("shoulder", "elbow", "wrist", "hand")) else "body"
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        pts = (c + corners * h) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        low = pts[:, 2] < top_z
        if low.any():
            fx = max(to_robot_frame(robot0[:2], robot0[2], p[:2])[0] for p in pts[low])
            front[part] = max(front[part], float(fx))
    return {f"{k}_below_top": (round(v, 4) if np.isfinite(v) else None) for k, v in front.items()}


@dataclass
class Running:
    item: dict
    started: float
    policy_started: bool = False
    steps: int = 0
    robot0: list | None = None      # spawn x, y, yaw: the frame of every relative position
    target0: np.ndarray | None = None
    finger_ids: dict = field(default_factory=dict)
    max_lift: float = 0.0
    max_push: float = 0.0
    lift_onset_step: int | None = None
    min_dist: float = np.inf
    min_dist_step: int | None = None
    min_dist_point: list | None = None
    reach: float = -np.inf          # the right fingers' farthest forward position (robot frame) before the lift onset
    reach_point: list | None = None
    reach_step: int | None = None
    min_pelvis_z: float = np.inf
    touch_target: dict = field(default_factory=dict)  # robot body -> first step it touched the target
    touch_table: dict = field(default_factory=dict)   # robot body -> first step it touched the table
    trace: list = field(default_factory=list)
    placed: dict | None = None
    first_frame: np.ndarray | None = None


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    plan = json.loads((out / "plan.json").read_text())
    st = plan["settings"]
    trials = plan["trials"][:args.trials] if args.trials > 0 else plan["trials"]
    for d in ("traces", "videos", "frames"):
        (out / d).mkdir(exist_ok=True)
    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["trial"] for line in results_path.read_text().splitlines() if line.strip()}
    todo = [dict(t, label=f"t{t['trial']:03d}") for t in trials if t["trial"] not in done]
    n_before = len(trials) - len(todo)
    print(f"[sweep] {len(trials)} trials, {n_before} already in {results_path.name}, {len(todo)} to go"
          + (f", {min(args.max_new, len(todo))} in this process" if 0 < args.max_new < len(todo) else ""), flush=True)
    if args.max_new > 0:
        todo = todo[:args.max_new]
    if not todo:
        return
    confs = []
    for t in todo:
        conf = json.loads((out / t["conf"]).read_text())
        conf["_sweep"] = dict(seed=t["seed"])
        confs.append(conf)

    server_info = E._fetch_policy_info(args.host, args.port) if hasattr(E, "_fetch_policy_info") else {}
    if not server_info:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://{args.host}:{args.port}/info", timeout=5) as r:
                server_info = json.loads(r.read().decode())
        except Exception:
            server_info = {}
    (out / "conditions.json").write_text(json.dumps(dict(settings=st, args=vars(args), server_info=server_info,
                                                         near_band=NEAR_BAND, lift_onset=LIFT_ONSET,
                                                         reach_window=REACH_WINDOW), indent=1))

    state: dict = {"cur": None, "env": None, "n_done": 0, "t0": time.perf_counter(), "edge": 0.0, "drawn": None,
                   "look": args.look, "isaac_patched": False}

    # ------------------------------------------------------------------ the plan stands in for the eval dataset
    class _Plan:
        def __init__(self, *a, **kw):
            self.num_episodes = len(todo)
            self.meta = SimpleNamespace(fps=50)

    import lerobot.datasets.lerobot_dataset as LD
    LD.LeRobotDataset = _Plan

    def get_episode(_ds, idx, *a, **kw):
        state["cur"] = Running(item=todo[idx], started=time.perf_counter())
        state["heading"] = math.radians(todo[idx]["geom"]["robot"][2])  # the robot's yaw as the scene spawns it
        return confs[idx], None

    SL.get_episode_lerobot = get_episode

    orig_load = M.DRManager.load_state_dict

    def load_state_dict(self, state_dict, dr_level=None):
        sweep = state_dict.get("_sweep")
        if sweep is not None:
            random.seed(sweep["seed"])
            np.random.seed(sweep["seed"] % (2**32))
        return orig_load(self, state_dict, dr_level=dr_level)

    M.DRManager.load_state_dict = load_state_dict

    # ------------------------------------------------------------------ the table as drawn
    def setup_table(self, prim_path: str, table_box) -> None:
        state["base_setup_table"](self, prim_path, table_box)  # SIMPLE hid the counter already, and drew its slab
        if state["look"] != "training" or not prim_path.endswith("/table_cuboid"):
            return
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Imageable(stage.GetPrimAtPath(prim_path)).MakeInvisible()  # its collider stays: the recorded table
        yaw = state["heading"]
        state["drawn"] = draw_counter(stage, counter_prim(self, stage), state["edge"],
                                      np.array([math.cos(yaw), math.sin(yaw)]))

    orig_reset = LocoManipulationEnv.reset

    def reset(self, *a, **kw):
        state["env"] = self
        sd = ((kw.get("options") or {}).get("state_dict") or {})
        es = sd.get("edge_sweep") or {}
        state["edge"] = float(es.get("edge_closer", 0.0))
        state["look"] = es.get("look") or args.look  # a scene may ask for its own look
        state["drawn"] = None
        if self.isaac is not None and not state["isaac_patched"]:  # Isaac's module exists once its app runs
            state["base_setup_table"] = type(self.isaac)._setup_table
            type(self.isaac)._setup_table = setup_table
            state["isaac_patched"] = True
        if self.isaac is not None and COUNTER:  # before SIMPLE measures the counter to place the room
            import omni.usd
            restore_counter(omni.usd.get_context().get_stage())
        return orig_reset(self, *a, **kw)

    LocoManipulationEnv.reset = reset

    # ------------------------------------------------------------------ the policy's steps
    orig_get_action = Psi0Agent.get_action

    def get_action(self, observation, instruction=None, **kw):
        cur = state["cur"]
        if cur is not None and not cur.policy_started:
            cur.policy_started = True
            try:
                _start(cur, observation)
            except Exception:
                traceback.print_exc()
        return orig_get_action(self, observation, instruction=args.instruction or instruction, **kw)

    Psi0Agent.get_action = get_action

    def _start(cur: Running, observation) -> None:
        env = state["env"]
        m, d = env.mujoco.mjModel, env.mujoco.mjData
        for side in ("left", "right"):
            ids = []
            for b in FINGERS:
                try:
                    ids.append(m.body(f"{side}_{b}").id)
                except KeyError:
                    pass
            cur.finger_ids[side] = ids or [m.body(f"{side}_wrist_yaw_link").id]
        sp = cur.item["geom"]["robot"]
        cur.robot0 = [sp[0], sp[1], math.radians(sp[2])]
        tgt = np.asarray(env._get_info()["target"][:3], dtype=float)
        cur.target0 = tgt
        pelvis = d.body("pelvis")
        front = body_front(m, d, cur.robot0, top_z=float(tgt[2]) - 0.039)
        drawn = state["drawn"] or {}
        if state["look"] == "training":
            planned = cur.item["geom"]["counter"]["x"][0]  # the near edge's world x (the robot faces +x)
            err = (drawn.get("lo") or [math.nan])[0] - planned
            drawn["error"] = round(err, 4) if np.isfinite(err) else None
            if not np.isfinite(err) or abs(err) > 0.005:
                print(f"[sweep] WARNING {cur.item['label']}: counter edge drawn at {drawn.get('lo')}, planned x "
                      f"{planned:+.4f}", flush=True)
        cur.placed = dict(target_world=np.round(tgt, 4).tolist(),
                          target_rel=[round(v, 4) for v in to_robot_frame(cur.robot0[:2], cur.robot0[2], tgt[:2])],
                          pelvis_at_start=[round(v, 4) for v in to_robot_frame(cur.robot0[:2], cur.robot0[2],
                                                                               pelvis.xpos[:2])],
                          pelvis_z_at_start=round(float(pelvis.xpos[2]), 4),
                          body_front=front, drawn=drawn, instruction=env.task.instruction)
        img = observation.get("head_stereo_left")
        if img is not None:
            cur.first_frame = np.asarray(img)

    orig_step = LocoManipulationEnv.step

    def step(self, action):
        res = orig_step(self, action)
        state["env"] = self
        cur = state["cur"]
        if cur is not None and cur.policy_started and cur.robot0 is not None:
            try:
                _record(self, cur, res[-1])
            except Exception:  # never let bookkeeping kill the eval
                traceback.print_exc()
        return res

    LocoManipulationEnv.step = step

    def _record(env, cur: Running, info: dict) -> None:
        m, d = env.mujoco.mjModel, env.mujoco.mjData
        cur.steps += 1
        tgt = np.asarray(info.get("target", cur.target0), dtype=float)[:3]
        lift = float(tgt[2] - cur.target0[2])
        cur.max_lift = max(cur.max_lift, lift)
        cur.max_push = max(cur.max_push, float(np.linalg.norm(tgt[:2] - cur.target0[:2])))
        if cur.lift_onset_step is None and lift > LIFT_ONSET:
            cur.lift_onset_step = cur.steps
        hands = {s: d.xpos[ids].mean(axis=0) for s, ids in cur.finger_ids.items()}
        r = hands["right"]
        dist = float(np.linalg.norm(r - tgt))
        if dist < cur.min_dist:
            cur.min_dist, cur.min_dist_step, cur.min_dist_point = dist, cur.steps, r.round(4).tolist()
        if cur.steps <= REACH_WINDOW:  # the approach: the demos reach the can within about 1.5 s
            rr = to_robot_frame(cur.robot0[:2], cur.robot0[2], r[:2])
            if rr[0] > cur.reach:
                cur.reach, cur.reach_step = rr[0], cur.steps
                cur.reach_point = [round(rr[0], 4), round(rr[1], 4), round(float(r[2]), 4)]
        pelvis = d.body("pelvis")
        cur.min_pelvis_z = min(cur.min_pelvis_z, float(pelvis.xpos[2]))
        n = d.ncon
        if n:
            b1, b2 = m.geom_bodyid[d.contact.geom1[:n]], m.geom_bodyid[d.contact.geom2[:n]]
            root = m.body("pelvis").id
            robot = m.body_rootid == root
            tb = state.get("target_body")
            if tb is None:
                lab = str(env.task.target.asset.label)
                tb = state["target_body"] = [i for i in range(m.nbody) if lab in m.body(i).name]
                state["table_body"] = m.body("table").id
            for a, b in zip(b1, b2):
                for x, y in ((a, b), (b, a)):
                    if robot[x] and y in tb:
                        cur.touch_target.setdefault(m.body(int(x)).name, cur.steps)
                    elif robot[x] and y == state["table_body"]:
                        cur.touch_table.setdefault(m.body(int(x)).name, cur.steps)
        if cur.steps % TRACE_EVERY == 1:
            rel = lambda p: [round(v, 4) for v in to_robot_frame(cur.robot0[:2], cur.robot0[2], p[:2])]
            cur.trace.append([cur.steps, *rel(pelvis.xpos), round(float(pelvis.xpos[2]), 4),
                              *rel(hands["left"]), round(float(hands["left"][2]), 4),
                              *rel(r), round(float(r[2]), 4), *rel(tgt), round(float(tgt[2]), 4)])

    def append_stats(eval_dir, line):
        E_append(eval_dir, line)
        if line.startswith("episode_"):
            try:
                _finish()
            except Exception:
                print("[sweep] WARNING: could not record this trial:", flush=True)
                traceback.print_exc()

    E_append = E._append_eval_stats_line
    E._append_eval_stats_line = append_stats

    def _finish() -> None:
        cur, state["cur"] = state["cur"], None
        env = state["env"]
        item = cur.item
        success = bool(getattr(env, "_success", False))
        z0 = (cur.placed or {}).get("pelvis_z_at_start")
        drop = None if z0 is None or not np.isfinite(cur.min_pelvis_z) else round(z0 - cur.min_pelvis_z, 4)
        reach_err = None
        if cur.reach_point is not None and cur.placed is not None:
            reach_err = round(cur.reach_point[0] - cur.placed["target_rel"][0], 4)
        rec = dict(
            trial=item["trial"], label=item["label"], set=item["set"], index=item["index"], seed=item["seed"],
            base_episode=item["base_episode"], value=item["value"], geom=item["geom"], placed=cur.placed,
            success=success, steps=cur.steps, budget_steps=args.budget_steps or None, look=state["look"],
            max_lift=round(cur.max_lift, 4), max_push=round(cur.max_push, 4), lift_onset_step=cur.lift_onset_step,
            min_dist=None if not np.isfinite(cur.min_dist) else round(cur.min_dist, 4), min_dist_step=cur.min_dist_step,
            min_dist_point=cur.min_dist_point,
            reach=None if cur.reach_point is None else dict(point=cur.reach_point, step=cur.reach_step,
                                                            minus_target=reach_err),
            touched_target=dict(sorted(cur.touch_target.items(), key=lambda kv: kv[1])),
            touched_table=dict(sorted(cur.touch_table.items(), key=lambda kv: kv[1])),
            min_pelvis_z=None if not np.isfinite(cur.min_pelvis_z) else round(cur.min_pelvis_z, 4),
            pelvis_drop=drop, fell=bool(drop is not None and drop > FALL_DROP),
            video_dir=f"videos/{item['label']}" if args.save_video else None,
            wall_seconds=round(time.perf_counter() - cur.started, 1),
        )
        hand_touch = any("hand" in b for b in cur.touch_target)
        rec["outcome"] = ("success" if success else "lifted_low" if cur.max_lift > LIFT_ONSET
                          else "touched" if hand_touch else "pushed" if cur.max_push > 0.02
                          else "missed")
        (out / "traces" / f"{item['label']}.json").write_text(json.dumps(dict(
            columns=["step", "pelvis_x", "pelvis_y", "pelvis_z", "lh_x", "lh_y", "lh_z", "rh_x", "rh_y", "rh_z",
                     "t_x", "t_y", "t_z"], frame="robot start (x forward, y left); z world", rows=cur.trace)))
        if cur.first_frame is not None:
            try:
                from PIL import Image
                Image.fromarray(cur.first_frame[..., :3].astype(np.uint8)).save(out / "frames" / f"{item['label']}.png")
            except Exception:
                traceback.print_exc()
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        state["n_done"] += 1
        state["last_label"] = item["label"]
        el = time.perf_counter() - state["t0"]
        n_all = n_before + state["n_done"]
        eta = el / state["n_done"] * (len(trials) - n_all)
        eta = f"{eta / 3600:.1f} h" if eta > 5400 else f"{eta / 60:.0f} min"
        dr = (cur.placed or {}).get("drawn") or {}
        print(f"[sweep] {n_all}/{len(trials)} {item['label']} {item['set']} edge {item['value']['edge_closer'] * 100:+.1f} cm"
              f" (drawn x {dr.get('lo', [float('nan')])[0]:+.3f}): {rec['outcome']} ({cur.steps} steps, lift "
              f"{cur.max_lift * 100:.1f} cm, closest {rec['min_dist'] or float('nan'):.3f} m, reach-target "
              f"{(reach_err if reach_err is not None else float('nan')) * 100:+.1f} cm, {rec['wall_seconds']:.0f} s)"
              f"  eta {eta}", flush=True)

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

    E._run_eval_worker(
        env_id=f"simple/{st['task']}", policy="psi0", split="train", host=args.host, port=args.port,
        data_format="lerobot", sim_mode="mujoco_isaac", headless=args.headless, eval_dir=str(out / "eval"),
        max_episode_steps=args.budget_steps or None, num_episodes=len(todo), episode_start=0,
        data_dir=st["base_scenes"], rollout_save_dir=None, success_criteria=0.9, save_video=args.save_video)
    print(f"[sweep] finished {state['n_done']} trials in {(time.perf_counter() - state['t0']) / 60:.1f} min; "
          f"results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
