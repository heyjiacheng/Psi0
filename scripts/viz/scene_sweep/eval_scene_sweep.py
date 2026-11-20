"""Stage 2 of the scene sweep: closed-loop trials of the served checkpoint, one per scene of plan.json. SIMPLE's venv.

Everything but the scene is the target-pose sweep's driver (../target_pose_sweep/eval_target_pose.py, same command
line): SIMPLE's own Teleop eval loop with the psi0_decoupled_wbc agent, its success test (task.check_success: a
hand turned the handle past 0.7 rad the way the demos do), videos, the 10 Hz trace, results.jsonl and resuming.
What differs:
  - each trial loads its scene file (make_scene_trials.py) exactly as written: nothing is re-drawn, so trial i
    has the same randomization in every set;
  - a scene with scene_sweep.physics_table (c2_2) gets that table in MuJoCo, while Isaac draws the scene's table:
    the part of the drawn table in front of the physics table is not there for the robot;
  - a scene with scene_sweep.texture_table (c3) has its table's texture laid as on that table (the recorded one):
    the recorded part of a wider table looks as it did and the added part continues the texture, where SIMPLE
    would stretch it (Kit's cube has texture coordinates 0..1 on every face, whatever its size);
  - for both, Isaac's own table collider (Isaac steps PhysX after every MuJoCo substep, and the policy's joint
    positions are read from Isaac) is an invisible box equal to MuJoCo's table, not the drawn one;
  - a drawn table that is round, polygonal or turned against its physics table (t4, t5) gets its texture laid
    the same way, from where each face vertex is in the texture table's frame (Kit's cube mapping, by the face's
    dominant normal); without a physics table (the _phys twins) it keeps its own collider, MuJoCo's twin;
  - a scene with scene_sweep.untouchable_faucet (f3: the faucet floats in front of the table, fixed to the world)
    has a faucet nothing collides with: the robot passes through it, as it passes through the part of a table drawn
    closer than the recorded one (t1), so the faucet cannot stop its walk (in the closest scenes it sits between the
    fingers from the start); the handle cannot be turned either. Isaac has no faucet collider (SIMPLE turns them off);
  - a trial in another room (r3) unloads the rooms other than its own and the base scenes' one first, so an Isaac
    process holds at most two;
  - its results.jsonl record also holds its set, index and change (value), how the faucet handle was turned
    (handle_check), the control steps in which a robot body touched the (physics) table and which bodies
    (table_contact_steps, table_contact_bodies), the same for anything else but the floor (object_contact:
    for each robot body, the first control step it touched one and in how many; the things touched), and two more outcomes: "wrong_way" (turned past 0.7 rad the other
    way) and "arm_or_body" (turned past 0.7 rad the right way, but an arm or body link touched the handle while it
    turned);
  - and its gait: the control steps in which the legs ran gear_wbc's walking policy rather than its standing one
    (the controller's own switch, on the command the policy sent), as [start, end) runs of policy steps with each
    run's mean command (vx, vy, turn flag), and the pelvis every GAIT_EVERY steps in the robot's start frame
    (gait.walking, gait.cmd, gait.pelvis; the report finds where the robot stopped walking from them).

    python eval_scene_sweep.py --out OUT [--trials N] [--max-new N] [--host H] [--port P] [--headless] ...
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "target_pose_sweep"))
import eval_target_pose as T  # noqa: E402
import decoupled_wbc.control.policy.g1_gear_wbc_policy as GW  # noqa: E402
import simple.engines.mujoco as MJ  # noqa: E402
from simple.assets.primitive import Box  # noqa: E402
from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent  # noqa: E402
from simple.envs.sonic_loco_manip import SonicLocoManipEnv  # noqa: E402
from simple.tasks.g1_wholebody_open_faucet_teleop import _OPEN_ANGLE  # noqa: E402

GAIT_EVERY = 5  # policy steps (50 Hz) between pelvis samples of the gait record
CONTROL_HZ = 50
OUT = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
PLAN = {t["trial"]: t for t in json.loads((OUT / "plan.json").read_text())["trials"]}
BASE_ROOM = "hssd:" + json.loads((OUT / "room.json").read_text())["room"] if (OUT / "room.json").exists() else None
current: dict = {}


def virtual_episode(base: dict, item: dict, st: dict) -> dict:
    """The trial's scene as written, with the bookkeeping eval_target_pose.py reads from it."""
    conf = json.loads((OUT / item["conf"]).read_text())
    spatial = conf["dr_state_dict"]["spatial"]
    robot, target = spatial[conf["robot_cfg"]["uid"]], spatial[st["target"]]
    conf["_sweep"] = dict(seed=item["seed"], kind=st["kind"], xy=target["position"][:2],
                          yaw=T.yaw_of(target["quaternion"]),
                          robot=[*robot["position"][:2], T.yaw_of(robot["quaternion"])])
    return conf


def outcome(rec: dict) -> str:
    item = PLAN[rec["trial"]]
    env = current.get("env")
    check = getattr(env.task, "handle_check", None) if env is not None else None
    steps, bodies, other = current.get("contacts", (set(), set(), {}))
    try:
        gait = gait_record(env, rec)
    except Exception:  # never let bookkeeping lose the trial
        traceback.print_exc()
        gait = None
    rec.update(set=item["set"], index=item["index"], value=item["value"], handle_check=check,
               table_contact_steps=len(steps), table_contact_bodies=sorted(bodies),
               object_contact=dict(bodies={k: [min(v), len(v)] for k, v in sorted(other.items()) if k != "_touched"},
                                   touched=sorted(other.get("_touched", ()))), gait=gait)
    if not rec["success"] and check is not None:
        if check["wrong_way_angle"] > _OPEN_ANGLE:
            return "wrong_way"
        if check["open_angle"] > _OPEN_ANGLE and check["turned_by_arm_or_body"]:
            return "arm_or_body"
    return base_outcome(rec)


def gait_record(env, rec: dict) -> dict | None:
    """The trial's gait (see the module docstring), or None if the policy never ran."""
    gait = current.get("gait")
    if not gait or not gait["walk"]:
        return None
    path = list(gait["path"])
    if env is not None and (not path or path[-1][0] != gait["n"]):  # the last step too
        path.append([gait["n"], *env.mjData.body("pelvis").xpos[:2].tolist()])
    r0 = (rec.get("placed") or {}).get("robot")
    if r0:  # the robot's start frame, as every position in the record
        path = [[k, *(round(v, 4) for v in T.to_robot_frame(r0[:2], r0[2], [x, y]))] for k, x, y in path]
    flags = np.r_[0, np.asarray(gait["walk"], dtype=int), 0]
    runs = np.flatnonzero(np.diff(flags)).reshape(-1, 2).tolist()
    cmd = np.asarray(gait["cmd"], dtype=float)
    return dict(hz=CONTROL_HZ, steps=len(gait["walk"]), walking=runs,
                cmd=[np.round(cmd[a:b].mean(axis=0), 3).tolist() for a, b in runs], pelvis=path,
                frame="start" if r0 else "world")


def gear_get_action(self, *args, **kwargs):
    """gear_wbc's lower-body step; notes whether it ran the walking or the standing policy (its own test)."""
    res = base_gear_get_action(self, *args, **kwargs)
    gait = current.get("gait")
    if gait is not None and gait["started"]:
        nav = np.asarray(res["navigate_cmd"], dtype=float)  # the (vx, vy, turn flag) its policy switch tested
        if os.environ.get("MODIFIED_POLICY_SWITCH_BEHAVIOR", "0").lower() in ("1", "true"):
            stand = np.linalg.norm(nav[:2]) < 0.05 and abs(nav[2]) < 0.1
        else:
            stand = np.linalg.norm(nav[:3]) < 0.1
        gait["walk"].append(not stand)
        gait["cmd"].append(nav[:3])
    return res


def agent_get_action(self, *args, **kwargs):
    """The policy's step: from the first one on, the gait is recorded."""
    if current.get("gait") is not None:
        current["gait"]["started"] = True
    return base_agent_get_action(self, *args, **kwargs)


def step(self, action):
    res = base_step(self, action)
    gait = current.get("gait")
    if gait is not None and gait["started"]:
        gait["n"] += 1
        if gait["n"] % GAIT_EVERY == 1:
            gait["path"].append([gait["n"], *self.mjData.body("pelvis").xpos[:2].tolist()])
    return res


def build_primitive(self, spec, world, actor, table_name: str = "table"):
    """MuJoCo's table: scene_sweep.physics_table when the trial has one, else the scene's."""
    box = current.get("physics_table") if table_name == "table" else None
    if box is not None:
        actor = Box(list(box["size"]), list(box["pose"]["position"]), list(box["pose"]["quaternion"]))
    return base_build_primitive(self, spec, world, actor, table_name=table_name)


def box_of(b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, size and wxyz quaternion of a table (any simple.assets.primitive table, or a scene_sweep box dict)."""
    pose, size = (SimpleNamespace(**b["pose"]), b["size"]) if isinstance(b, dict) else (b.pose, b.size)
    return (np.asarray(pose.position, dtype=float), np.asarray(size, dtype=float),
            np.asarray(pose.quaternion, dtype=float))


def kit_cube_st(normal, p: np.ndarray) -> np.ndarray:
    """Texture coordinates Kit's unit cube (omni.kit.primitive.mesh, evaluators/cube.py) gives the face with this
    normal, at the points p of its frame."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return {(0, 0, 1): np.c_[x + 0.5, y + 0.5], (0, 0, -1): np.c_[0.5 - x, y + 0.5],
            (0, 1, 0): np.c_[x + 0.5, 0.5 - z], (0, -1, 0): np.c_[x + 0.5, z + 0.5],
            (-1, 0, 0): np.c_[z + 0.5, y + 0.5], (1, 0, 0): np.c_[0.5 - z, y + 0.5]}[tuple(int(round(v)) for v in normal)]


def setup_table(self, prim_path: str, table_box) -> None:
    """Isaac's table as SIMPLE draws it; redrawn (redraw_table) for a trial with a physics or texture table."""
    base_setup_table(self, prim_path, table_box)
    if not prim_path.endswith("/table_cuboid"):
        return
    import omni.usd
    redraw_table(omni.usd.get_context().get_stage(), prim_path, table_box, current.get("texture_table"),
                 current.get("physics_table"))


def redraw_table(stage, prim_path: str, table_box, texture: dict | None, physics: dict | None) -> None:
    """Re-mesh the drawn table (Kit's unit cube scaled to `table_box`) in the texture table's frame, so the texture
    lies on it as on that table (the drawn table's own when None), take its collider off, and put an invisible box
    collider where MuJoCo's table is (`physics`, or the drawn table). Nothing changes when both are None, except
    that an earlier trial's collider box goes."""
    from pxr import Gf, UsdGeom, UsdPhysics, Vt
    if stage.GetPrimAtPath(prim_path + "_collider").IsValid():
        stage.RemovePrim(prim_path + "_collider")
    if physics is None and texture is None:
        return
    drawn = box_of(table_box)
    (c, s, q), (pc, ps, pq) = box_of(texture) if texture else drawn, box_of(physics) if physics else drawn
    if not (isinstance(table_box, Box) and np.allclose(q, drawn[2]) and np.allclose(pq, drawn[2])):
        redraw_shape(stage, prim_path, drawn, (c, s, q), physics)
        return
    rot = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(*map(float, q)))).GetTranspose())  # columns: the table's axes
    offset = rot.T @ (drawn[0] - c)
    lo, hi = (offset - drawn[1] / 2) / s, (offset + drawn[1] / 2) / s  # the drawn box in the texture table's frame
    prim = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(prim)
    pts = np.where(np.array(mesh.GetPointsAttr().Get(), dtype=float) < 0, lo, hi)  # Kit's cube: corners at +-0.5
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in pts]))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    index, normals = np.array(mesh.GetFaceVertexIndicesAttr().Get()), np.array(mesh.GetNormalsAttr().Get(), dtype=float)
    st, k = [], 0
    for n in mesh.GetFaceVertexCountsAttr().Get():
        st.append(kit_cube_st(normals[k], pts[index[k:k + n]]))
        k += n
    UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, v)) for v in np.concatenate(st)]))
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, c)))
    prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(*map(float, s)))
    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    cube = UsdGeom.Cube.Define(stage, prim_path + "_collider")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pc)))
    cube.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, pq)))
    cube.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, ps)))
    cube.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def quat_mat(q) -> np.ndarray:
    """Rotation matrix of a wxyz quaternion."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def redraw_shape(stage, prim_path: str, drawn, texture, physics: dict | None) -> None:
    """redraw_table for a round or polygonal drawn table, or one turned against its texture or physics table: its
    mesh stays; each face vertex gets the texture coordinates Kit's cube would give it, with the texture table as the
    cube and the face's normal rounded to its dominant axis of that table. With a physics table, the drawn table's
    collider goes and an invisible box collider stands where MuJoCo's table is; without one it keeps its own."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt
    prim = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(prim)
    (dc, dsz, dq), (tc, tsz, tq) = drawn, texture
    rd, rt = quat_mat(dq), quat_mat(tq)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=float)  # unit-sized, scaled by the drawn size
    local = (((pts * dsz) @ rd.T + dc) - tc) @ rt / tsz  # in the texture table's frame, its box at +-0.5
    index, counts = np.array(mesh.GetFaceVertexIndicesAttr().Get()), list(mesh.GetFaceVertexCountsAttr().Get())
    normals = np.array(mesh.GetNormalsAttr().Get() or [], dtype=float)
    st, k = [], 0
    for n in counts:
        face = index[k:k + n]
        if len(normals) == len(index):  # face-varying
            nrm = normals[k:k + n].sum(0)
        elif len(normals) == len(pts):
            nrm = normals[face].sum(0)
        else:  # Newell's normal of the face
            v = pts[face]
            nrm = np.cross(v, np.roll(v, -1, axis=0)).sum(0)
        nrm = rt.T @ (rd @ (nrm / dsz))  # the normal in the texture table's frame
        axis = int(np.argmax(np.abs(nrm)))
        st.append(kit_cube_st(tuple(np.sign(nrm[axis]) * (np.arange(3) == axis)), local[face]))
        k += n
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    if not pv:
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    if pv.IsIndexed():
        pv.BlockIndices()
    pv.SetInterpolation(UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, v)) for v in np.concatenate(st)]))
    if physics is None:
        return
    pc, ps, pq = box_of(physics)
    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    cube = UsdGeom.Cube.Define(stage, prim_path + "_collider")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pc)))
    cube.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, pq)))
    cube.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, ps)))
    cube.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def unload_rooms(isaac, keep: set) -> None:
    """Take the rooms Isaac has loaded, but those in `keep`, off its stage (SIMPLE only hides a room it switches
    away from)."""
    scenes = getattr(isaac, "scenes", None)
    if not scenes:
        return
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for uid in [u for u in scenes if u not in keep]:
        prim = scenes[uid][0]
        path = prim.GetPath() if prim is not None and prim.IsValid() else \
            f"{isaac.SCENE_PRIM_PATH}/s_{uid.replace(':', '_')}"
        stage.RemovePrim(path)
        del scenes[uid]
        print(f"[sweep] unloaded room {uid}", flush=True)


def contact_counter(env):
    """A substep callback recording the control steps in which a robot body touches the table, and the bodies; and,
    for anything else but the floor (a plane), the control steps in which each robot body touched it."""
    m, d = env.mjModel, env.mjData
    table, root = m.body("table").id, m.body("pelvis").id
    robot = m.body_rootid == root
    geom_body = m.geom_bodyid
    floor = m.geom_type == 0  # mjGEOM_PLANE
    steps, bodies, other = set(), set(), {"_touched": set()}

    def callback() -> None:
        n = d.ncon
        if n == 0:
            return
        b1, b2 = geom_body[d.contact.geom1[:n]], geom_body[d.contact.geom2[:n]]
        hit = ((b1 == table) & robot[b2]) | ((b2 == table) & robot[b1])
        if hit.any():
            steps.add(env.step_count)
            bodies.update(m.body(int(b if a == table else a)).name for a, b in zip(b1[hit], b2[hit]))
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        thing = lambda g, b: (b != table) & ~robot[b] & ~floor[g]  # a geom fused into the world body still counts
        hit = (thing(g1, b1) & robot[b2]) | (thing(g2, b2) & robot[b1])
        for ga, gb, a, b in zip(g1[hit], g2[hit], b1[hit], b2[hit]):
            (r, o, og) = (int(b), int(a), int(ga)) if robot[b] else (int(a), int(b), int(gb))
            other.setdefault(m.body(r).name, set()).add(env.step_count)
            other["_touched"].add(m.geom(og).name or f"{m.body(o).name}/geom{og}")

    return callback, (steps, bodies, other)


def untouchable_faucet(env) -> int:
    """MuJoCo's collision filter so that nothing touches the faucet (the articulated object, fixed to the world): its
    collision geoms have contype 0 already, conaffinity goes to 0 too. Returns how many geoms changed."""
    m = env.mjModel
    faucet_root = m.body_rootid[m.joint("articulate_joint_0").bodyid[0]]
    faucet = (m.body_rootid[m.geom_bodyid] == faucet_root) & (m.geom_conaffinity > 0)
    assert faucet.any() and not m.geom_contype[faucet].any(), "unexpected faucet collision setup"
    m.geom_conaffinity[faucet] = 0
    return int(faucet.sum())


def reset(self, *args, **kwargs):
    global base_setup_table
    current["env"] = self
    sw = ((kwargs.get("options") or {}).get("state_dict") or {}).get("scene_sweep") or {}
    current["physics_table"], current["texture_table"] = sw.get("physics_table"), sw.get("texture_table")
    room = ((((kwargs.get("options") or {}).get("state_dict") or {}).get("dr_state_dict") or {}).get("scene") or {}).get("uid")
    if self.isaac is not None and room:
        unload_rooms(self.isaac, {room, BASE_ROOM})
    if self.isaac is not None and base_setup_table is None:  # Isaac's module exists once its app runs
        base_setup_table = type(self.isaac)._setup_table
        type(self.isaac)._setup_table = setup_table
    current["gait"] = None
    res = base_reset(self, *args, **kwargs)
    if sw.get("untouchable_faucet"):
        untouchable_faucet(self)
    self.substep_callback, current["contacts"] = contact_counter(self)
    current["gait"] = dict(started=False, walk=[], cmd=[], path=[], n=0)
    return res


base_outcome, base_reset, base_step = T.outcome, SonicLocoManipEnv.reset, SonicLocoManipEnv.step
base_build_primitive, base_setup_table = MJ.MujocoSimulator._build_primitive, None
base_gear_get_action, base_agent_get_action = GW.G1GearWbcPolicy.get_action, Psi0DecoupledWbcAgent.get_action
T.virtual_episode, T.outcome, SonicLocoManipEnv.reset = virtual_episode, outcome, reset
SonicLocoManipEnv.step = step  # under eval_target_pose's own step and get_action hooks, which it adds on top
MJ.MujocoSimulator._build_primitive = build_primitive
GW.G1GearWbcPolicy.get_action, Psi0DecoupledWbcAgent.get_action = gear_get_action, agent_get_action

if __name__ == "__main__":
    T.main()
