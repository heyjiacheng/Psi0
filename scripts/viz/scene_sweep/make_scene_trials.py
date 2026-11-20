"""Stage 0b of the scene sweep: every trial's scene, for every set. SIMPLE's venv, CPU only.

Trial i of every set starts from one scene: a base scene of the task's eval pack (robot spawn, room, table) randomized
the way SIMPLE's highest replay level does it (DRManager.load_state_dict dr_level=2: distractors, materials (table,
floor, robot, objects), lighting, and the poses of the faucet and the clutter, all within the task's ranges), seeded
per trial, with see-through table materials (glass, gems, liquids) re-drawn. Each set then changes one thing:

  c0_baseline          nothing
  c1_1_table_y         the table, with its clutter, slides sideways (y); same distance to the robot, same height; the
                       faucet base stays MARGIN inside the table's ends
  c1_2_table_width     the table gets narrower or wider (y) about its centre line; the faucet base stays MARGIN inside;
                       clutter the narrower table no longer holds is removed
  c1_3_room_shift      the room moves: robot, table and faucet go together to another spot of the floor
  c1_4_room_rotation   the room turns about the table centre: robot, table and faucet turn together
  c2_1_table_x         the table, with its clutter, comes 0 to X_NEAR closer to the robot (physically)
  c2_2_table_x_visual  the table, with its clutter, is drawn 0 to X_FAR closer; MuJoCo keeps the recorded table
                       (scene_sweep.physics_table), so the part in front of the recorded edge is not there for the robot
  c3_1_width_left      the table grows by EXT_MIN..EXT_MAX on the robot's left; its texture keeps the recorded table's
                       scale (scene_sweep.texture_table: Isaac maps the texture as on the recorded table)
  c3_2_width_right     the same on the robot's right
  c3_3_width_both      the same on both sides; trial i adds the same amount in all three c3 sets
  c4_1_faucet_x        the faucet moves 0 to STOP_MAX farther from the robot, deeper on the table; the table and the
                       robot stay (clutter the faucet would hit is removed)
  c4_2_edge_x_visual   the table's near edge is drawn 0 to STOP_MAX closer to the robot (the table grows toward it; its
                       far side, the clutter and the faucet stay); MuJoCo keeps the recorded table, and its texture
                       keeps the recorded scale; trial i uses the same amount in both c4 sets. c4_1 redraws its
                       (unchanged) table the same way, so both sets go through the same rendering path

The full preset (does the policy act on the robot-faucet relation alone?). In the t sets the robot collides only with
the recorded table (scene_sweep.physics_table) and the drawn one keeps the recorded texture scale; their _phys twins
build the same table in MuJoCo too. The near-edge point in front of the faucet stays where it was in t2-t5:

  c0_redraw            nothing; the table redrawn as it is (the t sets' rendering path)
  t1_edge_x            the near edge drawn EDGE_X farther to EDGE_X closer (the far side stays; past the faucet base
                       the faucet floats in front of the table)
  t2_edge_y            one side edge (left on even, right on odd trials) drawn in to the faucet base (plus MARGIN) or
                       out by up to SIDE_OUT; half the trials each way
  t3_scale_xy          depth and width drawn SCALE[0]-SCALE[1] times the recorded ones (log-spread, independently),
                       about the near-edge point in front of the faucet
  t4_rotate            the table drawn turned TURN[0]-TURN[1] degrees (half each way) about the near-edge point in
                       front of the faucet, less where the turned table would come within CLEAR of the robot's body
                       (ROBOT_BODY around its spawn) or of the room; t4_rotate_phys the same turn in MuJoCo too
  t5_shape             another shape (SHAPES, one per trial in turn) with its near edge in front of the faucet where the
                       recorded one was; t5_shape_phys the same shape in MuJoCo too
  r1_room_shift        as c1_3_room_shift
  r2_room_rotation     as c1_4_room_rotation
  r3_room_other        another HSSD room (--room-maps, cycled), placed at the middle of its configured placement range,
                       moved and turned a little where its furniture would come within AROUND of the table (at any
                       height) or in the robot's way
  f1_faucet_x          the faucet moves 0 to STOP_MAX farther from the robot (as c4_1, without the redraw)
  f2_faucet_y          the faucet moves up to FAUCET_Y to the robot's left or right along the edge
  f3_faucet_closer     the faucet moves 0 to STOP_MAX closer to the robot: off the table's near edge, it floats in
                       front of the table (it is fixed to the world); with f1_faucet_x the faucet-only twin of t1.
                       Nothing touches it (scene_sweep.untouchable_faucet): the robot passes through it, as through
                       t1's drawn edge, so it cannot stop the walk (at 15-20 cm it sits between the fingers at the
                       start); its handle cannot be turned, so f3 has no successes
Clutter the changed table no longer holds (or the moved faucet would hit) is removed. With --clutter none (the full
preset) every set starts without the table's clutter objects, so no set differs from its baseline in them, except

  c0_with_clutter      nothing removed: the scene with its clutter, as in the training scenes (the reference for what
                       taking the clutter away does)

With --counts a set runs fewer trials than --trials (its first ones).

So the base randomization of trial i is the same in every set. Each set's values are spread over its range across
the trials (Latin hypercube). Every table, and the robot with the floor it walks on, stays clear of the room's
furniture and walls (room_map.py). Writes into --out:

  scenes/<set>/<i>.json   the scene (a SIMPLE environment_config; its "scene_sweep" entry says what was changed)
  scenes/<set>.png        top-down: the room and every trial's table and robot in it
  plan.json               the trials eval_scene_sweep.py runs, ordered trial index by trial index (all sets of
                          index 0, then of index 1, ...) so a partial run covers every set alike
  n_trials.txt            how many trials

    python make_scene_trials.py --out OUT --simple-dir SIMPLE --base-scenes DIR --room-map NPZ --trials 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

SETS = {
    "c0_baseline": "no change",
    "c1_1_table_y": "table moved sideways (y)",
    "c1_2_table_width": "table width (y) changed",
    "c1_3_room_shift": "robot, table and faucet moved across the room",
    "c1_4_room_rotation": "robot, table and faucet turned in the room",
    "c2_1_table_x": "table, with its clutter, moved 0-5 cm toward the robot",
    "c2_2_table_x_visual": "table, with its clutter, drawn 0-25 cm closer; the robot collides with the recorded table",
    "c3_1_width_left": "table widened on the robot's left",
    "c3_2_width_right": "table widened on the robot's right",
    "c3_3_width_both": "table widened on both sides",
    "c4_1_faucet_x": "faucet moved 0-20 cm farther from the robot, deeper on the table; table and robot unchanged",
    "c4_2_edge_x_visual": "table's near edge drawn 0-20 cm closer to the robot; the robot collides with the recorded "
                          "table; faucet unchanged",
    "c0_redraw": "no change; the table redrawn as it is (the drawn-table sets' rendering path)",
    "c0_with_clutter": "no change, and the table's clutter kept as in the training scenes (every other set has none)",
    "t1_edge_x": "table's near edge drawn 20 cm farther to 20 cm closer; the robot collides with the recorded table",
    "t2_edge_y": "one side edge of the table drawn in to the faucet base or up to 100 cm out; physics unchanged",
    "t3_scale_xy": "table drawn 0.4-2x as deep and as wide, its near edge in front of the faucet kept; physics unchanged",
    "t4_rotate": "table drawn turned 5-45 deg about its near-edge point in front of the faucet; physics unchanged",
    "t4_rotate_phys": "table turned 5-45 deg about its near-edge point in front of the faucet, also in the physics",
    "t5_shape": "table drawn round, half-round, triangular, trapezoid, hexagonal or as a pedestal; physics unchanged",
    "t5_shape_phys": "table round, half-round, triangular, trapezoid, hexagonal or a pedestal, also in the physics",
    "r1_room_shift": "robot, table and faucet moved across the room",
    "r2_room_rotation": "robot, table and faucet turned in the room",
    "r3_room_other": "another room; robot, table and faucet as recorded",
    "f1_faucet_x": "faucet moved 0-20 cm farther from the robot; table and robot unchanged",
    "f2_faucet_y": "faucet moved up to 25 cm left or right along the table edge; table and robot unchanged",
    "f3_faucet_closer": "faucet moved 0-20 cm closer to the robot, in front of the table edge (floating, untouchable: "
                        "the robot passes through it); table and robot unchanged",
}
WIDTH_SETS = {"c3_1_width_left": "left", "c3_2_width_right": "right", "c3_3_width_both": "both"}
MARGIN = 0.02             # m, faucet base corners to the table edge
OBJECT_MARGIN = 0.02      # m, clutter footprint to the table edge
MIN_SHIFT = 0.10          # m, smallest table shift of c1_1
NARROW, WIDE = 0.80, 1.25  # m, c1_2 widths are below NARROW or above WIDE (the recorded table is 1.0 m wide)
MIN_DIST, MAX_DIST = 0.30, 3.0  # m, c1_3 distances (up to where the floor ends)
MIN_TURN = 20.0           # deg, smallest c1_4 rotation
X_NEAR, X_FAR = 0.05, 0.25  # m, c2_1 moves the table 0..X_NEAR toward the robot, c2_2 draws it 0..X_FAR closer
EXT_MIN, EXT_MAX = 0.05, 1.5  # m, c3 adds this much per side (log-spread); the room allows 1.52 m on either side
STOP_MAX = 0.20           # m, c4_1 moves the faucet 0..STOP_MAX away from the robot, c4_2 draws the edge 0..STOP_MAX closer
EDGE_X = 0.20             # m, t1 draws the near edge up to this much closer (+) or farther (-)
SIDE_OUT = 1.0            # m, t2 draws a side edge up to this much out (in: to the faucet base)
SCALE = (0.4, 2.0)        # t3 depth and width factors
TURN = (5.0, 45.0)        # deg, t4 turns
ROBOT_BODY = 0.25         # m, around the robot's spawn, kept CLEAR of a turned table (its hands are above the top)
SHAPES = ("round", "half_round", "triangle", "trapezoid", "hexagon", "pedestal")  # t5, one per trial in turn
SIDE_MARGIN = 0.08        # m, faucet base corners to a shape's side edges
FAUCET_Y = 0.25           # m, f2 moves the faucet up to this far sideways
CLEAR = 0.03              # m, kept free around the table
AROUND = 0.25             # m, r3_room_other: nothing of the other room within this of the table, at any height
ROBOT_RADIUS = 0.32       # m, the robot's footprint
WALK = 0.35               # m, how far forward of its spawn the robot's footprint is kept free (it steps to the table)
STEP = 0.02               # m, footprint sampling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--task", required=True, help="SIMPLE env id without 'simple/', e.g. G1WholebodyOpenFaucetTeleop-v0")
    p.add_argument("--simple-dir", required=True)
    p.add_argument("--base-scenes", required=True, help="the eval pack the scenes start from (dr-level-2)")
    p.add_argument("--room-map", required=True, help="room_map.py's .npz for the base scenes' room and room pose")
    p.add_argument("--trials", type=int, default=50, help="trials per set (at most, with --counts)")
    p.add_argument("--counts", default="", help="set=n,... : these sets run their first n trials only")
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--room-maps", default="", help="room_map.py --all-rooms's folder (index.json), for r3_room_other")
    p.add_argument("--clutter", choices=("keep", "none"), default="keep",
                   help="none: every set but c0_with_clutter starts without the table's clutter objects")
    p.add_argument("--strict-materials", action="store_true",
                   help="also re-draw table materials whose file name says they are clear (e.g. Plastic/*_Clear.mdl)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settings", default="", help="JSON of settings to record in plan.json (checkpoint, ...)")
    return p.parse_args()


def rot2(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def yaw_deg(q) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def lhs(rng: np.random.Generator, n: int) -> np.ndarray:
    """n fractions in [0, 1), one in each of n equal bins, in random order."""
    return (rng.permutation(n) + rng.uniform(size=n)) / n


def pick(f: float, segments: list[tuple[float, float]]) -> float:
    """The point a fraction f of the way along the (non-empty) segments laid end to end."""
    segments = [(a, b) for a, b in segments if b > a]
    t = f * sum(b - a for a, b in segments)
    for a, b in segments:
        if t <= b - a:
            return a + t
        t -= b - a
    return segments[-1][1]


class Room:
    """The room map, and what in a scene must stay clear of it."""

    def __init__(self, path: str) -> None:
        z = np.load(path)
        self.lo, self.res, self.table, self.robot = z["lo"], float(z["res"]), z["table"], z["robot"]
        self.offset0, self.yaw0 = np.asarray(z["offset"], dtype=float), float(z["yaw"])

    def hits(self, grid: np.ndarray, pts: np.ndarray) -> bool:
        ij = np.floor((pts - self.lo) / self.res).astype(int)
        inside = (ij >= 0).all(1) & (ij[:, 0] < grid.shape[0]) & (ij[:, 1] < grid.shape[1])
        return bool((~inside).any() or grid[ij[inside, 0], ij[inside, 1]].any())

    def footprint(self, conf: dict, clear: float = CLEAR) -> tuple[np.ndarray, np.ndarray]:
        """World xy points of the table (plus `clear`) and of the robot from its spawn to WALK ahead of it."""
        table = conf["dr_state_dict"]["scene"]["table"]
        if table.get("uid", "box") == "box":
            sx, sy = table["size"][0] / 2 + clear, table["size"][1] / 2 + clear
            gx, gy = np.meshgrid(np.arange(-sx, sx + 1e-9, STEP), np.arange(-sy, sy + 1e-9, STEP))
            top = np.c_[gx.ravel(), gy.ravel()] @ rot2(yaw_deg(table["pose"]["quaternion"])).T + table["pose"]["position"][:2]
        else:  # round or polygonal: the grid points within CLEAR of its outline
            poly = table_outline(table)
            lo, hi = poly.min(0) - clear, poly.max(0) + clear
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1e-9, STEP), np.arange(lo[1], hi[1] + 1e-9, STEP))
            top = np.c_[gx.ravel(), gy.ravel()]
            top = top[inside(top, poly) >= -clear]
        robot = conf["dr_state_dict"]["spatial"][conf["robot_cfg"]["uid"]]
        heading = rot2(yaw_deg(robot["quaternion"]))
        disc = np.array([[r * math.cos(a), r * math.sin(a)] for r in np.linspace(0, ROBOT_RADIUS, 5)
                         for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)])
        walk = np.concatenate([disc + heading @ [t, 0.0] for t in np.arange(0, WALK + 1e-9, 0.05)])
        return top, walk + robot["position"][:2]

    def clear(self, conf: dict, shift=(0.0, 0.0), turn: float = 0.0, pivot=(0.0, 0.0)) -> bool:
        """Is the scene's table, and the robot's floor, free of the room when the scene moves by `shift` and
        turns by `turn` (deg) about `pivot` in the room (see move_room)?"""
        to_room = lambda w: (w - pivot) @ rot2(turn).T + np.asarray(pivot) + shift
        top, robot = self.footprint(conf)
        return not self.hits(self.table, to_room(top)) and not self.hits(self.robot, to_room(robot))

    def bare(self, conf: dict, shift=(0.0, 0.0), turn: float = 0.0, pivot=(0.0, 0.0)) -> bool:
        """clear(), and nothing of the room at any height (ROBOT_BAND) under, on or within AROUND of the table."""
        to_room = lambda w: (w - pivot) @ rot2(turn).T + np.asarray(pivot) + shift
        return self.clear(conf, shift, turn, pivot) and not self.hits(self.robot, to_room(self.footprint(conf, AROUND)[0]))


# ------------------------------------------------------------------------------------------- scene edits
def tables(conf: dict) -> list[dict]:
    """Every copy of the table in a scene (the DR state is what SIMPLE loads; the layout mirrors it)."""
    return [conf["dr_state_dict"]["scene"]["table"], conf["layout"]["actors"]["table"], conf["layout"]["scene"]["table"]]


def clutter(conf: dict) -> list[tuple[str, str]]:
    """(uid, layout name) of the clutter objects, in the order SIMPLE hands out their object materials."""
    return [(str(a["uid"]), name) for name, a in conf["layout"]["actors"].items()
            if name == "target" or name.startswith("distractor_")]


def move_table(conf: dict, dx: float, dy: float) -> None:
    """Move the table, with its clutter, by (dx, dy) in the world."""
    for t in tables(conf):
        t["pose"]["position"][0] += dx
        t["pose"]["position"][1] += dy
    spatial, actors = conf["dr_state_dict"]["spatial"], conf["layout"]["actors"]
    for uid, name in clutter(conf):
        for pos in (spatial[uid]["position"], actors[name]["pose"]["position"]):
            pos[0] += dx
            pos[1] += dy


def widen_table(conf: dict, left: float, right: float, robot_left: np.ndarray) -> None:
    """Grow the table by `left` on the robot's left side and by `right` on its right side; the clutter stays."""
    for t in tables(conf):
        axis = rot2(yaw_deg(t["pose"]["quaternion"])) @ [0.0, 1.0]  # the table's +y side
        plus, minus = (left, right) if axis @ robot_left >= 0 else (right, left)
        t["size"][1] += plus + minus
        t["pose"]["position"][0] += float((plus - minus) / 2 * axis[0])
        t["pose"]["position"][1] += float((plus - minus) / 2 * axis[1])


def move_faucet(conf: dict, key: str, dx: float, dy: float) -> None:
    """Move the faucet (spatial state `key`) by (dx, dy) in the world."""
    for pos in (conf["dr_state_dict"]["spatial"][key]["position"], conf["layout"]["actors"]["articulated"]["pose"]["position"]):
        pos[0] += dx
        pos[1] += dy


def extend_table(conf: dict, near: float, forward: np.ndarray) -> None:
    """Grow the table by `near` on its side facing the robot (heading `forward`); its far side stays."""
    for t in tables(conf):
        along = rot2(yaw_deg(t["pose"]["quaternion"])).T @ forward  # the robot's heading in the table's axes
        k = int(np.argmax(np.abs(along)))
        if abs(along[k]) < 0.999:
            raise SystemExit("the table is not square to the robot; c4_2 needs it to be")
        t["size"][k] += near
        t["pose"]["position"][0] -= float(near / 2 * forward[0])
        t["pose"]["position"][1] -= float(near / 2 * forward[1])


def reach_of(mjcf_path: str) -> float:
    """How far (m, horizontally) an articulated object reaches around its origin: its farthest mesh vertex."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(mjcf_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    reach = 0.0
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mesh]: m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            p = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            reach = max(reach, float(np.linalg.norm(p[:, :2], axis=1).max()))
    return reach


def table_box(conf: dict) -> dict:
    """The table's pose and size, as the scene has it (what scene_sweep.physics_table / texture_table hold)."""
    t = conf["dr_state_dict"]["scene"]["table"]
    return dict(pose=deepcopy(t["pose"]), size=list(t["size"]))


def remove_clutter(conf: dict, drop: set[str]) -> None:
    """Remove these clutter objects; the rest keep their materials, renumbered as SIMPLE numbers them on loading."""
    ds, actors = conf["dr_state_dict"], conf["layout"]["actors"]
    order = clutter(conf)
    params = ds["material"]["object_shader_params"]
    keep = [(name, params[k]) for k, (uid, name) in enumerate(order) if uid not in drop]
    for uid in drop:
        ds["spatial"].pop(uid)
    if ds.get("target") and str(ds["target"]["uid"]) in drop:
        ds["target"] = None
    ds["distractors"] = {uid: v for uid, v in ds["distractors"].items() if uid not in drop}
    old = {name: actors.pop(name) for _, name in order}
    articulated = actors.pop("articulated")
    n = 0
    for name, _ in keep:
        if name == "target":
            actors["target"] = old[name]
        else:
            actors[f"distractor_{n}"] = old[name]
            n += 1
    actors["articulated"] = articulated
    ds["material"]["object_shader_params"] = [p for _, p in keep]


def move_room(conf: dict, shift=(0.0, 0.0), turn: float = 0.0, pivot=(0.0, 0.0)) -> None:
    """Move the robot, table and faucet by `shift` and turn them by `turn` (deg) about `pivot`, in the room. The
    simulation keeps them and moves the room the opposite way: SIMPLE puts the room's own table centre at
    center_offset and turns the room by center_orientation[2] about it, so the room point q is seen at
    Rz(-turn) (q - pivot - shift) + pivot."""
    for scene in (conf["dr_state_dict"]["scene"], conf["layout"]["scene"]):
        off = np.asarray(scene["center_offset"][:2], dtype=float)
        new = rot2(-turn) @ (off - np.asarray(pivot) - np.asarray(shift)) + np.asarray(pivot)
        scene["center_offset"] = [float(new[0]), float(new[1]), *scene["center_offset"][2:]]
        ori = scene["center_orientation"]
        scene["center_orientation"] = [ori[0], ori[1], float(ori[2] - turn)]


def yaw_quat(deg: float) -> list[float]:
    """wxyz quaternion of a turn about z."""
    return [math.cos(math.radians(deg) / 2), 0.0, 0.0, math.sin(math.radians(deg) / 2)]


def table_outline(t: dict, arc: int = 48) -> np.ndarray:
    """World xy outline of a table (box, cylinder or prism), counterclockwise."""
    if t.get("uid") == "cylinder":
        a = np.linspace(0, 2 * math.pi, arc, endpoint=False)
        local = 0.5 * t["size"][0] * np.c_[np.cos(a), np.sin(a)]
    elif t.get("uid") == "prism":
        local = np.asarray(t["vertices"], dtype=float)
    else:
        hx, hy = t["size"][0] / 2, t["size"][1] / 2
        local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return local @ rot2(yaw_deg(t["pose"]["quaternion"])).T + np.asarray(t["pose"]["position"][:2], dtype=float)


def inside(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """How far (m) points lie inside a convex counterclockwise polygon: the smallest distance to an edge line,
    negative outside."""
    edges = np.roll(poly, -1, axis=0) - poly
    inward = np.stack([-edges[:, 1], edges[:, 0]], 1) / np.linalg.norm(edges, axis=1, keepdims=True)
    return ((np.atleast_2d(pts)[:, None, :] - poly[None]) * inward[None]).sum(-1).min(1)


def poly_dist(p, poly: np.ndarray) -> float:
    """Distance (m) from a point to a convex polygon (0 inside)."""
    p = np.asarray(p, dtype=float)
    if inside(p, poly)[0] >= 0:
        return 0.0
    a, b = poly, np.roll(poly, -1, axis=0)
    t = np.clip(((p - a) * (b - a)).sum(1) / ((b - a) ** 2).sum(1), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * (b - a) - p, axis=1).min())


def set_table(conf: dict, new: dict) -> None:
    """Make every copy of the scene's table `new` (uid, size, pose and, for a prism, vertices); materials stay."""
    for t in tables(conf):
        t.pop("vertices", None)
        t.update(deepcopy({k: v for k, v in new.items() if k != "material"}))


def off_top(conf: dict, radius: dict) -> set[str]:
    """The clutter objects whose footprint is not OBJECT_MARGIN inside the scene's (drawn) table."""
    poly = table_outline(conf["dr_state_dict"]["scene"]["table"])
    sp = conf["dr_state_dict"]["spatial"]
    return {uid for uid, _ in clutter(conf)
            if inside(np.asarray(sp[uid]["position"][:2]), poly)[0] < radius[uid] + OBJECT_MARGIN}


def removed(drop: set[str]) -> dict:
    return dict(clutter_removed=sorted(drop)) if drop else {}


def turn_table(t: dict, deg: float, pivot) -> dict:
    """The table turned by `deg` about the world point `pivot`."""
    out = deepcopy(t)
    c = rot2(deg) @ (np.asarray(t["pose"]["position"][:2], dtype=float) - pivot) + pivot
    out["pose"]["position"] = [float(c[0]), float(c[1]), *t["pose"]["position"][2:]]
    out["pose"]["quaternion"] = yaw_quat(yaw_deg(t["pose"]["quaternion"]) + deg)
    return out


def shape_table(kind: str, f: list[float], t0: dict, u_edge: float, cu: np.ndarray, cv: np.ndarray,
                forward: np.ndarray, left: np.ndarray) -> tuple[dict, dict]:
    """A t5 table: `kind` sized and placed by the fractions f (4 of them), its near edge in front of the faucet at
    u_edge along the robot's heading, the faucet base (corners at cu forward, cv left, m) on it. Returns the table
    config (square to the robot) and what was chosen."""
    vf = float(cv.mean())
    if kind in ("round", "pedestal"):
        if kind == "round":
            r = 0.5 + 0.4 * f[0]
            off = (2 * f[1] - 1) * min(0.25, r - 0.15)  # the centre this far to the left of the faucet
            centre = (u_edge + math.sqrt(r * r - off * off), vf + off)
        else:  # just under the faucet base
            cu0 = float(cu.mean())
            r = max(0.08 + 0.04 * f[0], float(np.hypot(cu - cu0, cv - vf).max()) + 0.005)
            centre = (cu0, vf)
        uid, size, verts, dims = "cylinder", [2 * r, 2 * r], None, dict(radius=round(r, 4))
    else:
        if kind == "half_round":  # flat side toward the robot
            r = 0.5 + 0.4 * f[0]
            a = np.linspace(-math.pi / 2, math.pi / 2, 25)
            verts = np.c_[r * np.cos(a) - r / 2, r * np.sin(a)]
            dims = dict(radius=round(r, 4))
        elif kind == "triangle":  # base on the near edge, apex away from the robot
            base, h = 0.9 + 0.9 * f[0], 0.8 + 0.7 * f[2]
            verts = np.array([[-h / 2, -base / 2], [h / 2, 0.0], [-h / 2, base / 2]])
            dims = dict(base=round(base, 4), height=round(h, 4))
        elif kind == "trapezoid":  # the long side on the near edge
            near, far, depth = 1.0 + 0.8 * f[0], 0.4 + 0.6 * f[2], 0.6 + 0.9 * f[3]
            verts = np.array([[-depth / 2, -near / 2], [depth / 2, -far / 2], [depth / 2, far / 2], [-depth / 2, near / 2]])
            dims = dict(near=round(near, 4), far=round(far, 4), depth=round(depth, 4))
        else:  # hexagon, a flat side toward the robot
            R = 0.5 + 0.4 * f[0]
            a = np.radians(30 + 60 * np.arange(6))
            verts = np.c_[R * np.cos(a), R * np.sin(a)]
            dims = dict(radius=round(R, 4))
        x0, n = verts[:, 0].min(), len(verts)
        near = [k for k in range(n) if abs(verts[k, 0] - x0) < 1e-9 and abs(verts[(k + 1) % n, 0] - x0) < 1e-9]
        edges = np.roll(verts, -1, axis=0) - verts
        inward = np.stack([-edges[:, 1], edges[:, 0]], 1) / np.linalg.norm(edges, axis=1, keepdims=True)
        others = [k for k in range(n) if k not in near]

        def fits(c: float) -> bool:
            """The faucet base SIDE_MARGIN inside every edge but the near one (it overhangs that as recorded)."""
            pts = np.c_[np.maximum(cu - u_edge + x0, x0), cv - c]
            return bool(((pts[:, None, :] - verts[None]) * inward[None]).sum(-1)[:, others].min() >= SIDE_MARGIN)
        ok = [c for c in np.linspace(vf - 1.0, vf + 1.0, 401) if fits(c)]  # where along the edge it fits; f[1] picks
        if not ok:
            raise SystemExit(f"no place for a {kind} table under the faucet ({dims})")
        cvc = ok[min(len(ok) - 1, int(f[1] * len(ok)))]
        lo, hi = verts.min(0), verts.max(0)
        mid = (lo + hi) / 2
        verts = verts - mid
        centre = (u_edge - x0 + mid[0], cvc + mid[1])
        uid, size = "prism", (hi - lo).tolist()
    xy = centre[0] * forward + centre[1] * left
    heading = math.degrees(math.atan2(forward[1], forward[0]))
    out = dict(uid=uid, size=[float(size[0]), float(size[1]), t0["size"][2]],
               pose=dict(position=[float(xy[0]), float(xy[1]), t0["pose"]["position"][2]], quaternion=yaw_quat(heading)))
    if verts is not None:
        out["vertices"] = [[round(float(x), 5), round(float(y), 5)] for x, y in verts]
    return out, dict(shape=kind, shape_i=SHAPES.index(kind), **dims)


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    out = Path(args.out)
    simple = Path(args.simple_dir)
    sys.path.insert(0, str(simple / "scripts"))
    import tyro
    import yaml
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from make_round_table_dataset import base_footprint, object_radius

    import simple.tasks  # noqa: F401  registers the tasks
    from simple.tasks.registry import TaskRegistry
    from simple.utils import NumpyArrayEncoder, resolve_data_path, resolve_res_path

    sets = [s for s in args.sets.split(",") if s]
    unknown = set(sets) - set(SETS)
    if unknown:
        raise SystemExit(f"unknown sets {sorted(unknown)}; known: {', '.join(SETS)}")
    base = [json.loads(json.loads(line)["environment_config"])
            for line in (Path(args.base_scenes) / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    room = Room(args.room_map)
    for b in base:
        sc = b["dr_state_dict"]["scene"]
        if not (np.allclose(sc["center_offset"][:2], room.offset0) and math.isclose(sc["center_orientation"][2], room.yaw0)):
            raise SystemExit(f"a base scene puts the room at {sc['center_offset']}, {sc['center_orientation']}; "
                             f"the room map is for {room.offset0}, {room.yaw0}")

    sonic = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[]).load_wbc_yaml()
    sonic["ENV_NAME"] = "simple"
    task = TaskRegistry.make(base[0]["uid"], render_hz=50, sonic_config=sonic)
    split = task.metadata.get("split", "train")
    materials = np.load(resolve_res_path("vMaterials_2/material_split.npy"), allow_pickle=True).item()[split]["table"]
    see_through = lambda m: (bool({"Glass", "Gems", "Liquids"} & set(Path(m["path"]).parts))
                             or bool(re.search(r"clear|translucent|transparent", m["name"], re.IGNORECASE))
                             or (args.strict_materials and bool(re.search(r"clear|translucent|transparent|glass",
                                                                           Path(m["path"]).stem, re.IGNORECASE))))
    opaque = [m for m in materials if not see_through(m)]
    art = base[0]["dr_state_dict"]["articulated"]["uid"]
    target_key = f"articulated_{art}"
    mjcf = resolve_data_path(f"assets/articulated/{int(art):03d}/output_mjcf/{int(art):03d}.xml")
    footprint, faucet_reach = base_footprint(mjcf), reach_of(mjcf)
    train = json.loads((out / "train_targets.json").read_text()) if (out / "train_targets.json").exists() else None

    def faucet_pose(conf: dict) -> tuple[dict, bool | None]:
        """The faucet in the robot's start frame (dx forward, dy left, dyaw), and whether that is in the training
        range (None without train_targets.json)."""
        sp = conf["dr_state_dict"]["spatial"]
        robot, faucet = sp[conf["robot_cfg"]["uid"]], sp[target_key]
        rel = rot2(-yaw_deg(robot["quaternion"])) @ (np.asarray(faucet["position"][:2]) - robot["position"][:2])
        pose = dict(dx=round(float(rel[0]), 4), dy=round(float(rel[1]), 4),
                    dyaw=round(yaw_deg(faucet["quaternion"]) - yaw_deg(robot["quaternion"]), 2))
        if train is None:
            return pose, None
        box = train["box"]
        return pose, bool(box["dx"][0] <= rel[0] <= box["dx"][1] and box["dy"][0] <= rel[1] <= box["dy"][1])

    rng = np.random.default_rng(args.seed)
    base_order = rng.permutation(len(base))
    seeds = [int(s) for s in np.random.SeedSequence(args.seed).generate_state(args.trials)]
    counts = {k: int(v) for k, v in (c.split("=") for c in args.counts.split(",") if c)}  # (sets not run: ignored)
    n_set = {s: min(counts.get(s, args.trials), args.trials) for s in sets}  # trial indices 0..n-1 of each set
    frac = {s: [lhs(np.random.default_rng([args.seed, k, j]), n_set.get(s, args.trials)) for j in range(4)]
            for k, s in enumerate(SETS)}
    width_frac = lhs(np.random.default_rng([args.seed, 77]), args.trials)  # shared by the c3 sets
    stop_frac = lhs(np.random.default_rng([args.seed, 88]), args.trials)  # shared by the c4 sets
    twin = lambda *ss: max([n_set[x] for x in ss if x in n_set] or [1])
    turn_frac = [lhs(np.random.default_rng([args.seed, 66, j]), twin("t4_rotate", "t4_rotate_phys")) for j in range(2)]
    shape_frac = [lhs(np.random.default_rng([args.seed, 55, j]), twin("t5_shape", "t5_shape_phys")) for j in range(4)]
    others = {}  # r3_room_other: uid -> (its room map, its place)
    if "r3_room_other" in sets:
        if not args.room_maps:
            raise SystemExit("r3_room_other needs --room-maps")
        index = json.loads((Path(args.room_maps) / "index.json").read_text())
        others = {uid: (Room(str(Path(args.room_maps) / r["npz"])), r) for uid, r in index.items()}
        if not others:
            raise SystemExit(f"no other room in {args.room_maps}")
    room_order = [sorted(others)[k] for k in np.random.default_rng([args.seed, 44]).permutation(len(others))]
    r3_next = [0]  # where in room_order the next r3 trial starts looking
    hssd = {r["uid"]: r for r in yaml.safe_load(open(resolve_res_path("hssd-scenes/config.yaml")))}

    def scene0(i: int) -> dict:
        """Trial i's randomized base scene (what c0_baseline runs)."""
        random.seed(seeds[i])
        np.random.seed(seeds[i] % 2**32)
        task.reset(options={"state_dict": base[base_order[i % len(base)]], "dr_level": 2})
        conf = json.loads(json.dumps(task.state_dict(), cls=NumpyArrayEncoder))
        ds = conf["dr_state_dict"]
        if see_through(ds["material"]["table_material"]):
            m = opaque[np.random.default_rng([args.seed, 99, i]).integers(len(opaque))]
            ds["material"]["table_material"] = m
            for t in tables(conf):
                t["material"] = m
        return conf

    (out / "scenes").mkdir(parents=True, exist_ok=True)
    trials, placed = [], {s: [] for s in sets}
    for i in range(max(n_set.values())):
        c0 = scene0(i)
        with_clutter = deepcopy(c0)
        if args.clutter == "none":
            remove_clutter(c0, {uid for uid, _ in clutter(c0)})
        ds = c0["dr_state_dict"]
        robot, faucet = ds["spatial"][c0["robot_cfg"]["uid"]], ds["spatial"][target_key]
        table = ds["scene"]["table"]
        cy, half = table["pose"]["position"][1], table["size"][1] / 2
        corners = footprint @ rot2(yaw_deg(faucet["quaternion"])).T + faucet["position"][:2]
        faucet_rel, _ = faucet_pose(c0)
        if not room.clear(c0):
            print(f"[trials] WARNING trial {i}: the base scene itself touches the room's furniture", flush=True)
        pivot = tuple(table["pose"]["position"][:2])
        forward = rot2(yaw_deg(robot["quaternion"])) @ [1.0, 0.0]
        left_of_robot = rot2(yaw_deg(robot["quaternion"])) @ [0.0, 1.0]
        radius = {uid: object_radius(uid, with_clutter["dr_state_dict"]["spatial"][uid]["quaternion"])
                  for uid, _ in clutter(with_clutter)}
        objects = [dict(uid=uid, r=round(r, 4)) for uid, r in radius.items()]
        e = math.exp(math.log(EXT_MIN) + width_frac[i] * (math.log(EXT_MAX) - math.log(EXT_MIN)))
        while True:  # the c3 sets add the same e: less if the room is in the way of both sides (not, to 1.52 m, here)
            wide = deepcopy(c0)
            widen_table(wide, e, e, left_of_robot)
            if room.clear(wide) or e <= EXT_MIN:
                break
            e = max(EXT_MIN, e - STEP)

        # the full preset's frame: along the robot's heading (u) and to its left (v); the recorded table is square to it
        uv = lambda xy: np.c_[np.asarray(xy) @ forward, np.asarray(xy) @ left_of_robot]
        top0 = uv(table_outline(table))
        u_edge, cu, cv = float(top0[:, 0].min()), uv(corners)[:, 0], uv(corners)[:, 1]
        v_f = float(np.asarray(faucet["position"][:2]) @ left_of_robot)
        edge_pivot = u_edge * forward + v_f * left_of_robot  # the near-edge point in front of the faucet

        for s in sets:
            if i >= n_set[s]:
                continue
            conf, value, extra = deepcopy(c0), {}, {}
            f = [fr[i] for fr in frac[s]]
            if s == "c1_1_table_y":
                lo, hi = corners[:, 1].max() + MARGIN - half - cy, corners[:, 1].min() - MARGIN + half - cy
                dy = pick(f[0], [(lo, -MIN_SHIFT), (MIN_SHIFT, hi)])
                moved = deepcopy(conf)
                move_table(moved, 0.0, dy)
                while not room.clear(moved) and abs(dy) > MIN_SHIFT:  # back towards the recorded spot until clear
                    dy -= math.copysign(STEP, dy)
                    moved = deepcopy(conf)
                    move_table(moved, 0.0, dy)
                conf = moved
                value = dict(table_dy=round(dy, 4))
            elif s == "c1_2_table_width":
                narrowest = 2 * (max(corners[:, 1].max() - cy, cy - corners[:, 1].min()) + MARGIN)
                widest = table["size"][1]
                while widest < 6 and room.clear({**conf, "dr_state_dict": {**ds, "scene": {
                        **ds["scene"], "table": {**table, "size": [table["size"][0], widest + STEP, table["size"][2]]}}}}):
                    widest += STEP
                width = math.exp(pick(f[0], [(math.log(narrowest), math.log(NARROW)), (math.log(WIDE), math.log(widest))]))
                for t in tables(conf):
                    t["size"][1] = width
                drop = {uid for uid, _ in clutter(conf)
                        if abs(ds["spatial"][uid]["position"][1] - cy) + radius[uid]
                        > width / 2 - OBJECT_MARGIN}
                remove_clutter(conf, drop)
                value = dict(table_width=round(width, 4), clutter_removed=sorted(drop))
            elif s in ("c1_3_room_shift", "r1_room_shift"):
                for k in range(36):  # a direction the floor allows at least MIN_DIST along
                    heading = (360 * f[0] + 137.5 * k) % 360
                    u = rot2(heading) @ [1.0, 0.0]
                    reach = 0.0
                    for d in np.arange(STEP, MAX_DIST + 1e-9, STEP):
                        if not room.clear(conf, shift=tuple(d * u)):
                            break
                        reach = d
                    if reach >= MIN_DIST:
                        break
                else:
                    raise SystemExit(f"trial {i}: no free spot {MIN_DIST} m away in any direction")
                dist = MIN_DIST + f[1] * (reach - MIN_DIST)
                move_room(conf, shift=tuple(dist * u))
                value = dict(room_shift=[round(float(dist * u[0]), 4), round(float(dist * u[1]), 4)],
                             distance=round(float(dist), 4), direction_deg=round(heading, 1))
            elif s in ("c1_4_room_rotation", "r2_room_rotation"):
                sign = 1.0 if f[1] < 0.5 else -1.0
                want = MIN_TURN + f[0] * (180.0 - MIN_TURN)
                for turn in sorted(np.arange(MIN_TURN, 180.0 + 1e-9, 1.0), key=lambda a: abs(a - want)):
                    if room.clear(conf, turn=sign * turn, pivot=pivot):
                        break
                else:
                    raise SystemExit(f"trial {i}: no rotation keeps the scene clear of the room")
                move_room(conf, turn=float(sign * turn), pivot=pivot)
                value = dict(room_rotation_deg=round(float(sign * turn), 1))
            elif s in ("c2_1_table_x", "c2_2_table_x_visual"):
                lo, hi = 0.0, X_NEAR if s == "c2_1_table_x" else X_FAR
                d = lo + f[0] * (hi - lo)
                moved = deepcopy(conf)
                move_table(moved, *(-d * forward))
                while not room.clear(moved) and d > lo:  # back towards the recorded spot until clear
                    d = max(lo, d - STEP)
                    moved = deepcopy(conf)
                    move_table(moved, *(-d * forward))
                if s == "c2_2_table_x_visual":
                    extra["physics_table"] = table_box(conf)
                conf = moved
                value = dict(table_closer=round(d, 4))
            elif s in WIDTH_SETS:
                side = WIDTH_SETS[s]
                wide = deepcopy(conf)
                widen_table(wide, e if side != "right" else 0.0, e if side != "left" else 0.0, left_of_robot)
                if not room.clear(wide):
                    print(f"[trials] WARNING trial {i}: {s} touches the room's furniture", flush=True)
                extra["texture_table"] = table_box(conf)
                conf = wide
                value = dict(extend=round(e, 4), side=side,
                             table_width=round(conf["dr_state_dict"]["scene"]["table"]["size"][1], 4))
            elif s == "c4_1_faucet_x":
                o = STOP_MAX * stop_frac[i]
                move_faucet(conf, target_key, *(o * forward))
                at = np.asarray(conf["dr_state_dict"]["spatial"][target_key]["position"][:2])
                drop = {uid for uid, _ in clutter(conf)
                        if np.linalg.norm(np.asarray(ds["spatial"][uid]["position"][:2]) - at)
                        < faucet_reach + radius[uid] + OBJECT_MARGIN}
                remove_clutter(conf, drop)
                extra["physics_table"] = extra["texture_table"] = table_box(conf)  # redrawn as it is, as in c4_2
                value = dict(faucet_farther=round(o, 4), **(dict(clutter_removed=sorted(drop)) if drop else {}))
            elif s == "c4_2_edge_x_visual":
                o = STOP_MAX * stop_frac[i]
                extra["physics_table"] = extra["texture_table"] = table_box(conf)
                extend_table(conf, o, forward)
                if not room.clear(conf):
                    print(f"[trials] WARNING trial {i}: {s} touches the room's furniture", flush=True)
                value = dict(edge_closer=round(o, 4))
            elif s == "c0_redraw":
                extra["physics_table"] = extra["texture_table"] = table_box(conf)
            elif s == "c0_with_clutter":
                conf = deepcopy(with_clutter)
            elif s == "t1_edge_x":
                d = EDGE_X * (2 * f[0] - 1)
                extra["physics_table"] = extra["texture_table"] = table_box(conf)
                extend_table(conf, d, forward)
                if not room.clear(conf):
                    print(f"[trials] WARNING trial {i}: {s} touches the room's furniture", flush=True)
                drop = off_top(conf, radius)
                remove_clutter(conf, drop)
                value = dict(edge_closer=round(d, 4), **removed(drop))
            elif s == "t2_edge_y":
                side = "left" if i % 2 == 0 else "right"
                out_dir = left_of_robot if side == "left" else -left_of_robot
                room_in = float((table_outline(table) @ out_dir).max() - (corners @ out_dir).max() - MARGIN)
                a = -f[0] * room_in if f[1] < 0.5 else f[0] * SIDE_OUT
                while True:  # out: back toward the recorded edge until the room is clear
                    wide = deepcopy(conf)
                    widen_table(wide, a if side == "left" else 0.0, a if side == "right" else 0.0, left_of_robot)
                    if a <= 0 or room.clear(wide):
                        break
                    a = max(0.0, a - STEP)
                extra["physics_table"] = extra["texture_table"] = table_box(conf)
                conf = wide
                drop = off_top(conf, radius)
                remove_clutter(conf, drop)
                value = dict(edge_out=round(a, 4), side=side,
                             table_centre_left=round(a / 2 * (1 if side == "left" else -1), 4), **removed(drop))
            elif s == "t3_scale_xy":
                ls = math.log(SCALE[1] / SCALE[0])
                sx0, sy0 = (math.exp(math.log(SCALE[0]) + fr * ls) for fr in f[:2])
                v_r, v_l = float(top0[:, 1].min()), float(top0[:, 1].max())
                depth, width = float(np.ptp(top0[:, 0])), v_l - v_r
                sy_min = max((cv.max() + MARGIN - v_f) / (v_l - v_f), (v_f - cv.min() + MARGIN) / (v_f - v_r))
                k = int(np.argmax(np.abs(rot2(yaw_deg(table["pose"]["quaternion"])).T @ forward)))  # the depth axis
                for tt in np.linspace(1, 0, 21):  # toward the recorded size until the room is clear
                    sx, sy = sx0 ** tt, max(sy0 ** tt, sy_min)
                    scaled = deepcopy(conf)
                    u_c, v_c = u_edge + depth * sx / 2, v_f + ((v_l + v_r) / 2 - v_f) * sy
                    xy = u_c * forward + v_c * left_of_robot
                    for t in tables(scaled):
                        t["size"][k], t["size"][1 - k] = depth * sx, width * sy
                        t["pose"]["position"][:2] = [float(xy[0]), float(xy[1])]
                    if room.clear(scaled):
                        break
                extra["physics_table"] = extra["texture_table"] = table_box(conf)
                conf = scaled
                drop = off_top(conf, radius)
                remove_clutter(conf, drop)
                value = dict(depth_scale=round(sx, 4), width_scale=round(sy, 4), depth=round(depth * sx, 4),
                             width=round(width * sy, 4), **removed(drop))
            elif s in ("t4_rotate", "t4_rotate_phys"):
                sign = 1.0 if turn_frac[1][i] < 0.5 else -1.0
                r0 = np.asarray(robot["position"][:2], dtype=float)

                def clear_turn(deg: float) -> bool:
                    t = turn_table(table, deg, edge_pivot)
                    probe = deepcopy(conf)
                    set_table(probe, t)
                    return poly_dist(r0, table_outline(t)) >= ROBOT_BODY + CLEAR and room.clear(probe)
                cap = next((a for a in np.arange(TURN[1], TURN[0] - 1e-9, -1.0) if clear_turn(sign * a)), TURN[0])
                deg = sign * (TURN[0] + turn_frac[0][i] * (cap - TURN[0]))
                turned = turn_table(table, deg, edge_pivot)
                if s == "t4_rotate":
                    extra["physics_table"] = table_box(conf)
                extra["texture_table"] = dict(pose=deepcopy(turned["pose"]), size=list(turned["size"]))
                set_table(conf, turned)
                drop = off_top(conf, radius)
                remove_clutter(conf, drop)
                value = dict(table_turn_deg=round(float(deg), 2), max_turn_deg=round(float(cap), 1), **removed(drop))
            elif s in ("t5_shape", "t5_shape_phys"):
                new, dims = shape_table(SHAPES[i % len(SHAPES)], [fr[i] for fr in shape_frac], table, u_edge, cu, cv,
                                        forward, left_of_robot)
                if s == "t5_shape":
                    extra["physics_table"] = table_box(conf)
                extra["texture_table"] = table_box(conf)
                set_table(conf, new)
                if not room.clear(conf):
                    print(f"[trials] WARNING trial {i}: {s} touches the room's furniture", flush=True)
                drop = off_top(conf, radius)
                remove_clutter(conf, drop)
                value = dict(**dims, **removed(drop))
            elif s == "r3_room_other":
                for k in range(len(room_order)):  # the rooms in turn, skipping those it does not fit in
                    uid = room_order[(r3_next[0] + k) % len(room_order)]
                    rmap, place = others[uid]
                    moved = deepcopy(conf)
                    for sc in (moved["dr_state_dict"]["scene"], moved["layout"]["scene"]):
                        sc.update(uid=f"hssd:{uid}", conf=deepcopy(hssd[uid]), name=hssd[uid]["name"],
                                  data_dir=f"{hssd[uid]['data_dir']}/{hssd[uid]['name']}",
                                  center_offset=[*place["offset"], 0.0], center_orientation=[90.0, 0.0, place["yaw"]])
                    tries = [(np.zeros(2), 0.0)] + sorted(
                        ((d * (rot2(a) @ [1.0, 0.0]), tn) for d in np.arange(0.05, 0.51, 0.05)
                         for a in range(0, 360, 45) for tn in (0.0, -10.0, 10.0, -20.0, 20.0)),
                        key=lambda c: np.linalg.norm(c[0]) + abs(c[1]) / 100)
                    fit = next(((sh, tn) for sh, tn in tries if rmap.bare(moved, shift=tuple(sh), turn=tn, pivot=pivot)),
                               None)
                    if fit is not None:
                        r3_next[0] += k + 1
                        move_room(moved, shift=tuple(fit[0]), turn=fit[1], pivot=pivot)
                        conf = moved
                        value = dict(room=uid, room_i=sorted(others).index(uid),
                                     room_shift=round(float(np.linalg.norm(fit[0])), 3), room_turn_deg=fit[1])
                        break
                else:
                    raise SystemExit(f"trial {i}: no other room fits")
            elif s == "f1_faucet_x":
                o = STOP_MAX * f[0]
                move_faucet(conf, target_key, *(o * forward))
                at = np.asarray(conf["dr_state_dict"]["spatial"][target_key]["position"][:2])
                drop = {uid for uid, _ in clutter(conf)
                        if np.linalg.norm(np.asarray(ds["spatial"][uid]["position"][:2]) - at)
                        < faucet_reach + radius[uid] + OBJECT_MARGIN}
                remove_clutter(conf, drop)
                value = dict(faucet_farther=round(o, 4), **removed(drop))
            elif s == "f2_faucet_y":
                v_r, v_l = float(top0[:, 1].min()), float(top0[:, 1].max())
                o = float(np.clip(FAUCET_Y * (2 * f[0] - 1), v_r + MARGIN - cv.min(), v_l - MARGIN - cv.max()))
                move_faucet(conf, target_key, *(o * left_of_robot))
                at = np.asarray(conf["dr_state_dict"]["spatial"][target_key]["position"][:2])
                drop = {uid for uid, _ in clutter(conf)
                        if np.linalg.norm(np.asarray(ds["spatial"][uid]["position"][:2]) - at)
                        < faucet_reach + radius[uid] + OBJECT_MARGIN}
                remove_clutter(conf, drop)
                value = dict(faucet_left=round(o, 4), **removed(drop))
            elif s == "f3_faucet_closer":  # (nothing of the table's clutter is in front of its edge)
                o = -STOP_MAX * f[0]
                move_faucet(conf, target_key, *(o * forward))
                value, extra["untouchable_faucet"] = dict(faucet_farther=round(o, 4)), True
            conf["scene_sweep"] = dict(set=s, index=i, change=SETS[s], value=value, seed=seeds[i],
                                       base_episode=int(base_order[i % len(base)]), base_scenes=str(args.base_scenes),
                                       recorded_table=table_box(c0), **extra)
            path = out / "scenes" / s / f"{i:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(conf))
            placed[s].append((conf, value))
            pose, in_range = faucet_pose(conf)
            trials.append(dict(trial=len(trials), set=s, index=i, value=value, conf=str(path.relative_to(out)),
                               seed=seeds[i], base_episode=int(base_order[i % len(base)]), in_train_range=in_range,
                               objects=[o for o in objects if o["uid"] in conf["dr_state_dict"]["spatial"]],
                               **pose))
        print(f"[trials] {i + 1}/{max(n_set.values())}: faucet dx {faucet_rel['dx']:+.3f} dy {faucet_rel['dy']:+.3f} "
              f"yaw {faucet_rel['dyaw']:+.1f}; "
              + "; ".join(f"{t['set']} {t['value']}" for t in trials[-len(sets):] if t["value"]), flush=True)

    settings = json.loads(args.settings) if args.settings else {}
    settings.update(task=args.task, target=target_key,
                    kind="articulated", base_scenes=str(Path(args.base_scenes).resolve()), seed=args.seed,
                    trials_per_set=max(n_set.values()), per_set=n_set, sets={s: SETS[s] for s in sets}, clutter=args.clutter,
                    room_map=str(args.room_map), shapes=list(SHAPES), rooms=sorted(others),
                    margins=dict(faucet=MARGIN, clutter=OBJECT_MARGIN, table_clear=CLEAR, robot_radius=ROBOT_RADIUS),
                    ranges=dict(min_table_shift=MIN_SHIFT, narrow_below=NARROW, wide_above=WIDE,
                                shift_m=[MIN_DIST, MAX_DIST], min_turn_deg=MIN_TURN, table_closer_m=dict(c2_1_table_x=[0.0, X_NEAR], c2_2_table_x_visual=[0.0, X_FAR]),
                                extend_m=[EXT_MIN, EXT_MAX], stop_m=[0.0, STOP_MAX], edge_x_m=[-EDGE_X, EDGE_X],
                                side_m=[None, SIDE_OUT], scale=list(SCALE), turn_deg=list(TURN),
                                faucet_y_m=[-FAUCET_Y, FAUCET_Y], robot_body=ROBOT_BODY))
    (out / "plan.json").write_text(json.dumps(dict(settings=settings, trials=trials), indent=1))
    (out / "n_trials.txt").write_text(f"{len(trials)}\n")
    draw(out, room, placed, target_key)
    per = sorted(set(n_set.values()))
    print(f"[trials] {len(trials)} trials ({'-'.join(map(str, per[::max(1, len(per) - 1)]))} per set, {len(sets)} sets) "
          f"-> {out / 'plan.json'}")


def draw(out: Path, room: Room, placed: dict, target_key: str) -> None:
    """Top-down per set: the room (dark: where a table top would hit something, grey: where the robot would) and
    every trial's table (outline) and robot (dot) where they are in the room."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for s, items in placed.items():
        if not items:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        if s != "r3_room_other":  # (its trials are in other rooms)
            img = np.where(room.table.T, 0.15, np.where(room.robot.T, 0.65, 1.0))
            ext = [room.lo[0], room.lo[0] + room.table.shape[0] * room.res, room.lo[1],
                   room.lo[1] + room.table.shape[1] * room.res]
            ax.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
        xs, ys = [], []
        for conf, value in items:
            scene = conf["dr_state_dict"]["scene"]
            turn = room.yaw0 - scene["center_orientation"][2]  # how far the robot, table and faucet turned in the room
            # where a world point is in the room: invert move_room's  q -> Rz(-turn)(q - pivot - shift) + pivot
            off = np.asarray(scene["center_offset"][:2])
            to_room = lambda w: (np.asarray(w) - off) @ rot2(turn).T + room.offset0
            outline = table_outline(scene["table"])
            c = to_room(np.vstack([outline, outline[:1]]))
            ax.plot(c[:, 0], c[:, 1], lw=0.8, color="#2a78d6", alpha=0.6)
            physics = conf["scene_sweep"].get("physics_table")
            if physics:  # the table the robot collides with, where it differs from the drawn one
                pc = table_outline(physics)
                ax.plot(*to_room(np.vstack([pc, pc[:1]])).T, lw=0.8, ls="--", color="#0b0b0b", alpha=0.5)
            r = to_room(conf["dr_state_dict"]["spatial"][conf["robot_cfg"]["uid"]]["position"][:2])
            f = to_room(conf["dr_state_dict"]["spatial"][target_key]["position"][:2])
            ax.plot(*r, "o", ms=3, color="#d03b3b")
            ax.plot(*f, ".", ms=2, color="#0b0b0b")
            xs += [*c[:, 0], r[0]]
            ys += [*c[:, 1], r[1]]
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
        ax.set_aspect("equal")
        ax.set_title(f"{s} ({len(items)} trials): {SETS[s]}\nblue table, red robot, black faucet"
                     + (", dashed: the table the robot collides with"
                        if any(c["scene_sweep"].get("physics_table") for c, _ in items) else ""), fontsize=8,
                     wrap=True)
        fig.tight_layout()
        fig.savefig(out / "scenes" / f"{s}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
