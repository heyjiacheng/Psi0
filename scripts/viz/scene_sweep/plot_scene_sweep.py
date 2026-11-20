"""Report of the scene sweep (stage 3). Psi0 venv, CPU only.

Reads what make_scene_trials.py and eval_scene_sweep.py wrote (plan.json, scenes/, results.jsonl, traces/, videos/)
and writes

report/ (self-contained: open report/index.html, or share the folder; kept when the rest of the sweep is deleted)
  index.html    report_template.html filled in: per condition the success rate with its 95% (Wilson) interval and the
                trial-by-trial comparison with c0_baseline (McNemar's exact test; trial i has the same base scene and
                randomization in every condition); success against each condition's change (where the table comes
                closer, also where the robot ended up; for the c4 sets, how far its pelvis was from the drawn table
                edge and from the faucet when it stopped walking); every trial as a strip; a top-down map of the
                condition's faucets (each trial lined up on its drawn table edge; click one for its trial); the
                selected trial's video, a few facts and a top-down sketch. Line 1 is a doctype for local viewing;
                drop it to publish the page.
  results.json  the settings, the per-condition numbers and every finished trial's record, with the near edges of
                its drawn and physics tables and the faucet (edge, edge_phys, faucet: m, robot start frame) and,
                for a trial with a gait record, where it stopped walking (stop, see stop_of)
  videos/       one head-camera clip per trial, re-encoded small for the browser

Where the robot stopped walking (stop_of): the end of its walk toward the table. Its legs walk in runs (gear_wbc in
its walking policy). A run is forward when it moves the pelvis at least FORWARD, at least AHEAD times as far forward as
sideways (the demos often walk to the faucet diagonally), from the run's start to the stand after it (at most
SETTLE_S later); in place when it moves it less; else sideways or back (the demos turn the handle by stepping to the
side, and so does the policy, often far). Runs separated by stands
shorter than STAND_S make one bout (the demos also step sideways on the way). The walk toward the table is the first
bout that takes the pelvis SET_OFF or more forward; it ended at that bout's last forward run (so a side-step after
it is not part of it), the pelvis taken as it stood next (median over up to SETTLE_S), in the robot's start frame. A robot
without such a bout did not walk: it stopped where it started. The distances to the table edge and the faucet are
along the robot's start heading: to the drawn table's near edge (what the camera sees), the physics table's (where
the robot would bump into it) and the faucet as MuJoCo placed it.

Everything the page shows is in report/, so it survives the runner deleting the rest of OUT. Safe to run while
the sweep is still going (the runner does, every REPORT_EVERY s).

    python plot_scene_sweep.py --results OUT
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "target_pose_sweep"))
from plot_target_pose import read_jsonl, wilson, write_atomic  # noqa: E402

TEMPLATE = Path(__file__).with_name("report_template.html")
OUTCOMES = {"success": "success", "wrong_way": "turned the wrong way", "arm_or_body": "turned by an arm or the body",
            "moved_not_done": "moved the handle, not open", "reached_not_moved": "reached, did not move it",
            "not_reached": "never reached it", "fell": "fell"}
NAMES = {  # condition -> (code, label)
    "c0_baseline": ("C0", "Baseline"),
    "c1_1_table_y": ("C1.1", "Table sideways"),
    "c1_2_table_width": ("C1.2", "Table width"),
    "c1_3_room_shift": ("C1.3", "Room shifted"),
    "c1_4_room_rotation": ("C1.4", "Room turned"),
    "c2_1_table_x": ("C2.1", "Table closer"),
    "c2_2_table_x_visual": ("C2.2", "Table drawn closer"),
    "c3_1_width_left": ("C3.1", "Wider on the left"),
    "c3_2_width_right": ("C3.2", "Wider on the right"),
    "c3_3_width_both": ("C3.3", "Wider on both sides"),
    "c4_1_faucet_x": ("C4.1", "Faucet farther"),
    "c4_2_edge_x_visual": ("C4.2", "Edge drawn closer"),
    "c0_redraw": ("C0r", "Baseline, redrawn"),
    "c0_with_clutter": ("C0c", "Baseline, with clutter"),
    "t1_edge_x": ("T1", "Near edge moved"),
    "t2_edge_y": ("T2", "Side edge moved"),
    "t3_scale_xy": ("T3", "Table scaled"),
    "t4_rotate": ("T4", "Table turned"),
    "t4_rotate_phys": ("T4p", "Table turned, physical"),
    "t5_shape": ("T5", "Table shape"),
    "t5_shape_phys": ("T5p", "Table shape, physical"),
    "r1_room_shift": ("R1", "Room shifted"),
    "r2_room_rotation": ("R2", "Room turned"),
    "r3_room_other": ("R3", "Other room"),
    "f1_faucet_x": ("F1", "Faucet farther"),
    "f2_faucet_y": ("F2", "Faucet sideways"),
    "f3_faucet_closer": ("F3", "Faucet closer"),
}
# what each set is compared with, trial by trial: the drawn-table sets with the baseline drawn the same way
CONTROL = {s: "c0_redraw" for s in ("t1_edge_x", "t2_edge_y", "t3_scale_xy", "t4_rotate", "t4_rotate_phys", "t5_shape",
                                     "t5_shape_phys")}
SHAPE_NAMES = {"round": "round", "half_round": "half-round", "triangle": "triangle", "trapezoid": "trapezoid",
               "hexagon": "hexagon", "pedestal": "pedestal"}
CLOSER = dict(key="table_closer", scale=100, label="table closer, cm", ticks=[0, 5, 10, 15, 20, 25, 30], min=0)
WIDER = dict(key="extend", scale=100, log=True, ticks=[5, 10, 20, 50, 100, 200])
STOP_X = dict(scale=100, ticks=[0, 5, 10, 15, 20], min=0)
FARTHER = dict(key="faucet_farther", label="faucet farther, cm", **STOP_X)
NEARER = dict(key="faucet_farther", scale=100, label="faucet moved, cm (− = closer)", ticks=[-20, -15, -10, -5, 0])
EDGE = dict(key="edge_closer", label="table edge drawn closer, cm", **STOP_X)
GROUPS = [  # chart groups; panels whose conditions are not in the run are left out
    ("Table", [
        dict(title="Moved sideways", kind="success", sets=["c1_1_table_y"], key="table_dy", scale=100,
             label="cm, + = robot's left", ticks=[-40, -20, 0, 20, 40]),
        dict(title="Width", kind="success", sets=["c1_2_table_width"], key="table_width", scale=100, log=True,
             label="width, cm (recorded 100)", ticks=[20, 50, 100, 200, 400]),
    ]),
    ("Room", [
        dict(title="Moved across the room", kind="success", sets=["c1_3_room_shift"], key="distance", scale=100,
             label="cm", ticks=[50, 100, 150, 200, 250]),
        dict(title="Turned in the room", kind="success", sets=["c1_4_room_rotation"], key="room_rotation_deg",
             scale=1, label="degrees, + = left", ticks=[-180, -90, 0, 90, 180]),
    ]),
    ("Table closer to the robot", [
        dict(title="Physical", kind="success", sets=["c2_1_table_x"], **dict(CLOSER, ticks=[0, 1, 2, 3, 4, 5])),
        dict(title="Drawn only", kind="success", sets=["c2_2_table_x_visual"], **CLOSER),
        dict(title="Pelvis stopped, cm short of the faucet", kind="stand",
             sets=["c2_1_table_x", "c2_2_table_x_visual"], **CLOSER, slope="with the table"),
    ]),
    ("Table wider", [
        dict(title="On the left", kind="success", sets=["c3_1_width_left"], label="cm added on the left", **WIDER),
        dict(title="On the right", kind="success", sets=["c3_2_width_right"], label="cm added on the right", **WIDER),
        dict(title="On both sides", kind="success", sets=["c3_3_width_both"], label="cm added on each side", **WIDER),
    ]),
    ("Faucet farther, table unchanged", [
        dict(title="Stopped walking: pelvis to", kind="stop", sets=["c4_1_faucet_x"], **FARTHER),
        dict(title="Success", kind="success", sets=["c4_1_faucet_x"], **FARTHER),
    ]),
    ("Table edge drawn closer, faucet unchanged", [
        dict(title="Stopped walking: pelvis to", kind="stop", sets=["c4_2_edge_x_visual"], **EDGE),
        dict(title="Success", kind="success", sets=["c4_2_edge_x_visual"], **EDGE),
    ]),
]
SIDEWAYS = [["y", "se", "pelvis"], ["side", "sf", "faucet − pelvis"]]  # cm, + = the robot's left
HEADING = [["heading", "se", "heading"]]  # degrees, + = left
T1 = dict(key="edge_closer", scale=100, label="near edge drawn closer, cm (− farther)", ticks=[-20, -10, 0, 10, 20])
T2 = dict(key="edge_out", scale=100, label="side edge drawn out, cm (− in)", ticks=[-40, -20, 0, 25, 50, 75, 100])
T4 = dict(key="table_turn_deg", scale=1, label="degrees, + = left", ticks=[-45, -30, -15, 0, 15, 30, 45])
T5 = dict(key="shape_i", scale=1, cats=["round", "half", "tri", "trap", "hex", "ped"], label="")
F2 = dict(key="faucet_left", scale=100, label="faucet moved, cm, + = robot's left", ticks=[-25, -10, 0, 10, 25])
GROUPS_FULL = [  # the full preset
    ("Where the robot stopped walking, relative to the faucet", [dict(title="", kind="overview", sets=None)]),
    ("Near edge moved, drawn only", [
        dict(title="Stopped walking: pelvis to", kind="stop", sets=["t1_edge_x"], **T1),
        dict(title="Success", kind="success", sets=["t1_edge_x"], **T1),
    ]),
    ("Side edge moved, drawn only", [
        dict(title="Stopped walking: pelvis to", kind="stop", sets=["t2_edge_y"], **T2),
        dict(title="Success", kind="success", sets=["t2_edge_y"], **T2),
        dict(title="Sideways, against the table's centre", kind="stop", sets=["t2_edge_y"], key="table_centre_left",
             scale=100, label="table centre moved, cm, + = robot's left", ticks=[-25, -10, 0, 10, 25, 50],
             series=SIDEWAYS, track="y"),
    ]),
    ("Table scaled, drawn only", [
        dict(title="Stopped walking: pelvis to", kind="stop", sets=["t3_scale_xy"], key="depth_scale", scale=1, log=True,
             label="depth, × recorded", ticks=[0.4, 0.5, 1, 1.5, 2]),
        dict(title="Success against depth", kind="success", sets=["t3_scale_xy"], key="depth_scale", scale=1, log=True,
             label="depth, × recorded", ticks=[0.4, 0.5, 1, 1.5, 2]),
        dict(title="Success against width", kind="success", sets=["t3_scale_xy"], key="width_scale", scale=1, log=True,
             label="width, × recorded", ticks=[0.4, 0.5, 1, 1.5, 2]),
    ]),
    ("Table turned about the edge in front of the faucet", [
        dict(title="Drawn only: stopped walking", kind="stop", sets=["t4_rotate"], **T4),
        dict(title="Drawn only: success", kind="success", sets=["t4_rotate"], **T4),
        dict(title="Drawn only: heading when it stopped", kind="stop", sets=["t4_rotate"], series=HEADING, mul=1,
             track="heading", **T4),
        dict(title="Also in physics: stopped walking", kind="stop", sets=["t4_rotate_phys"], **T4),
        dict(title="Also in physics: success", kind="success", sets=["t4_rotate_phys"], **T4),
        dict(title="Also in physics: heading when it stopped", kind="stop", sets=["t4_rotate_phys"], series=HEADING,
             mul=1, track="heading", **T4),
    ]),
    ("Table shape", [
        dict(title="Drawn only: stopped walking", kind="stop", sets=["t5_shape"], **T5),
        dict(title="Drawn only: success", kind="success", sets=["t5_shape"], **T5),
        dict(title="Also in physics: stopped walking", kind="stop", sets=["t5_shape_phys"], **T5),
        dict(title="Also in physics: success", kind="success", sets=["t5_shape_phys"], **T5),
    ]),
    ("Room", [
        dict(title="Moved across the room", kind="success", sets=["r1_room_shift"], key="distance", scale=100,
             label="cm", ticks=[50, 100, 150, 200, 250]),
        dict(title="Turned in the room", kind="success", sets=["r2_room_rotation"], key="room_rotation_deg",
             scale=1, label="degrees, + = left", ticks=[-180, -90, 0, 90, 180]),
        dict(title="Another room", kind="success", sets=["r3_room_other"], key="room_i", scale=1, cats="rooms", label=""),
    ]),
    ("Faucet moved, table unchanged", [
        dict(title="Farther: stopped walking, pelvis to", kind="stop", sets=["f1_faucet_x"], **FARTHER),
        dict(title="Farther: closest hand, forward", kind="stop", sets=["f1_faucet_x"], src="hand", track="x",
             series=[["x", "se", "hand"], ["off_x", "sf", "faucet − hand"]], **FARTHER),
        dict(title="Farther: success", kind="success", sets=["f1_faucet_x"], **FARTHER),
        dict(title="Closer: stopped walking, pelvis to", kind="stop", sets=["f3_faucet_closer"], **NEARER),
        dict(title="Closer: success", kind="success", sets=["f3_faucet_closer"], **NEARER),
        dict(title="Sideways: where it stopped", kind="stop", sets=["f2_faucet_y"], series=SIDEWAYS, track="y", **F2),
        dict(title="Sideways: closest hand", kind="stop", sets=["f2_faucet_y"], src="hand", track="y",
             series=[["y", "se", "hand"], ["off_y", "sf", "faucet − hand"]], **F2),
        dict(title="Sideways: success", kind="success", sets=["f2_faucet_y"], **F2),
    ]),
]
PATH_POINTS = 60  # most points kept of each hand's (and the pelvis') path
FORWARD = 0.01    # m a walking run moves the pelvis to count as moving it
AHEAD = 0.3       # and it is forward when it moves it at least this times as far forward as sideways (within ~73°)
SET_OFF = 0.03    # m forward a bout of runs must take the pelvis to be the walk toward the table
STAND_S = 2.0     # s of standing that end a bout
SETTLE_S = 1.0    # s after a run over which the pelvis is taken


def describe(s: str, v: dict) -> str:
    if s == "c1_1_table_y":
        return f"table {abs(v['table_dy']) * 100:.0f} cm to the {'left' if v['table_dy'] > 0 else 'right'}"
    if s == "c1_2_table_width":
        n = len(v.get("clutter_removed", []))
        return f"table {v['table_width'] * 100:.0f} cm wide" + (f", {n} objects removed" if n else "")
    if s == "c1_3_room_shift":
        return f"moved {v['distance'] * 100:.0f} cm across the room"
    if s == "c1_4_room_rotation":
        return f"turned {abs(v['room_rotation_deg']):.0f}° {'left' if v['room_rotation_deg'] > 0 else 'right'} in the room"
    if s == "c2_1_table_x":
        return f"table {v['table_closer'] * 100:.1f} cm closer"
    if s == "c2_2_table_x_visual":
        return f"table drawn {v['table_closer'] * 100:.1f} cm closer"
    if s in ("c3_1_width_left", "c3_2_width_right", "c3_3_width_both"):
        where = {"left": "on the left", "right": "on the right", "both": "on each side"}[v["side"]]
        return f"+{v['extend'] * 100:.0f} cm {where}"
    if s == "c4_1_faucet_x":
        n = len(v.get("clutter_removed", []))
        return f"faucet {v['faucet_farther'] * 100:.1f} cm farther" + (f", {n} objects removed" if n else "")
    if s == "c4_2_edge_x_visual":
        return f"table edge drawn {v['edge_closer'] * 100:.1f} cm closer"
    n = len(v.get("clutter_removed", []))
    gone = f", {n} objects removed" if n else ""
    if s == "t1_edge_x":
        d = v["edge_closer"] * 100
        return f"near edge drawn {abs(d):.1f} cm {'closer' if d >= 0 else 'farther'}" + gone
    if s == "t2_edge_y":
        d = v["edge_out"] * 100
        return f"{v['side']} edge drawn {abs(d):.0f} cm {'out' if d >= 0 else 'in'}" + gone
    if s == "t3_scale_xy":
        return f"table drawn {v['depth'] * 100:.0f} cm deep (×{v['depth_scale']:.2f}), {v['width'] * 100:.0f} cm wide " \
               f"(×{v['width_scale']:.2f})" + gone
    if s in ("t4_rotate", "t4_rotate_phys"):
        return f"table {'drawn ' if s == 't4_rotate' else ''}turned {abs(v['table_turn_deg']):.0f}° " \
               f"{'left' if v['table_turn_deg'] > 0 else 'right'}" + gone
    if s in ("t5_shape", "t5_shape_phys"):
        size = ", ".join(f"{k} {v[k] * 100:.0f} cm" for k in ("radius", "base", "height", "near", "far", "depth") if k in v)
        return f"{SHAPE_NAMES[v['shape']]} table{' (drawn)' if s == 't5_shape' else ''}, {size}" + gone
    if s == "r1_room_shift":
        return f"moved {v['distance'] * 100:.0f} cm across the room"
    if s == "r2_room_rotation":
        return f"turned {abs(v['room_rotation_deg']):.0f}° {'left' if v['room_rotation_deg'] > 0 else 'right'} in the room"
    if s == "r3_room_other":
        return f"room {v['room']}" + (f", moved {v['room_shift'] * 100:.0f} cm" if v.get("room_shift") else "") + \
            (f", turned {v['room_turn_deg']:+.0f}°" if v.get("room_turn_deg") else "")
    if s == "f1_faucet_x":
        return f"faucet {v['faucet_farther'] * 100:.1f} cm farther" + gone
    if s == "f3_faucet_closer":
        return f"faucet {abs(v['faucet_farther']) * 100:.1f} cm closer" + gone
    if s == "f2_faucet_y":
        d = v["faucet_left"] * 100
        return f"faucet {abs(d):.1f} cm to the {'left' if d >= 0 else 'right'}" + gone
    return "none"


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def web_video(src: Path, dst: Path) -> None:
    """The clip re-encoded small for the browser (H.264, fast start); a plain copy if ffmpeg is missing."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    tmp = dst.with_name(dst.stem + ".part.mp4")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "medium", "-crf", "30",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    if shutil.which("ffmpeg") and subprocess.run(cmd).returncode == 0:
        tmp.replace(dst)
    else:
        tmp.unlink(missing_ok=True)
        shutil.copyfile(src, dst)


def rot(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def yaw_deg(q) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def outline(tb: dict, arc: int = 48) -> np.ndarray:
    """A table's top outline in its own frame (box, cylinder or prism), counterclockwise."""
    if tb.get("uid") == "cylinder":
        a = np.linspace(0, 2 * math.pi, arc, endpoint=False)
        return 0.5 * tb["size"][0] * np.c_[np.cos(a), np.sin(a)]
    if tb.get("uid") == "prism":
        return np.asarray(tb["vertices"], dtype=float)
    hx, hy = tb["size"][0] / 2, tb["size"][1] / 2
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])


def edge_ahead(poly: list, y: float) -> float:
    """How far ahead of the robot's start (m) a table's outline (robot frame: forward, left) begins at the sideways
    position y: where the line straight ahead at y enters it, or, if it misses the table, its nearest point."""
    p = np.asarray(poly, dtype=float)
    q = np.roll(p, -1, axis=0)
    hits = [a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) for a, b in zip(p, q)
            if (a[1] - y) * (b[1] - y) <= 0 and a[1] != b[1]]
    if hits:
        return float(min(hits))
    near = p[np.argmin(np.abs(p[:, 1] - y))]
    return float(near[0])


def near_edge(tb: dict) -> float:
    """How far ahead of the robot's start (m) a table (robot frame, as robot_frame gives it) begins."""
    return float(min(x for x, _ in tb["poly"]))


def stop_of(gait: dict | None) -> dict | None:
    """Where and when the robot's walk toward the table ended (see the module docstring), from eval_scene_sweep.py's
    gait record: the policy step and t (s), the pelvis x, y then (m, robot start frame), how far forward of its first
    position that is (walked), when that walk began (set_off_t, s; None if it did not walk), whether its legs were
    still in that walk when the trial ended (kept_walking) and how far the pelvis was from the stop at the end
    (moved_after, m). None without a gait record (or one in the world frame)."""
    if not gait or gait.get("frame") != "start" or not gait.get("pelvis"):
        return None
    hz, n, runs = gait["hz"], gait["steps"], gait["walking"]
    p = np.asarray(gait["pelvis"], dtype=float)  # rows: policy steps done, x, y
    at = lambda k: np.array([np.interp(k, p[:, 0], p[:, 1]), np.interp(k, p[:, 0], p[:, 2])])
    settle, stand = int(round(SETTLE_S * hz)), int(round(STAND_S * hz))

    def settled(j: int) -> np.ndarray:
        """The pelvis as it stood after run j (median over up to SETTLE_S of the stand that follows it)."""
        b = runs[j][1]
        q = p[(p[:, 0] >= b) & (p[:, 0] <= min(b + settle, runs[j + 1][0] if j + 1 < len(runs) else n))]
        return np.median(q[:, 1:], axis=0) if len(q) else at(b)

    kind = []  # per run: "forward", "in place" or "other" (sideways or back)
    for j, (a, _) in enumerate(runs):
        d = settled(j) - at(max(a, 1))
        kind.append("forward" if d[0] >= FORWARD and d[0] >= AHEAD * abs(d[1]) else
                    "in place" if math.hypot(*d) < FORWARD else "other")
    walk, bout = None, []  # the walk toward the table: (first, last forward run)
    for j in range(len(runs) + 1):
        if bout and (j == len(runs) or runs[j][0] - runs[bout[-1]][1] >= stand):
            fwd = [i for i in bout if kind[i] == "forward"]
            if fwd and settled(fwd[-1])[0] - at(max(runs[bout[0]][0], 1))[0] >= SET_OFF:
                walk = (fwd[0], fwd[-1])
                break
            bout = []
        if j < len(runs):
            bout.append(j)
    x0 = p[0, 1:]
    if walk is None:  # it did not walk toward the table: it stopped where it started
        q = p[p[:, 0] <= settle]
        xy, k = (np.median(q[:, 1:], axis=0) if len(q) else x0), 0
    else:
        xy, k = settled(walk[1]), runs[walk[1]][1]
    return dict(step=int(k), t=round(k / hz, 2), x=round(float(xy[0]), 4), y=round(float(xy[1]), 4),
                walked=round(float(xy[0] - x0[0]), 4), set_off_t=None if walk is None else round(runs[walk[0]][0] / hz, 2),
                kept_walking=bool(walk is not None and runs[walk[1]][1] >= n),
                moved_after=round(math.hypot(*(p[-1, 1:] - xy)), 4))


def slope(xs, ys) -> float | None:
    """Least-squares slope of ys on xs (None for fewer than 3 points or no spread)."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) < 3 or np.ptp(xs) == 0:
        return None
    return round(float(np.polyfit(xs, ys, 1)[0]), 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="the sweep directory")
    root = Path(p.parse_args().results)
    plan = json.loads((root / "plan.json").read_text())
    st, planned = plan["settings"], plan["trials"]
    sets = list(st["sets"])
    n_per_set = int(st["trials_per_set"])
    rows = {r["trial"]: r for r in read_jsonl(root / "results.jsonl")}
    report = root / "report"
    (report / "videos").mkdir(parents=True, exist_ok=True)

    def robot_frame(conf: dict):
        """The robot's start xy and yaw (deg), and a table (pose, size) in its frame."""
        robot = conf["dr_state_dict"]["spatial"][conf["robot_cfg"]["uid"]]
        rxy, ryaw = np.asarray(robot["position"][:2], dtype=float), yaw_deg(robot["quaternion"])

        def table(tb: dict) -> dict:
            c = rot(-ryaw) @ (np.asarray(tb["pose"]["position"][:2], dtype=float) - rxy)
            yaw = yaw_deg(tb["pose"]["quaternion"]) - ryaw
            poly = outline(tb) @ rot(yaw).T + c
            return dict(cx=round(float(c[0]), 4), cy=round(float(c[1]), 4), sx=round(float(tb["size"][0]), 4),
                        sy=round(float(tb["size"][1]), 4), yaw=round(yaw, 2), shape=tb.get("uid", "box"),
                        poly=poly.round(4).tolist())
        return rxy, ryaw, table

    confs = {t["trial"]: json.loads((root / t["conf"]).read_text()) for t in planned}
    recorded = {t["index"]: robot_frame(confs[t["trial"]])[2](confs[t["trial"]]["dr_state_dict"]["scene"]["table"])
                for t in planned if t["set"] == "c0_baseline"}

    trials, geometry = [], {}  # geometry: per finished trial, for results.json (the scenes go with the cleanup)
    for t in planned:
        r = rows.get(t["trial"])
        conf = confs[t["trial"]]
        ds = conf["dr_state_dict"]
        rxy, ryaw, table = robot_frame(conf)
        drawn = table(ds["scene"]["table"])
        sw = conf.get("scene_sweep") or {}
        reference = sw.get("physics_table") or sw.get("recorded_table")  # what the trial's table was changed from
        ref = table(reference) if reference else recorded.get(t["index"])
        physics = table(sw["physics_table"]) if sw.get("physics_table") else drawn  # the table the robot can touch
        edge, edge_phys = edge_ahead(drawn["poly"], t["dy"]), edge_ahead(physics["poly"], t["dy"])  # in front of the faucet
        entry = dict(s=sets.index(t["set"]), i=t["index"], label=f"t{t['trial']:03d}", trial=t["trial"],
                     change=describe(t["set"], t["value"]), x={k: v for k, v in t["value"].items()
                                                                if isinstance(v, (int, float))},
                     dx=t["dx"], dy=t["dy"], dyaw=t["dyaw"], base=t["base_episode"], seed=t["seed"], table=drawn,
                     table0=ref if ref and ref != drawn else None, edge=round(edge, 4),
                     edge_phys=round(edge_phys, 4) if abs(edge_phys - edge) > 1e-3 else None, done=r is not None)
        if r is not None:
            rel = (r.get("placed") or {}).get("rel") or [t["dx"], t["dy"]]
            end = r.get("final_pelvis_rel")
            entry.update(success=bool(r["success"]), outcome=r["outcome"], steps=r.get("steps"), end=end,
                         gap=None if not end else round(rel[0] - end[0], 4), touch=r.get("table_contact_steps"))
            gait = r.get("gait")
            stop = stop_of(gait)
            if stop is not None:  # the edges straight ahead of where it stopped
                stop.update(to_edge=round(edge_ahead(drawn["poly"], stop["y"]) - stop["x"], 4),
                            to_faucet=round(rel[0] - stop["x"], 4), side=round(rel[1] - stop["y"], 4),
                            to_physical_edge=round(edge_ahead(physics["poly"], stop["y"]) - stop["x"], 4))
            entry["stop"] = stop
            hand = (r.get("closest_hand") or {}).get("rel")
            entry["hand"] = None if not hand else dict(x=hand[0], y=hand[1], off_x=round(rel[0] - hand[0], 4),
                                                       off_y=round(rel[1] - hand[1], 4), d=r.get("min_hand_dist"))
            geometry[t["trial"]] = dict(edge=entry["edge"], edge_phys=round(edge_phys, 4), faucet=rel, stop=stop,
                                        hand=entry["hand"])
            if gait and gait.get("frame") == "start":
                pts = gait["pelvis"]
                entry["path_p"] = [q[1:] for q in pts[:: max(1, math.ceil(len(pts) / PATH_POINTS))]]
            dst = report / "videos" / f"{entry['label']}.mp4"
            vids = sorted(glob.glob(str(root / (r.get("video_dir") or "-") / "head_stereo_left*.mp4")))
            imported = r.get("imported_from") or {}
            if vids:
                web_video(Path(vids[0]), dst)
            elif imported and not dst.exists():  # an imported trial's clip, again from where it came from
                src = Path(imported["report"]) / "videos" / f"{imported['label']}.mp4"
                if src.exists():
                    shutil.copy2(src, dst)
            entry["video"] = f"videos/{dst.name}" if dst.exists() else None
            trace = root / "traces" / f"{entry['label']}.json"
            r0 = (r.get("placed") or {}).get("robot")
            if trace.exists() and r0:
                tr = json.loads(trace.read_text())
                cols, data = tr["columns"], np.asarray(tr["rows"], dtype=float)
                if len(data) and stop is not None:  # which way it faced when it stopped walking (deg, + = left)
                    k = int(np.argmin(np.abs(data[:, cols.index("step")] - stop["step"])))
                    stop["heading"] = round((data[k, cols.index("yaw")] - math.degrees(r0[2]) + 180) % 360 - 180, 2)
                if len(data):
                    keep = data[:: max(1, math.ceil(len(data) / PATH_POINTS))]
                    back = lambda xy: ((xy - r0[:2]) @ rot(-math.degrees(r0[2])).T).round(3).tolist()
                    for side, key in (("r", "rh"), ("l", "lh")):
                        entry[f"path_{side}"] = back(keep[:, [cols.index(f"{key}_x"), cols.index(f"{key}_y")]])
            elif r.get("hand_paths"):  # a trial imported from an earlier run keeps its paths in its record
                entry["path_r"], entry["path_l"] = r["hand_paths"].get("r"), r["hand_paths"].get("l")
        trials.append(entry)

    used_rooms = sorted({t["value"]["room"] for t in planned if t["set"] == "r3_room_other"})
    for entry in trials:  # the other rooms, numbered among those the sweep uses
        if sets[entry["s"]] == "r3_room_other":
            entry["x"]["room_i"] = used_rooms.index(planned[entry["trial"]]["value"]["room"])

    summary, sets_out = {}, []
    by = {(t["s"], t["i"]): t for t in trials}
    for k, s in enumerate(sets):
        done = [t for t in trials if t["s"] == k and t["done"]]
        n, ok = len(done), sum(t["success"] for t in done)
        lo, hi = wilson(ok, n)
        code, label = NAMES.get(s, (s, s))
        e = dict(key=s, code=code, label=label, desc=st["sets"][s], planned=sum(t["s"] == k for t in trials), n=n,
                 k=ok, lo=None if n == 0 else lo, hi=None if n == 0 else hi,
                 outcomes={o: sum(t["outcome"] == o for t in done) for o in OUTCOMES})
        counted = [t for t in done if t.get("touch") is not None]
        if counted:
            e["touched"] = [sum(t["touch"] > 0 for t in counted), len(counted)]
        n_imp = sum(1 for t in done if rows[t["trial"]].get("imported_from"))
        notes = ([("earlier run" if n_imp == n else f"{n_imp}/{n} from an earlier run")] if n_imp else []) + \
            ([f"touched the table {e['touched'][0]}/{e['touched'][1]}"] if counted else [])
        if notes:
            e["note"] = " · ".join(notes)
        ctrl = CONTROL.get(s) if CONTROL.get(s) in sets else "c0_baseline"
        if s not in ("c0_baseline", ctrl) and ctrl in sets:
            b0 = sets.index(ctrl)
            e["control"] = b0
            pairs = [(by[(b0, t["i"])]["success"], t["success"]) for t in done if by.get((b0, t["i"]), {}).get("done")]
            b = sum(a and not c for a, c in pairs)
            c = sum(c and not a for a, c in pairs)
            e["paired"] = dict(pairs=len(pairs), both=sum(a and c for a, c in pairs), baseline_only=b, this_only=c,
                               neither=sum(not a and not c for a, c in pairs), p=mcnemar(b, c), against=ctrl)
        stopped = [t for t in done if t.get("stop")]
        if stopped:  # how the stop distances change with the set's (first numeric) change, cm per cm
            key = next((kk for kk in stopped[0]["x"]), None)
            xs = [t["x"][key] for t in stopped] if key else []
            med = lambda k: round(float(np.median([t["stop"][k] for t in stopped])), 4)
            e["stop"] = dict(n=len(stopped), did_not_walk=sum(t["stop"]["set_off_t"] is None for t in stopped),
                             kept_walking=sum(t["stop"]["kept_walking"] for t in stopped),
                             to_edge=med("to_edge"), to_faucet=med("to_faucet"), side=med("side"),
                             slope_edge=slope(xs, [t["stop"]["to_edge"] for t in stopped]) if key else None,
                             slope_faucet=slope(xs, [t["stop"]["to_faucet"] for t in stopped]) if key else None,
                             slope_y=slope(xs, [t["stop"]["y"] for t in stopped]) if key else None,
                             slope_side=slope(xs, [t["stop"]["side"] for t in stopped]) if key else None,
                             against=key)
            if "control" in e:  # trial by trial: how much farther from the faucet it stopped than in its control
                b0 = e["control"]
                d = [(t["stop"]["to_faucet"] - by[(b0, t["i"])]["stop"]["to_faucet"],
                      t["stop"]["side"] - by[(b0, t["i"])]["stop"]["side"])
                     for t in stopped if (by.get((b0, t["i"])) or {}).get("stop")]
                if d:
                    e["stop"].update(pairs=len(d), d_faucet=round(float(np.median([a for a, _ in d])), 4),
                                     d_side=round(float(np.median([b for _, b in d])), 4))
        sets_out.append(e)
        summary[s] = {key: v for key, v in e.items() if key not in ("key", "code", "label", "note")}

    ranked = sorted((e["paired"]["p"], k) for k, e in enumerate(sets_out) if "paired" in e)  # Holm, over the sets
    running = 0.0
    for j, (pv, k) in enumerate(ranked):
        running = max(running, min(1.0, (len(ranked) - j) * pv))
        sets_out[k]["paired"]["p_holm"] = round(running, 6)
        summary[sets[k]]["paired"] = sets_out[k]["paired"]

    done_rows = list(rows.values())
    walls = [r["wall_seconds"] for r in done_rows if r.get("wall_seconds") and not r.get("imported_from")]
    eta_h = (np.mean(walls) * (len(planned) - len(done_rows)) / 3600) if walls else None
    short = st["task"].replace("G1Wholebody", "").replace("Teleop", "").removesuffix("-v0")
    pretty = "".join(" " + ch if ch.isupper() and i else ch for i, ch in enumerate(short))
    title = f"{pretty} {st.get('preset', 'scene')} sweep"
    full = st.get("preset") == "full"
    train_band = {}
    for key, set_ in (("faucet_farther", "f1_faucet_x"), ("faucet_left", "f2_faucet_y"), ("faucet_farther", "f3_faucet_closer")):
        inr = sorted(t["value"][key] for t in planned if t["set"] == set_ and t.get("in_train_range"))
        if len(inr) >= 5:
            train_band[set_] = [inr[int(0.05 * (len(inr) - 1))], inr[int(round(0.95 * (len(inr) - 1)))]]
    groups = []
    for g, panels in (GROUPS_FULL if full else GROUPS):
        keep = []
        for p in panels:
            p = dict(p, sets=p["sets"] or sets)
            if p.get("cats") == "rooms":
                p["cats"] = [r.removeprefix("scene") for r in used_rooms]
            band = train_band.get((p["sets"] or [None])[0])
            if band and p.get("scale"):
                p["train"] = [round(v * p["scale"], 2) for v in band]
            if any(s in sets for s in p["sets"]):
                keep.append(p)
        groups.append(dict(title=g, panels=keep))
    data = dict(task=st["task"], title=title, checkpoint=str(st.get("run_dir", "")), ckpt_step=str(st.get("ckpt_step", "")),
                updated=time.strftime("%Y-%m-%d %H:%M"), planned=len(planned), done=len(done_rows),
                eta_h=None if eta_h is None else round(float(eta_h), 2), n_per_set=n_per_set,
                per_set=sorted({e["planned"] for e in sets_out}), sets=sets_out,
                groups=[g for g in groups if g["panels"]], outcomes=OUTCOMES, trials=trials)
    page = (TEMPLATE.read_text().replace("__TITLE__", html.escape(title))
            .replace("__DATA_JSON__", json.dumps(data).replace("</", "<\\/")))
    write_atomic(report / "index.html", "<!doctype html>\n" + page)
    records = [dict(r, **geometry.get(r["trial"], {})) for r in sorted(rows.values(), key=lambda r: r["trial"])]
    write_atomic(report / "results.json", json.dumps(dict(settings=st, summary=summary, trials=records), indent=1))
    print(f"[report] {len(done_rows)}/{len(planned)} trials: " + ", ".join(
        f"{NAMES.get(s, (s,))[0]} {summary[s]['k']}/{summary[s]['n']}" for s in sets) + f" -> {report / 'index.html'}")


if __name__ == "__main__":
    main()
