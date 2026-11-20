"""Stage 0 of the LLMD experiment (experiment 4): the plan of the new single-frame renders. Psi0 venv, CPU only.

Experiment 4 asks whether the LLMD detector of VLA-FAIL (Mahalanobis distance of the action expert's last-layer
features to the training data) tells a training-like standing pose from a far, sideways or turned one. The 281
renders of experiment 3 (one training scene) are scored again; this stage plans the renders that are still missing
and writes <out>/render_plan.json for render_starts.py. Three sets, each item one frame:

  start   every training episode's own start: its recorded scene, the robot at its first training frame. A training
          image rendered again (render_starts.py measures how far the re-render is from the stored frame).
  unused  the starts of the raw teleop recordings no training episode was cut from, each in its own recorded scene:
          starts drawn like the training starts that the checkpoint has not seen. Kept only where the start is
          inside the training range of base poses, in a room and with a faucet the training set uses; the others
          are listed under "dropped" with the reason.
  extra   19 more base poses in experiment 3's scene (the training episode of its plan.json), the posture of that
          episode's first training frame kept and the pelvis moved rigidly over the floor as render_base_grid.py's
          place() does: the approach line inside the training range, closer than any training start, turned at
          two other distances, and sideways turned to face the faucet.

Read from experiment 3 (--exp3, never written): prep/train_index.json (the 99 training starts and the training
range), prep/plan.json (the anchor scene), prep/room_map.npz (room clearance); and the train and raw teleop packs.

Every item carries what the renderer needs (base_pose, the 43 joints in the raw recording's order = robot.joint_names,
the recorded poses of the free objects on the table) and what the model needs (the 32-D state), plus experiment 3's
ground truth: faucet_rel (the faucet in the pelvis frame), pos_dist, yaw_dist_deg, pose_dist, in_range, with the
formulas and the train_range of prepare_base_grid.py.

The 32-D state of a raw frame is built as SIMPLE's scripts/postprocess_psi0_sonic.py builds the pack's "states"
column (see model_state). Checks, all run on every call and written to the plan under "checks"; the script stops if
one fails:
  states      the derivation reproduces states[0] of all training episodes from their raw recordings
  formulas    the ground truth re-computed for every pose of experiment 3's plan.json equals what is stored there
  scenes      each training episode's scene in the train pack equals the scene of its raw recording, so a raw
              recording's own scene line is the same kind of record
  pairing     each unused recording's object poses lie where its scene line puts them (the parquet file and the
              line of meta/episodes.jsonl belong together)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# the SAME geometry as experiment 3: its functions, not copies of them
sys.path.insert(0, str(HERE.parent / "vlm_camera_sweep"))
sys.path.insert(0, str(HERE.parent / "trajectory_sweep"))
from prepare_base_grid import YAW_LEVER, box_dist, from_frame, table_distance, to_frame, wrap, yaw_of  # noqa: E402
from prepare_trajectory_sweep import scene_key  # noqa: E402

# the extra base poses: offsets in the heading frame of the grid episode's start (x forward, y left), deg + = left
APPROACH = (-0.03, 0.02, 0.04, 0.07, 0.09)  # m forward: the approach line inside the training range
CLOSER = (0.25, 0.30)                       # m forward: closer to the faucet than any training start
TURNED_AT = (-0.25, 0.15)                   # m forward, each turned by TURNED_BY
TURNED_BY = (-20.0, -10.0, 10.0, 20.0)
SIDEWAYS = (-0.30, -0.15, 0.15, 0.30)       # m to the left, the robot turned to face the faucet
FLAG_TABLE, FLAG_ROOM = 0.12, 0.25          # m: an extra pose closer than this to the table / the room is flagged

# postprocess_psi0_sonic.py, STATE_SLICES: where the 14 hand and 14 arm joints sit in the 43 raw joints
# (left thumb, left middle, left index, right hand, left arm, right arm), then the waist joints as roll, pitch, yaw
# (raw order is yaw, roll, pitch), then the height command of the frame before (0.74 before the first frame)
STATE_SLICES = ((29, 32), (34, 36), (32, 34), (36, 43), (15, 22), (22, 29))
WAIST_RPY = (13, 14, 12)
DEFAULT_HEIGHT = 0.74
RAW_COLUMNS = ["observation.state", "observation.base_pose", "observation.object_poses", "teleop.base_height_command"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp3", required=True, help="experiment 3's output directory (holds prep/)")
    p.add_argument("--simple-dir", required=True, help="the SIMPLE repo: where the packs are looked for if the paths "
                                                       "in experiment 3's train_index.json are gone")
    p.add_argument("--out", required=True, help="the prep directory; render_plan.json is written there")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- the packs
def pack_dir(recorded: str, fallback: Path) -> Path:
    """The pack where experiment 3 found it, or where SIMPLE keeps it."""
    for path in (Path(recorded), fallback):
        if (path / "meta").is_dir() or any(path.glob("*/level-0/meta")):
            return path
    raise SystemExit(f"pack not found: {recorded} or {fallback}")


def read_raw(path: Path) -> dict:
    """One raw teleop recording: the 43 joints, the pelvis pose, the free objects' poses and the height command."""
    import pandas as pd

    df = pd.read_parquet(path, columns=RAW_COLUMNS)
    return dict(q=np.stack(df["observation.state"]).astype(np.float64),
                base=np.stack(df["observation.base_pose"]).astype(np.float64),
                objects=np.stack(df["observation.object_poses"]).astype(np.float64),
                height=np.asarray(df["teleop.base_height_command"], dtype=np.float64).reshape(-1))


def model_state(raw: dict, frame: int) -> np.ndarray:
    """The 32-D model state of raw frame `frame`, as postprocess_psi0_sonic.py's build_vectors() makes it (float32):
    14 hand joints, 14 arm joints, the waist as roll/pitch/yaw, the height command of the frame before."""
    q = raw["q"][frame].astype(np.float32)
    prev_height = np.float32(DEFAULT_HEIGHT if frame == 0 else raw["height"][frame - 1])
    return np.concatenate([*(q[s:e] for s, e in STATE_SLICES), q[list(WAIST_RPY)], [prev_height]]).astype(np.float32)


def raw_recordings(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    """Every raw recording that has a scene line in its pack's meta/episodes.jsonl, and the parquet files without."""
    with_scene, without = [], []
    for meta in sorted(raw_dir.glob("*/level-0/meta")):
        pack, data, seen = meta.parent.parent.name, meta.parent / "data", set()
        names = json.loads((meta / "info.json").read_text())["features"]
        for n, line in enumerate((meta / "episodes.jsonl").read_text().splitlines()):
            if not line.strip():
                continue
            e = json.loads(line)
            path = data / "chunk-000" / f"episode_{e['episode_index']:06d}.parquet"
            seen.add(path)
            if "environment_config" in e:
                with_scene.append(dict(pack=pack, episode=int(e["episode_index"]), jsonl=meta / "episodes.jsonl",
                                       line=n, conf=json.loads(e["environment_config"]), path=path,
                                       rel=str(path.relative_to(raw_dir)),
                                       joint_names=names["observation.state"]["names"],
                                       object_names=names["observation.object_poses"]["names"]))
            else:
                without.append(dict(raw=str(path.relative_to(raw_dir)), reason="its line of meta/episodes.jsonl has no scene"))
        without += [dict(raw=str(path.relative_to(raw_dir)), reason="no line in meta/episodes.jsonl (no recorded scene)")
                    for path in sorted(data.glob("chunk-*/episode_*.parquet")) if path not in seen]
    return with_scene, without


def object_labels(names: list[str]) -> list[str]:
    """The free objects of observation.object_poses in order (7 numbers each): target, distractor_0, ..."""
    labels = list(dict.fromkeys(n.split(".")[0] for n in names))
    if len(names) != 7 * len(labels):
        raise SystemExit(f"observation.object_poses: {len(names)} entries are not 7 per object ({labels})")
    return labels


def object_poses(raw: dict, frame: int, labels: list[str]) -> dict:
    """Where the free objects were at that frame (the training renderer put them there frame by frame; a few had
    slid or tipped by the time the recording started, so the scene's reset layout is not the same thing)."""
    return {name: raw["objects"][frame, 7 * k:7 * k + 7].round(6).tolist() for k, name in enumerate(labels)}


def scene_objects(conf: dict) -> np.ndarray:
    """Where the scene line puts the free objects, in the order of observation.object_poses (xyz each)."""
    dr = conf["dr_state_dict"]
    keys = [dr["target"]["uid"]] + list(dr["distractors"] or {})
    return np.array([dr["spatial"][k]["position"] for k in keys], dtype=float)


def scene_assets(conf: dict) -> tuple[str, str]:
    """The room and the faucet asset of a recorded scene."""
    dr = conf["dr_state_dict"]
    faucet = next(k for k in dr["spatial"] if k.startswith("articulated"))
    return dr["scene"]["uid"], f"{faucet}:{Path((dr.get('articulated') or {}).get('usd_path') or '').name}"


# ---------------------------------------------------------------------------------------------- ground truth
def ground_truth(rel_xy, rel_yaw: float, tr: dict) -> dict:
    """How far the faucet, seen from the pelvis, is outside the training range: prepare_base_grid.py's pose()."""
    d_pos = math.hypot(box_dist(rel_xy[0], *tr["dx"]), box_dist(rel_xy[1], *tr["dy"]))
    d_yaw = box_dist(rel_yaw, *tr["dyaw"])
    return dict(pos_dist=round(d_pos, 4), yaw_dist_deg=round(math.degrees(d_yaw), 2),
                pose_dist=round(math.hypot(d_pos, YAW_LEVER * d_yaw), 4), in_range=bool(d_pos < 1e-9 and d_yaw < 1e-9))


def recorded_rel(base_pose: np.ndarray, conf: dict) -> list[float]:
    """The faucet in the pelvis frame of a recorded start, rounded as prepare_base_grid.py's step_data() stores it
    (the training range is the box of these rounded values)."""
    art = next(v for k, v in conf["dr_state_dict"]["spatial"].items() if k.startswith("articulated"))
    yaw = yaw_of(base_pose[3:7])
    rel = to_frame(base_pose[:2], yaw, art["position"][:2])
    return [round(float(rel[0]), 5), round(float(rel[1]), 5), round(wrap(yaw_of(art["quaternion"]) - yaw), 5)]


class Grid:
    """Base poses around one training start, in its scene: prepare_base_grid.py's pose() and toward_faucet(), and
    the pelvis pose render_base_grid.py's place() gives them."""

    def __init__(self, ep: dict, tr: dict, room) -> None:
        from scipy.ndimage import distance_transform_edt

        self.ep, self.tr = ep, tr
        self.x0, self.y0, self.yaw0 = ep["pelvis"]
        self.faucet, self.table = ep["faucet"], ep["table"]
        self.lo, self.res, occ = room["lo"], float(room["res"]), room["robot"]
        self.clear = distance_transform_edt(~occ) * self.res  # m from each free cell to the nearest wall or furniture

    def room_clear(self, xy) -> float:
        i, j = np.floor((np.asarray(xy) - self.lo) / self.res).astype(int)
        inside = 0 <= i < self.clear.shape[0] and 0 <= j < self.clear.shape[1]
        return float(self.clear[i, j]) if inside else 0.0

    def toward_faucet(self, fx: float, fy: float) -> float:
        """Heading offset (deg) that turns the robot at this offset to face the faucet."""
        xy = from_frame([self.x0, self.y0], self.yaw0, np.array([fx, fy]))
        return math.degrees(wrap(math.atan2(self.faucet[1] - xy[1], self.faucet[0] - xy[0]) - self.yaw0))

    def pose(self, kind: str, fx: float, fy: float, dyaw: float) -> dict:
        xy = from_frame([self.x0, self.y0], self.yaw0, np.array([fx, fy]))
        yaw = wrap(self.yaw0 + math.radians(dyaw))
        rel = to_frame(xy, yaw, np.asarray(self.faucet[:2]))
        ryaw = wrap(self.faucet[2] - yaw)
        return dict(kind=kind, fx=round(fx, 4), fy=round(fy, 4), dyaw=round(dyaw, 2),
                    x=round(float(xy[0]), 4), y=round(float(xy[1]), 4), yaw=round(yaw, 5),
                    faucet_rel=[round(float(rel[0]), 4), round(float(rel[1]), 4), round(ryaw, 5)],
                    **ground_truth(rel, ryaw, self.tr), faucet_dist=round(float(np.hypot(*rel)), 4),
                    table_clear=round(float(table_distance(self.table, xy)), 3),
                    room_clear=round(self.room_clear(xy), 3))

    def base_pose(self, q: dict) -> list[float]:
        """place(): the pelvis at the pose's (x, y) and heading; height, tilt and posture of the start kept."""
        import transforms3d as t3d

        base0 = np.asarray(self.ep["base_pose"], dtype=np.float64)
        turn = t3d.axangles.axangle2mat([0, 0, 1], q["yaw"] - yaw_of(base0[3:7]))
        quat = t3d.quaternions.mat2quat(turn @ t3d.quaternions.quat2mat(base0[3:7]))
        return [q["x"], q["y"], float(base0[2]), *np.round(quat, 6).tolist()]

    def extras(self) -> list[dict]:
        poses = [self.pose("approach", fx, 0.0, 0.0) for fx in APPROACH]
        poses += [self.pose("closer", fx, 0.0, 0.0) for fx in CLOSER]
        poses += [self.pose("turned", fx, 0.0, dyaw) for fx in TURNED_AT for dyaw in TURNED_BY]
        poses += [self.pose("sideways", 0.0, fy, self.toward_faucet(0.0, fy)) for fy in SIDEWAYS]
        for q in poses:
            q["clearance_flag"] = bool(q["table_clear"] < FLAG_TABLE or q["room_clear"] < FLAG_ROOM)
        return poses


# ---------------------------------------------------------------------------------------------- checks
def check_states(index: dict, raw_dir: Path, train_dir: Path) -> tuple[dict, dict]:
    """Rebuild states[0] of every training episode from its raw recording; returns the check and the recordings."""
    import pandas as pd

    info = json.loads((train_dir / "meta" / "info.json").read_text())
    worst, raws = np.zeros(32), {}
    for e in index["episodes"]:
        raw = raws[e["episode"]] = read_raw(raw_dir / e["raw"])
        pack = pd.read_parquet(train_dir / info["data_path"].format(
            episode_chunk=e["episode"] // info.get("chunks_size", 1000), episode_index=e["episode"]), columns=["states"])
        want = np.asarray(pack["states"].iloc[0], dtype=np.float32)
        got = model_state(raw, e["raw_frame"])
        worst = np.maximum(worst, np.abs(got.astype(np.float64) - want.astype(np.float64)))
        # train_index.json keeps the same state rounded to 6 decimals; the plan stores states that way too
        if np.abs(got.astype(np.float64).round(6) - np.asarray(e["state"])).max() > 1e-9:
            raise SystemExit(f"training episode {e['episode']}: its state in train_index.json is not the pack's states[0]")
    check = dict(episodes=len(index["episodes"]), max_abs_error=float(worst.max()),
                 entries_that_differ={int(k): float(v) for k, v in enumerate(worst) if v > 0})
    if worst.max() > 1e-6:
        raise SystemExit("the 32-D state is not reproduced from the raw recordings; max abs error per entry: "
                         f"{check['entries_that_differ']}")
    return check, raws


def check_formulas(index: dict, plan3: dict, room) -> dict:
    """The ground truth of this script against experiment 3's own plan: every pose re-computed, every field equal."""
    grid = Grid(index["episodes"][plan3["episode"]], plan3["train_range"], room)
    bad, fields = [], set()
    for q in plan3["poses"]:
        mine = grid.pose(q["kind"], q["fx"], q["fy"], q["dyaw"])
        fields |= set(mine)
        # a facing pose's heading is stored rounded to 0.01 deg, which moves what follows from it by up to 1e-4 rad
        # (and a rounded yaw_dist_deg by one step); every other pose has to come out identical
        facing = q["kind"] == "facing"
        for k, v in mine.items():
            if isinstance(v, (str, bool)):
                same = v == q[k]
            else:
                same = np.allclose(v, q[k], rtol=0, atol=(0.011 if k == "yaw_dist_deg" else 2e-4) if facing else 1e-9)
            if not same:
                bad.append((q["id"], k, v, q[k]))
    if bad:
        raise SystemExit(f"the ground truth formulas do not reproduce experiment 3's plan.json: {bad[:5]}")
    return dict(poses=len(plan3["poses"]), exact=sum(q["kind"] != "facing" for q in plan3["poses"]),
                within_heading_rounding=sum(q["kind"] == "facing" for q in plan3["poses"]),
                fields=sorted(fields), mismatches=0)


def outside(value: np.ndarray, pool: np.ndarray) -> float:
    """How far the entry of `value` furthest outside the per-entry min..max of `pool` is outside it (0 = inside)."""
    return float(np.maximum(np.maximum(pool.min(0) - value, value - pool.max(0)), 0.0).max())


# ---------------------------------------------------------------------------------------------- the plan
def main() -> None:
    args = parse_args()
    exp3, simple, out = Path(args.exp3).resolve(), Path(args.simple_dir).resolve(), Path(args.out)
    index = json.loads((exp3 / "prep" / "train_index.json").read_text())
    plan3 = json.loads((exp3 / "prep" / "plan.json").read_text())
    room = np.load(exp3 / "prep" / "room_map.npz")
    task, tr = index["task"], plan3["train_range"]
    train_dir = pack_dir(index["train_dir"], simple / "data" / "psi-data" / "simple" / task)
    raw_dir = pack_dir(index["raw_dir"], simple / "data" / "psi-data" / "simple-teleop" / task)
    episodes, grid_ep = index["episodes"], plan3["episode"]
    if tr != index["train_range"]:
        raise SystemExit("plan.json and train_index.json of experiment 3 disagree on the training range")
    ep0 = episodes[grid_ep]
    if str(room["room"]) != ep0["room"]["uid"].split(":")[-1] or not np.allclose(room["offset"], ep0["room"]["offset"]):
        raise SystemExit(f"room_map.npz is the map of {room['room']}, not of training episode {grid_ep}'s room")
    offsets = sorted({e["raw_frame"] for e in episodes})
    if len(offsets) != 1:
        raise SystemExit(f"training frame 0 is not the same raw frame in every episode: {offsets}")
    off = offsets[0]

    checks = {}
    checks["states"], train_raw = check_states(index, raw_dir, train_dir)
    checks["formulas"] = check_formulas(index, plan3, room)
    recordings, dropped = raw_recordings(raw_dir)
    by_rel = {r["rel"]: r for r in recordings}
    joint_names = recordings[0]["joint_names"]
    labels = object_labels(recordings[0]["object_names"])
    if any(r["joint_names"] != joint_names or r["object_names"] != recordings[0]["object_names"] for r in recordings):
        raise SystemExit("the raw packs do not list the same joints / objects")

    # ------------------------------------------------------------------ start: the 99 training starts
    train_lines = (train_dir / "meta" / "episodes.jsonl").read_text().splitlines()
    grid = Grid(ep0, tr, room)
    scenes, same_scene, object_gap = [], 0, {}
    for e in episodes:
        k = e["episode"]
        line = json.loads(train_lines[k])
        if int(line["episode_index"]) != k:
            raise SystemExit(f"line {k} of the train pack's episodes.jsonl is episode {line['episode_index']}")
        conf = json.loads(line["environment_config"])
        same_scene += conf == by_rel[e["raw"]]["conf"]
        object_gap[k] = np.linalg.norm(train_raw[k]["objects"][e["raw_frame"]].reshape(-1, 7)[:, :3]
                                       - scene_objects(conf), axis=1)
        item = dict(id=f"start_{k:03d}", set="start", kind="start", episode=k, raw=e["raw"], raw_frame=e["raw_frame"],
                    base_pose=e["base_pose"], joints=e["joints"], state=e["state"],
                    object_poses=object_poses(train_raw[k], e["raw_frame"], labels), faucet_rel=e["faucet_rel"],
                    **ground_truth(e["faucet_rel"][:2], e["faucet_rel"][2], tr))
        if not item["in_range"]:
            raise SystemExit(f"training episode {k} starts outside the training range: {item}")
        scenes.append(dict(scene=f"train_{k:03d}", source="train", episode=k,
                           episodes_jsonl=str(train_dir / "meta" / "episodes.jsonl"), line=k, items=[item]))
    checks["scenes"] = dict(train_scene_equals_raw_scene=int(same_scene), episodes=len(episodes))
    if same_scene != len(episodes):
        raise SystemExit(f"only {same_scene} of {len(episodes)} training scenes equal the scene of their raw recording")

    # ------------------------------------------------------------------ extra: 19 more poses in the grid scene
    grid_scene = next(s for s in scenes if s["episode"] == ep0["episode"])
    start0 = grid_scene["items"][0]
    for n, q in enumerate(grid.extras()):
        grid_scene["items"].append(dict(
            id=f"extra_{n:02d}", set="extra", kind=q.pop("kind"), base_pose=grid.base_pose(q), joints=ep0["joints"],
            state=ep0["state"], object_poses=start0["object_poses"], **q))
    scenes.insert(0, scenes.pop(scenes.index(grid_scene)))  # the grid scene first: start, then its extras

    # ------------------------------------------------------------------ unused: starts the checkpoint has not seen
    used = {e["raw"] for e in episodes}
    train_keys = {scene_key(by_rel[rel]["conf"]) for rel in used}
    train_assets = {scene_assets(by_rel[rel]["conf"]) for rel in used}
    start_states = np.array([e["state"] for e in episodes])
    start_joints = np.array([e["joints"] for e in episodes])
    # how far a recorded object may be from where its scene line puts it: what the training recordings show
    pairing_limit = max(0.005, 2 * max(float(np.median(g)) for g in object_gap.values()))
    unused, pairing = [], []
    for r in recordings:
        if r["rel"] in used:
            continue
        note = dict(raw=r["rel"], pack=r["pack"], episode=r["episode"])
        if scene_key(r["conf"]) in train_keys:
            raise SystemExit(f"{r['rel']}: its scene is a training scene, but experiment 3 matched another recording")
        if scene_assets(r["conf"]) not in train_assets:
            dropped.append(dict(note, reason=f"room / faucet {scene_assets(r['conf'])} is not used by a training episode"))
            continue
        raw = read_raw(r["path"]) if r["path"].is_file() else None
        if raw is None or len(raw["q"]) <= off:
            dropped.append(dict(note, reason=f"no raw frame {off} (file missing or too short)"))
            continue
        gap = float(np.median(np.linalg.norm(raw["objects"][off].reshape(-1, 7)[:, :3] - scene_objects(r["conf"]), axis=1)))
        pairing.append(gap)
        if gap > pairing_limit:
            dropped.append(dict(note, reason=f"its objects are {gap:.3f} m (median) from where its scene line puts them: "
                                             "the file and the line do not belong together"))
            continue
        rel = recorded_rel(raw["base"][off], r["conf"])
        truth = ground_truth(rel[:2], rel[2], tr)
        if not truth["in_range"]:
            dropped.append(dict(note, reason="starts outside the training range of base poses", faucet_rel=rel, **truth))
            continue
        state = model_state(raw, off).astype(np.float64).round(6)
        joints = raw["q"][off].round(6)
        unused.append((r, dict(
            set="unused", kind="unused", raw=r["rel"], raw_frame=off, base_pose=raw["base"][off].round(6).tolist(),
            joints=joints.tolist(), state=state.tolist(), object_poses=object_poses(raw, off, labels), faucet_rel=rel,
            **truth,
            # how far the posture is outside what the 99 training starts span (0 = every entry inside): rad
            state_outside_train=round(outside(state[:31], start_states[:, :31]), 5),
            joints_outside_train=round(outside(joints, start_joints), 5))))
    for n, (r, item) in enumerate(unused):
        scenes.append(dict(scene=f"raw_{r['pack']}_{r['episode']:03d}", source="raw", episode=None,
                           episodes_jsonl=str(r["jsonl"]), line=r["line"], items=[dict(id=f"unused_{n:03d}", **item)]))
    checks["pairing"] = dict(unused_recordings=len(pairing), limit_m=round(pairing_limit, 4),
                             max_median_object_gap_m=round(max(pairing), 5) if pairing else None,
                             train_max_median_object_gap_m=round(max(float(np.median(g)) for g in object_gap.values()), 5))

    items = [it for s in scenes for it in s["items"]]
    extras = [it for it in items if it["set"] == "extra"]
    counts = {k: sum(it["set"] == k for it in items) for k in ("start", "unused", "extra")}
    if len({it["id"] for it in items}) != len(items) or len({s["scene"] for s in scenes}) != len(scenes):
        raise SystemExit("item ids or scene names are not unique")
    if any(len(it["joints"]) != 43 or len(it["state"]) != 32 or len(it["base_pose"]) != 7 for it in items):
        raise SystemExit("an item has the wrong number of joints / state entries")
    plan = dict(task=task, exp3=str(exp3), grid_episode=grid_ep, train_range=tr, counts=counts, dropped=dropped,
                yaw_lever_m_per_rad=YAW_LEVER, raw_frame=off, joint_names=joint_names, object_names=labels,
                checks=checks, scenes=scenes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "render_plan.json").write_text(json.dumps(plan, indent=1))

    st = checks["states"]
    print(f"[prepare] 32-D state rebuilt from the raw recordings for {st['episodes']} training episodes "
          f"(training frame 0 = raw frame {off}): max abs error {st['max_abs_error']:.3g}"
          + (f", entries that differ {st['entries_that_differ']}" if st["entries_that_differ"] else " (exact)"))
    fm = checks["formulas"]
    print(f"[prepare] ground truth formulas: {fm['poses']} poses of experiment 3's plan.json re-computed, {fm['mismatches']} "
          f"differ ({fm['exact']} identical, {fm['within_heading_rounding']} facing poses within the rounding of their "
          f"stored heading); {same_scene}/{len(episodes)} training scenes equal their raw scene")
    reasons = {}
    for d in dropped:
        reasons.setdefault(d["reason"].split(":")[0], []).append(d["raw"])
    print(f"[prepare] raw recordings: {len(recordings)} with a scene, {len(used)} are training episodes, "
          f"{len(recordings) - len(used)} are not; kept {counts['unused']} of those as 'unused' "
          f"(objects within {checks['pairing']['max_median_object_gap_m']} m of their scene line)")
    for reason, raws in reasons.items():
        print(f"[prepare]   dropped {len(raws)}: {reason}: {', '.join(raws[:4])}{' ...' if len(raws) > 4 else ''}")
    for d in dropped:
        if "faucet_rel" in d:
            print(f"[prepare]   {d['raw']}: faucet at {d['faucet_rel']} in the pelvis frame, {d['pos_dist'] * 1000:.1f} mm / "
                  f"{d['yaw_dist_deg']:.2f} deg outside the training range")
    far = [it for it in items if it["set"] == "unused" and max(it["state_outside_train"], it["joints_outside_train"]) > 0.02]
    print(f"[prepare] unused starts: {len(far)} of {counts['unused']} have a joint more than 0.02 rad outside what the "
          f"training starts span (worst {max((it['joints_outside_train'] for it in items if it['set'] == 'unused'), default=0):.3f} rad)")
    print(f"[prepare] extra poses in the scene of training episode {grid_ep} "
          f"(start: faucet at {ep0['faucet_rel']} in the pelvis frame; training range dx {tr['dx']}, dy {tr['dy']}, "
          f"dyaw {tr['dyaw']}):")
    for it in extras:
        print(f"[prepare]   {it['id']} {it['kind']:8s} fx {it['fx']:+.2f} fy {it['fy']:+.2f} dyaw {it['dyaw']:+6.2f}  "
              f"pose_dist {it['pose_dist']:.3f} (pos {it['pos_dist']:.3f} m, yaw {it['yaw_dist_deg']:.1f} deg) "
              f"in_range {int(it['in_range'])}  table {it['table_clear']:.2f} m  room {it['room_clear']:.2f} m"
              + ("  CLEARANCE FLAG" if it["clearance_flag"] else ""))
    wrong = [it["id"] for it in extras if (it["kind"] == "approach") != it["in_range"] and it["kind"] in ("approach", "closer")]
    not_closer = [it["id"] for it in extras if it["kind"] == "closer" and it["faucet_rel"][0] >= tr["dx"][0]]
    if wrong or not_closer:
        print(f"[prepare] WARNING: approach poses outside / closer poses inside the training range: {wrong + not_closer}")
    print(f"[prepare] plan: {len(items)} renders in {len(scenes)} scenes ({counts['start']} start, {counts['unused']} "
          f"unused, {counts['extra']} extra; {sum(it['clearance_flag'] for it in extras)} clearance flags) "
          f"-> {out / 'render_plan.json'}")


if __name__ == "__main__":
    main()
