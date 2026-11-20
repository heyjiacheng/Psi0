"""Stage 0 of the in-distribution experiment: the data, and the grid of base poses to render. Psi0 venv, CPU only.

Two steps (the runner calls room_map.py of the scene sweep in between, which needs Isaac's USD libraries):

  data   makes sure these are on disk, downloading what is missing (as the trajectory sweep does):
           checkpoint  the task's released Psi0 checkpoint (or --run-dir)
           train pack  USC-PSI-Lab/psi-data simple/<TASK>.zip: the episodes the checkpoint was trained on (images,
                       32-D states, each episode's recorded scene)
           raw pack    USC-PSI-Lab/psi-data simple-teleop/<TASK>.zip: the teleop recordings the training episodes
                       were cut from; they hold the pelvis pose and all 43 joints, which the train pack does not
         and writes
           run_dir.txt       the checkpoint run dir
           room.json         the room of the grid scene and where SIMPLE puts it (for room_map.py)
           train_index.json  the check that the pack is the checkpoint's training set, and for every training
                             episode its start state: pelvis pose, joints, faucet, table, and where the faucet is in
                             the pelvis frame (the "training range" of base poses relative to the faucet)

  plan   writes plan.json: at most --budget base poses in the scene of training episode --episode, each a pelvis
         position on the floor and a heading. The robot keeps the posture of that episode's first training frame and is
         moved rigidly. Offsets are in the heading frame of the training start pose (x forward, y left):
           anchor  the training start pose itself, --anchor-repeats times (render noise), and once more after each
                   of the three blocks below (what the renderer keeps of where it has been)
           fine    a --fine-step lattice over the training range of base poses, widened by --fine-margin
           medium  a --medium-step lattice over --medium-box around the start pose
           coarse  a lattice over the free floor within --reach of the table, at the finest of --coarse-steps that
                   keeps the plan within the budget
           yaw     the start position turned by --yaws
           facing  a --facing-step lattice over the same floor, the robot turned to face the faucet: the faucet
                   and the table in view from the wrong place (the lattices above keep the training heading)
         A position is kept only where the robot fits: --room-clearance from walls and furniture (room_map.npz) and
         --table-clearance from the table. Every pose gets its ground truth: where the faucet is in the pelvis frame,
         and how far that is outside the training range (pose_dist, 0 inside).
         plan.png shows the plan top-down.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
YAW_LEVER = 0.5  # m per rad: how a heading error counts in pose_dist (10 deg = 8.7 cm)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("data", "plan"))
    p.add_argument("--task", default="G1WholebodyOpenFaucetTeleop-v0")
    p.add_argument("--psi0-dir", required=True)
    p.add_argument("--simple-dir", required=True)
    p.add_argument("--out", required=True, help="the prep directory")
    p.add_argument("--run-dir", default="", help="a local checkpoint run dir; default: the task's released one")
    p.add_argument("--ckpt-step", default="40000")
    p.add_argument("--episode", type=int, default=0, help="training episode whose scene the grid is rendered in")
    p.add_argument("--budget", type=int, default=300, help="most base poses to render")
    p.add_argument("--anchor-repeats", type=int, default=4)
    p.add_argument("--fine-step", type=float, default=0.05)
    p.add_argument("--fine-margin", type=float, default=0.10, help="m the fine lattice reaches past the training range")
    p.add_argument("--medium-step", type=float, default=0.10)
    p.add_argument("--medium-box", type=float, nargs=4, default=(-0.5, 0.3, -0.6, 0.6),
                   metavar=("X_LO", "X_HI", "Y_LO", "Y_HI"), help="m around the start pose, heading frame")
    p.add_argument("--coarse-steps", default="0.2,0.25,0.3,0.35,0.4,0.5,0.6",
                   help="candidate spacings of the coarse lattice, finest first")
    p.add_argument("--reach", type=float, default=1.4, help="m around the table the coarse lattice covers")
    p.add_argument("--yaws", default="-135,-90,-60,-45,-30,-20,-15,-10,-5,5,10,15,20,30,45,60,90,135,180",
                   help="deg; headings at the start position ('' = none)")
    p.add_argument("--facing-step", type=float, default=0.5, help="m; lattice of the poses that face the faucet (0 = none)")
    p.add_argument("--room-clearance", type=float, default=0.30, help="m between the pelvis and walls or furniture")
    p.add_argument("--table-clearance", type=float, default=0.25, help="m between the pelvis and the table")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- geometry
def yaw_of(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_frame(origin_xy, origin_yaw: float, xy) -> np.ndarray:
    """Points (..., 2) in the frame at origin_xy turned by origin_yaw: x ahead, y to the left."""
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    d = np.asarray(xy, dtype=float) - np.asarray(origin_xy, dtype=float)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)


def from_frame(origin_xy, origin_yaw: float, rel) -> np.ndarray:
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    r = np.asarray(rel, dtype=float)
    return np.stack([origin_xy[0] + c * r[..., 0] - s * r[..., 1], origin_xy[1] + s * r[..., 0] + c * r[..., 1]], axis=-1)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def box_dist(v: float, lo: float, hi: float) -> float:
    return max(lo - v, 0.0, v - hi)


def table_distance(table: dict, xy) -> np.ndarray:
    """Distance from points (..., 2) to the table top's rectangle (0 inside)."""
    half = np.asarray(table["size"][:2], dtype=float) / 2
    local = np.abs(to_frame(table["position"][:2], table["yaw"], xy))
    return np.linalg.norm(np.maximum(local - half, 0.0), axis=-1)


# ---------------------------------------------------------------------------------------------- data
def step_data(args) -> None:
    import pandas as pd

    sys.path.insert(0, str(HERE.parent / "target_pose_sweep"))
    sys.path.insert(0, str(HERE.parent / "trajectory_sweep"))
    from prepare_target_sweep import resolve_run_dir
    from prepare_trajectory_sweep import ensure_raw_pack, ensure_train_pack, read_episodes, scene_key

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = resolve_run_dir(args)
    train_dir, raw_dir = ensure_train_pack(args), ensure_raw_pack(args)

    # the pack is the checkpoint's training set: named by the run, and its bounds are the run's normalisation
    run_cfg = json.loads((run_dir / "run_config.json").read_text())
    field = run_cfg["data"]["transform"]["field"]
    stats = json.loads((train_dir / "meta" / field.get("stat_path", "meta/stats_psi0.json").split("/")[-1]).read_text())
    same = {}
    for key, name in ((field.get("stat_state_key", "states"), "state"), (field.get("stat_action_key", "action"), "action")):
        for b in ("min", "max"):
            a, r = np.asarray(stats[key][b], dtype=float), np.asarray(field[f"{name}_{b}"], dtype=float)
            same[f"{name}_{b}"] = bool(np.allclose(a, r[:len(a)], atol=1e-5) and np.allclose(r[len(a):], 0.0))
    info = json.loads((train_dir / "meta" / "info.json").read_text())
    check = dict(ok=bool(args.task in run_cfg["data"]["train_repo_ids"] and all(same.values())),
                 train_repo_ids=run_cfg["data"]["train_repo_ids"], pack=train_dir.name, bounds_equal=same,
                 episodes=info["total_episodes"], frames=info["total_frames"])

    raw_index = {}
    for meta in sorted(raw_dir.glob("*/level-0/meta")):
        for e in read_episodes(meta):
            if "environment_config" in e:
                raw_index[scene_key(json.loads(e["environment_config"]))] = (
                    meta.parent / "data" / "chunk-000" / f"episode_{e['episode_index']:06d}.parquet")
    tasks = [json.loads(line) for line in (train_dir / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
    video_key = next(k for k, v in info["features"].items() if v.get("dtype") == "video")
    episodes = []
    for e in read_episodes(train_dir / "meta"):
        ep = int(e["episode_index"])
        conf = json.loads(e["environment_config"])
        src = raw_index.get(scene_key(conf))
        if src is None or not src.is_file():
            raise SystemExit(f"training episode {ep}: no raw episode with the same scene in {raw_dir}")
        raw = pd.read_parquet(src, columns=["observation.state", "observation.base_pose"])
        pack = pd.read_parquet(train_dir / info["data_path"].format(episode_chunk=ep // info.get("chunks_size", 1000),
                                                                    episode_index=ep))
        # training frame 0 is the raw frame with the same arm joints (raw joints 15..28): frame 1 in this pack
        arm0 = np.asarray(pack["observation.arm_joints"].iloc[0], dtype=np.float64)
        err = [float(np.abs(np.asarray(raw["observation.state"].iloc[o], dtype=np.float64)[15:29] - arm0).max())
               for o in range(min(4, len(raw)))]
        off = int(np.argmin(err))
        if err[off] > 1e-6:
            raise SystemExit(f"training episode {ep}: its first frame is not among the first raw frames of {src} ({err})")
        bp = np.asarray(raw["observation.base_pose"].iloc[off], dtype=np.float64)
        sp, scene = conf["dr_state_dict"]["spatial"], conf["dr_state_dict"]["scene"]
        art = next(v for k, v in sp.items() if k.startswith("articulated"))
        pel_xy, pel_yaw = bp[:2], yaw_of(bp[3:7])
        rel = to_frame(pel_xy, pel_yaw, art["position"][:2])
        table = scene["table"]
        lights = conf["dr_state_dict"].get("lighting") or {}
        episodes.append(dict(
            episode=ep, length=int(e["length"]), raw=str(src.relative_to(raw_dir)), raw_frame=off,
            base_pose=bp.round(6).tolist(),
            joints=np.asarray(raw["observation.state"].iloc[off], dtype=np.float64).round(6).tolist(),
            state=np.asarray(pack["states"].iloc[0], dtype=np.float64).round(6).tolist(),
            spawn=[*np.round(sp[conf["robot_cfg"]["uid"]]["position"][:2], 5).tolist(),
                   round(yaw_of(sp[conf["robot_cfg"]["uid"]]["quaternion"]), 5)],
            pelvis=[*pel_xy.round(5).tolist(), round(pel_yaw, 5)],
            faucet=[*np.round(art["position"][:2], 5).tolist(), round(yaw_of(art["quaternion"]), 5)],
            faucet_rel=[round(float(rel[0]), 5), round(float(rel[1]), 5),
                        round(wrap(yaw_of(art["quaternion"]) - pel_yaw), 5)],
            table=dict(size=table["size"], position=table["pose"]["position"],
                       yaw=round(yaw_of(table["pose"]["quaternion"]), 5),
                       material=(table.get("material") or {}).get("name")),
            room=dict(uid=scene["uid"], offset=scene["center_offset"][:2], yaw=scene["center_orientation"][2]),
            light_K=next((round(float(v["light_color_temperature"])) for v in lights.values()
                          if v.get("light_color_temperature") is not None), None)))
    rel = np.array([e["faucet_rel"] for e in episodes])
    index = dict(
        task=args.task, run_dir=str(run_dir), ckpt_step=str(args.ckpt_step), train_dir=str(train_dir),
        raw_dir=str(raw_dir), check=check, instruction=str(tasks[0]["task"]).lower(), fps=info["fps"],
        video_path=info["video_path"], video_key=video_key.split(".")[-1], chunks_size=info.get("chunks_size", 1000),
        image_shape=info["features"][video_key]["shape"],
        # where the faucet is in the pelvis frame at the first training frame: x ahead, y left, its yaw
        train_range=dict(dx=[float(rel[:, 0].min()), float(rel[:, 0].max())],
                         dy=[float(rel[:, 1].min()), float(rel[:, 1].max())],
                         dyaw=[float(rel[:, 2].min()), float(rel[:, 2].max())]),
        episodes=episodes)
    (out / "train_index.json").write_text(json.dumps(index, indent=1))
    (out / "run_dir.txt").write_text(f"{run_dir}\n")
    if not 0 <= args.episode < len(episodes):
        raise SystemExit(f"--episode {args.episode}: the train pack has episodes 0..{len(episodes) - 1}")
    room = episodes[args.episode]["room"]
    (out / "room.json").write_text(json.dumps(dict(room=room["uid"].split(":")[-1], offset=room["offset"], yaw=room["yaw"])))
    tr = index["train_range"]
    print(f"[prepare] checkpoint {run_dir}")
    print(f"[prepare] train pack {train_dir}: {check['episodes']} episodes, {check['frames']} frames; "
          f"{'IS' if check['ok'] else 'is NOT'} the checkpoint's training set {check['bounds_equal']}")
    print(f"[prepare] every training episode matched to its raw recording (training frame 0 = raw frame "
          f"{sorted({e['raw_frame'] for e in episodes})})")
    print(f"[prepare] faucet in the pelvis frame at the training starts: dx {tr['dx'][0]:+.3f}..{tr['dx'][1]:+.3f} m, "
          f"dy {tr['dy'][0]:+.3f}..{tr['dy'][1]:+.3f} m, yaw {math.degrees(tr['dyaw'][0]):+.1f}.."
          f"{math.degrees(tr['dyaw'][1]):+.1f} deg")
    if not check["ok"]:
        raise SystemExit("the train pack is not the checkpoint's training set; see train_index.json")


# ---------------------------------------------------------------------------------------------- plan
def step_plan(args) -> None:
    from scipy.ndimage import distance_transform_edt, label

    out = Path(args.out)
    index = json.loads((out / "train_index.json").read_text())
    ep = index["episodes"][args.episode]
    tr = index["train_range"]
    x0, y0, yaw0 = ep["pelvis"]
    faucet, table = ep["faucet"], ep["table"]
    room = np.load(out / "room_map.npz")
    lo, res, occ = room["lo"], float(room["res"]), room["robot"]
    clear = distance_transform_edt(~occ) * res  # m from each free cell to the nearest wall or furniture
    # where the pelvis may be: clear of the room and of the table, and reachable from the start pose over such floor
    # (the far side of a wall is free floor too, but not this room's)
    centres = np.stack(np.meshgrid(lo[0] + (np.arange(occ.shape[0]) + 0.5) * res,
                                   lo[1] + (np.arange(occ.shape[1]) + 0.5) * res, indexing="ij"), axis=-1)
    parts, _ = label((clear >= args.room_clearance) & (table_distance(table, centres) >= args.table_clearance))

    def cell(xy) -> tuple[np.ndarray, np.ndarray]:
        ij = np.floor((np.atleast_2d(xy) - lo) / res).astype(int)
        return ij, (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < occ.shape[0]) & (ij[:, 1] < occ.shape[1])

    ij0, ok0 = cell([x0, y0])
    home = int(parts[ij0[0, 0], ij0[0, 1]]) if ok0[0] else 0
    if home == 0:
        raise SystemExit(f"the training start of episode {args.episode} is closer than --room-clearance "
                         f"{args.room_clearance} m to the room or --table-clearance {args.table_clearance} m to the table")

    def room_clear(xy) -> np.ndarray:
        ij, inside = cell(xy)
        d = np.zeros(len(ij))
        d[inside] = clear[ij[inside, 0], ij[inside, 1]]
        return d

    def reachable(xy) -> bool:
        ij, inside = cell(xy)
        return bool(inside[0] and parts[ij[0, 0], ij[0, 1]] == home)

    def pose(kind: str, fx: float, fy: float, dyaw: float, cell: float) -> dict:
        xy = from_frame([x0, y0], yaw0, np.array([fx, fy]))
        yaw = wrap(yaw0 + math.radians(dyaw))
        rel = to_frame(xy, yaw, np.asarray(faucet[:2]))
        ryaw = wrap(faucet[2] - yaw)
        d_pos = math.hypot(box_dist(rel[0], *tr["dx"]), box_dist(rel[1], *tr["dy"]))
        d_yaw = box_dist(ryaw, *tr["dyaw"])
        return dict(kind=kind, fx=round(fx, 4), fy=round(fy, 4), dyaw=round(dyaw, 2), cell=cell,
                    x=round(float(xy[0]), 4), y=round(float(xy[1]), 4), yaw=round(yaw, 5),
                    faucet_rel=[round(float(rel[0]), 4), round(float(rel[1]), 4), round(ryaw, 5)],
                    pos_dist=round(d_pos, 4), yaw_dist_deg=round(math.degrees(d_yaw), 2),
                    pose_dist=round(math.hypot(d_pos, YAW_LEVER * d_yaw), 4), in_range=bool(d_pos < 1e-9 and d_yaw < 1e-9),
                    faucet_dist=round(float(np.hypot(*rel)), 4),
                    table_clear=round(float(table_distance(table, xy)), 3), room_clear=round(float(room_clear(xy)[0]), 3),
                    reachable=reachable(xy))

    def toward_faucet(fx: float, fy: float) -> float:
        """Heading offset (deg) that turns the robot at this offset to face the faucet."""
        xy = from_frame([x0, y0], yaw0, np.array([fx, fy]))
        return math.degrees(wrap(math.atan2(faucet[1] - xy[1], faucet[0] - xy[0]) - yaw0))

    def lattice(kind: str, step: float, x_lo: float, x_hi: float, y_lo: float, y_hi: float,
                facing: bool = False) -> list[dict]:
        """Nodes k * step of the start pose's heading frame inside the box, where the robot fits."""
        ix = range(math.ceil(x_lo / step - 1e-9), math.floor(x_hi / step + 1e-9) + 1)
        iy = range(math.ceil(y_lo / step - 1e-9), math.floor(y_hi / step + 1e-9) + 1)
        poses = [pose(kind, i * step, j * step, toward_faucet(i * step, j * step) if facing else 0.0, step)
                 for i in ix for j in iy]
        return [q for q in poses if q.pop("reachable")]

    # the training range of base poses in this scene, heading frame: the faucet seen from the pelvis is where a
    # training start had it (the start pose's own faucet_rel maps to offset 0)
    f0 = ep["faucet_rel"]
    fine_box = (f0[0] - tr["dx"][1] - args.fine_margin, f0[0] - tr["dx"][0] + args.fine_margin,
                f0[1] - tr["dy"][1] - args.fine_margin, f0[1] - tr["dy"][0] + args.fine_margin)
    fine = lattice("fine", args.fine_step, *fine_box)
    medium = lattice("medium", args.medium_step, *args.medium_box)
    yaws = [pose("yaw", 0.0, 0.0, float(v), 0.0) for v in args.yaws.split(",") if v.strip()]
    anchors = [pose("anchor", 0.0, 0.0, 0.0, 0.0) for _ in range(args.anchor_repeats)]
    for q in yaws + anchors:
        q.pop("reachable")

    # the coarse lattice: the floor within --reach of the table, in the heading frame's bounding box of that region
    half = np.asarray(table["size"][:2]) / 2 + args.reach
    corners = from_frame(table["position"][:2], table["yaw"], np.array([[sx * half[0], sy * half[1]]
                                                                       for sx in (-1, 1) for sy in (-1, 1)]))
    cf = to_frame([x0, y0], yaw0, corners)
    ring = (cf[:, 0].min(), cf[:, 0].max(), cf[:, 1].min(), cf[:, 1].max())
    facing = ([q for q in lattice("facing", args.facing_step, *ring, facing=True) if q["table_clear"] <= args.reach]
              if args.facing_step > 0 else [])
    seen = lambda q: (round(q["fx"], 3), round(q["fy"], 3))
    in_fine = {seen(q) for q in fine}
    medium = [q for q in medium if seen(q) not in in_fine]
    taken = in_fine | {seen(q) for q in medium}
    fixed = len(anchors) + 1 + len(fine) + len(medium) + len(yaws) + bool(yaws) + len(facing) + bool(facing)
    coarse = coarse_step = None
    for step in (float(v) for v in args.coarse_steps.split(",")):
        cand = [q for q in lattice("coarse", step, *ring) if q["table_clear"] <= args.reach and seen(q) not in taken]
        if fixed + len(cand) <= args.budget:
            coarse, coarse_step = cand, step
            break
    if coarse is None:
        raise SystemExit(f"no coarse spacing of {args.coarse_steps} keeps the plan within {args.budget} poses "
                         f"({fixed} are taken by the other sets); widen --coarse-steps or shrink another set")

    def tour(items: list[dict]) -> list[dict]:
        """Nearest-neighbour order from the start pose: consecutive renders stay close, which keeps the renderer's
        accumulated lighting settled."""
        rest, done, here = list(items), [], (0.0, 0.0)
        while rest:
            k = int(np.argmin([math.hypot(q["fx"] - here[0], q["fy"] - here[1]) for q in rest]))
            here = (rest[k]["fx"], rest[k]["fy"])
            done.append(rest.pop(k))
        return done

    yaws.sort(key=lambda q: (q["dyaw"] < 0, abs(q["dyaw"])))  # outward on each side
    poses = list(anchors)
    for block in (tour(fine + medium + coarse), yaws, tour(facing)):
        if block:  # the anchor again after every block: what the renderer keeps of where it has been
            poses += block + [dict(anchors[0], after=block[0]["kind"])]
    for i, q in enumerate(poses):
        q["id"] = i
    n_in = sum(q["in_range"] for q in poses if q["kind"] != "anchor")
    plan = dict(
        task=index["task"], episode=args.episode, budget=args.budget, n=len(poses),
        counts={k: sum(q["kind"] == k for q in poses) for k in ("anchor", "fine", "medium", "coarse", "yaw", "facing")},
        steps=dict(fine=args.fine_step, medium=args.medium_step, coarse=coarse_step, facing=args.facing_step),
        yaw_lever_m_per_rad=YAW_LEVER,
        clearance=dict(room=args.room_clearance, table=args.table_clearance),
        anchor=dict(episode=args.episode, pelvis=ep["pelvis"], base_pose=ep["base_pose"], faucet=faucet,
                    faucet_rel=ep["faucet_rel"], table=table),
        train_range=tr, poses=poses)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    draw_plan(out, plan, index, room)
    print(f"[plan] {len(poses)} base poses (budget {args.budget}) in the scene of training episode {args.episode}: "
          + ", ".join(f"{v} {k}" for k, v in plan["counts"].items())
          + f"; lattice steps fine {args.fine_step} / medium {args.medium_step} / coarse {coarse_step} / facing "
          f"{args.facing_step} m; "
          f"{n_in} poses inside the training range")


def draw_plan(out: Path, plan: dict, index: dict, room) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from indist_common import C1, C2, C3, MUTED, draw_scene, scene_handles, scene_labels, style

    style()
    facing = [q for q in plan["poses"] if q["kind"] == "facing"]
    fig, axs = plt.subplots(1, 3 if facing else 2, figsize=(18.5 if facing else 13, 6.6),
                            gridspec_kw=dict(width_ratios=[1.25, 1, 1.25][:3 if facing else 2]))
    kinds = (("coarse", MUTED, 14), ("medium", C1, 12), ("fine", C2, 9), ("yaw", C3, 26))
    for ax, zoom in zip(axs, (False, True)):
        draw_scene(ax, plan, index, room, zoom=zoom)
        for kind, color, size in kinds:
            pts = np.array([[q["fy"], q["fx"]] for q in plan["poses"] if q["kind"] == kind]).reshape(-1, 2)
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=size * (2.2 if zoom else 1), color=color, lw=0, zorder=4,
                           marker="*" if kind == "yaw" else "o",
                           label=f"{kind}: {len(pts)}" + (" headings at the start" if kind == "yaw" else ""))
        ax.set_title("training heading: around the training start" if zoom else "training heading: the whole plan",
                     loc="left")
    handles, names = axs[0].get_legend_handles_labels()
    fig.legend(handles + scene_handles(), names + scene_labels(index), loc="lower left", bbox_to_anchor=(0.03, -0.035),
               ncol=7, fontsize=8)
    if facing:
        ax = axs[2]
        draw_scene(ax, plan, index, room, labels=False)
        yaw0 = plan["anchor"]["pelvis"][2]
        fy, fx = np.array([q["fy"] for q in facing]), np.array([q["fx"] for q in facing])
        a = np.array([q["yaw"] - yaw0 for q in facing])  # heading in the heading frame: 0 = up the page
        ax.quiver(fy, fx, np.sin(a), np.cos(a), color=C1, angles="xy", scale_units="xy", scale=4.5, width=0.006,
                  headwidth=4, zorder=4, pivot="tail")
        ax.scatter(fy, fx, s=10, color=C1, lw=0, zorder=5)
        ax.set_title(f"facing the faucet: {len(facing)} poses", loc="left")
    fig.savefig(out / "plan.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    a = parse_args()
    step_data(a) if a.step == "data" else step_plan(a)
