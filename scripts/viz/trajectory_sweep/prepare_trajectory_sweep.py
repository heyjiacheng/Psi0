"""Stage 0 of the trajectory sweep: fetch the data, choose the training episodes, write the trial plan. Psi0 venv, CPU.

For TASK (a SIMPLE Teleop env id, default G1WholebodyOpenFaucetTeleop-v0) it makes sure these are on disk, downloading
what is missing from Hugging Face:
  checkpoint  the task's released Psi0 checkpoint (as in the target-pose sweep)
  train pack  USC-PSI-Lab/psi-data simple/<TASK>.zip, whole: the episodes the checkpoint was trained on (actions,
              32-D states, head-camera videos, each episode's recorded scene)
              -> SIMPLE/data/psi-data/simple/<TASK>
  raw pack    USC-PSI-Lab/psi-data simple-teleop/<TASK>.zip, meta/ and data/ only: the teleop recordings the training
              episodes were cut from. They hold the robot's base pose, which the train pack does not
              -> SIMPLE/data/psi-data/simple-teleop/<TASK>/<sub-pack>/level-0
A training episode is the raw episode with the same recorded scene; training frame t is raw frame t+1.

and writes into --out:
  train_paths.json  every training episode: spawn, faucet, the pelvis path (10 Hz) in the robot's start frame
  plan.json         the settings, the chosen episodes ("slots", each with backups) and one entry per trial
  n_trials.txt      how many of the plan's trials the runner is to finish

Episodes (--n-episodes, default 5): the most spread out. Each episode's pelvis path from the start to where a wrist
comes nearest the faucet, in the faucet's frame, resampled to 20 points along its length; the distance between two
episodes is the mean distance between their points. The first pick is the medoid, every next one the episode
farthest from all picked so far. An episode whose walk is shorter than --min-walk is never picked (its 1/3 and 2/3
points would sit on its start). Each slot keeps its nearest unpicked episodes as backups, for when its replay does not
open the faucet (calibrate stage). --episodes picks them by hand instead.

Trials:
  start  the robot's spawn moved 5 or 10 cm (--radii) in 8 directions, or turned by +-20 and +-40 deg (--yaws), or
         left as recorded. The start pose is the same in every episode, so each start condition runs --repeats times
         in all, the n-th time in slot n's scene, not --repeats times per episode
  path   each slot's training state at 1/3 and 2/3 of the way to its reach point, and at the reach point (found by
         replaying the episode: 0.5 s before its handle starts to turn), --repeats times each; reached by replaying, never moved
Ordered pass by pass (every condition once, then again), so a partial run covers everything. The first trials of the
plan are one of each kind (training start, a 10 cm move, a 40 deg turn, a 1/3 point, a reach point), for smoke tests.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "target_pose_sweep"))
from prepare_target_sweep import HF_DATA_REPO, fetch_zip, resolve_run_dir, yaw_of  # noqa: E402

TRACE_EVERY = 5         # frames between path samples (50 Hz data -> 10 Hz)
DIRS = ["forward", "forward-left", "left", "back-left", "back", "back-right", "right", "forward-right"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="SIMPLE env id without 'simple/', e.g. G1WholebodyOpenFaucetTeleop-v0")
    p.add_argument("--psi0-dir", required=True)
    p.add_argument("--simple-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-dir", default="", help="a local checkpoint run dir; default: the task's released one")
    p.add_argument("--ckpt-step", default="40000")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--episodes", default="", help="comma list of training episodes to use instead of choosing")
    p.add_argument("--min-walk", type=float, default=0.12, help="m; shortest start-to-reach walk an episode may have")
    p.add_argument("--radii", default="0.05,0.10", help="m; start moves, each in 8 directions")
    p.add_argument("--yaws", default="20,40", help="deg; start turns, each to the left and to the right")
    p.add_argument("--repeats", type=int, default=5, help="trials per condition")
    p.add_argument("--reach-lead", type=int, default=25,
                   help="frames; the reach point is this long before the faucet handle starts to turn")
    p.add_argument("--instruction", default="turn the faucet", help="the prompt the policy gets (the training one)")
    p.add_argument("--trials", type=int, default=0, help="run only the first N trials of the plan (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- data
def ensure_train_pack(args) -> Path:
    """The whole train pack (meta, data, videos) at SIMPLE/data/psi-data/simple/<TASK>."""
    dst = (Path(args.simple_dir) / "data" / "psi-data" / "simple" / args.task).resolve()
    if (dst / "meta" / "episodes.jsonl").is_file() and any((dst / "data").glob("chunk-*/*.parquet")) \
            and any((dst / "videos").glob("chunk-*/*/*.mp4")):
        return dst
    stage = Path(args.simple_dir) / "data" / "psi-data" / ".download"
    zp = stage / "simple" / f"{args.task}.zip"
    if not zp.is_file():
        zp = fetch_zip(f"simple/{args.task}.zip", stage)
    with zipfile.ZipFile(zp) as z:
        for m in z.namelist():
            if m.endswith("/") or "/" not in m:
                continue
            out = dst / m.split("/", 1)[1]
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f)
    return dst


def ensure_raw_pack(args) -> Path:
    """meta/ and data/ of every sub-pack of the raw teleop pack, at SIMPLE/data/psi-data/simple-teleop/<TASK>."""
    dst = (Path(args.simple_dir) / "data" / "psi-data" / "simple-teleop" / args.task).resolve()
    if (dst / "complete").is_file():
        return dst
    stage = Path(args.simple_dir) / "data" / "psi-data" / ".download"
    zp = stage / "simple-teleop" / f"{args.task}.zip"
    if not zp.is_file():
        zp = fetch_zip(f"simple-teleop/{args.task}.zip", stage)
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if not n.endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(z.read(n))) as inner:
                for m in inner.namelist():
                    parts = m.split("/")
                    if m.endswith("/") or len(parts) < 4 or parts[2] not in ("meta", "data"):
                        continue
                    out = dst / m
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with inner.open(m) as src, open(out, "wb") as f:
                        shutil.copyfileobj(src, f)
    (dst / "complete").write_text("ok\n")
    return dst


def read_episodes(meta: Path) -> list[dict]:
    return [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines() if line.strip()]


def scene_key(conf: dict) -> tuple:
    """What identifies a recorded scene: robot spawn and articulated-object pose (unique across the pack)."""
    sp = conf["dr_state_dict"]["spatial"]
    robot = sp[conf["robot_cfg"]["uid"]]
    art = next((v for k, v in sp.items() if k.startswith("articulated")), None)
    key = tuple(np.round(robot["position"][:2], 5))
    if art is not None:
        key += tuple(np.round(art["position"][:2], 5)) + tuple(np.round(art["quaternion"], 5))
    return key


def qrot(q, v) -> np.ndarray:
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, dtype=float)


def to_frame(origin_xy, origin_yaw, xy) -> np.ndarray:
    """Points (N, 2) in the frame at origin_xy turned by origin_yaw: x ahead, y to the left."""
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    d = np.atleast_2d(np.asarray(xy, dtype=float)) - np.asarray(origin_xy, dtype=float)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


def resample(path: np.ndarray, n: int = 20) -> np.ndarray:
    """n points spread evenly along the path's length (the path subsampled to 5 Hz first, so the gait sway is gone)."""
    p = path[::10] if len(path) > 20 else path
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if seg[-1] < 1e-6:
        return np.repeat(p[:1], n, axis=0)
    s = np.linspace(0, seg[-1], n)
    return np.stack([np.interp(s, seg, p[:, 0]), np.interp(s, seg, p[:, 1])], axis=1)


def training_paths(train_dir: Path, raw_dir: Path) -> list[dict]:
    """Each training episode's spawn, faucet and pelvis path, from the raw pack's base poses."""
    raw_index = {}
    for meta in sorted(raw_dir.glob("*/level-0/meta")):
        for e in read_episodes(meta):
            if "environment_config" in e:
                conf = json.loads(e["environment_config"])
                raw_index[scene_key(conf)] = meta.parent / "data" / "chunk-000" / f"episode_{e['episode_index']:06d}.parquet"
    out = []
    for e in read_episodes(train_dir / "meta"):
        conf = json.loads(e["environment_config"])
        src = raw_index.get(scene_key(conf))
        if src is None or not src.is_file():
            raise SystemExit(f"training episode {e['episode_index']}: no raw episode with the same scene in {raw_dir}")
        df = pd.read_parquet(src, columns=["observation.base_pose", "observation.eef_state"])
        n = e["length"]
        bp = np.stack(df["observation.base_pose"].values)[1:n + 1]   # training frame t = raw frame t+1
        eef = np.stack(df["observation.eef_state"].values)[1:n + 1]
        sp = conf["dr_state_dict"]["spatial"]
        robot = sp[conf["robot_cfg"]["uid"]]
        spawn_xy, spawn_yaw = np.asarray(robot["position"][:2], dtype=float), yaw_of(robot["quaternion"])
        art = next(v for k, v in sp.items() if k.startswith("articulated"))
        f_xy, f_yaw = np.asarray(art["position"][:2], dtype=float), yaw_of(art["quaternion"])
        yaw = np.array([yaw_of(q) for q in bp[:, 3:7]])
        wrists = np.stack([[bp[i, :3] + qrot(bp[i, 3:7], eef[i, j:j + 3]) for j in (0, 7)] for i in range(len(bp))])
        hand_d = np.linalg.norm(wrists[:, :, :2] - f_xy, axis=2).min(axis=1)
        reach = int(hand_d.argmin())
        rel = to_frame(spawn_xy, spawn_yaw, bp[:, :2])
        in_faucet = to_frame(f_xy, f_yaw, bp[:, :2])
        frel = to_frame(spawn_xy, spawn_yaw, f_xy)[0]
        idx = np.arange(0, len(bp), TRACE_EVERY)
        out.append(dict(
            episode=int(e["episode_index"]), length=int(n), raw=str(src.relative_to(raw_dir)),
            spawn=[*np.round(spawn_xy, 4).tolist(), round(spawn_yaw, 5)],
            faucet=dict(world=[*np.round(f_xy, 4).tolist(), round(f_yaw, 5)],
                        rel=[*np.round(frel, 4).tolist(), round(math.degrees(f_yaw - spawn_yaw), 2)]),
            reach_proxy=reach, walk=round(float(np.linalg.norm(rel[reach] - rel[0])), 4),
            path=[[int(i), round(float(rel[i, 0]), 4), round(float(rel[i, 1]), 4),
                   round(math.degrees(yaw[i] - spawn_yaw), 2)] for i in idx],
            _faucet_path=in_faucet[:reach + 1]))
    return out


# ---------------------------------------------------------------------------------------------- choice
def choose(eps: list[dict], k: int, min_walk: float, given: list[int]) -> list[dict]:
    """k slots: given episodes, or the k most spread out start-to-reach paths (faucet frame); each with backups.

    The 10% most isolated paths (largest mean distance to their 5 nearest) are never picked: spreading the picks out
    would otherwise favour one-off demos (a 30 cm sidestep away from the faucet) over the paths the data is made of.
    """
    feats_all = np.stack([resample(e["_faucet_path"]) for e in eps])  # (n, 20, 2)
    D_all = np.linalg.norm(feats_all[:, None] - feats_all[None], axis=3).mean(axis=2)
    isolation = np.sort(D_all, axis=1)[:, 1:6].mean(axis=1)
    typical = isolation <= np.quantile(isolation, 0.9)
    keep = [i for i, e in enumerate(eps) if e["walk"] >= min_walk and (typical[i] or e["episode"] in given)]
    ok = [eps[i] for i in keep]
    if len(ok) < k:
        raise SystemExit(f"only {len(ok)} typical episodes walk at least {min_walk} m; lower --min-walk")
    D = D_all[np.ix_(keep, keep)]
    by_ep = {e["episode"]: i for i, e in enumerate(ok)}
    if given:
        missing = [g for g in given if g not in by_ep]
        if missing:
            raise SystemExit(f"--episodes {missing}: not a training episode, or a walk shorter than --min-walk")
        picked = [by_ep[g] for g in given]
    else:
        picked = [int(D.sum(axis=1).argmin())]
        while len(picked) < k:
            picked.append(int(np.argmax(np.where(np.isin(np.arange(len(ok)), picked), -1, D[:, picked].min(axis=1)))))
    slots = []
    for s, i in enumerate(picked):
        near = [int(j) for j in np.argsort(D[i]) if j not in picked][:6]
        slots.append(dict(slot=s, episode=ok[i]["episode"], backups=[ok[j]["episode"] for j in near],
                          walk=ok[i]["walk"], faucet=ok[i]["faucet"]["rel"],
                          min_dist_to_others=round(float(min(D[i, j] for j in picked if j != i)), 4) if k > 1 else None))
    return slots


# ---------------------------------------------------------------------------------------------- plan
def conditions(radii: list[float], yaws: list[float]) -> list[dict]:
    """Start conditions: the training start, moves (8 directions per radius), turns (left and right per angle)."""
    cs = [dict(cond="start:0", group="training start", name="training start", dx=0.0, dy=0.0, dyaw=0.0)]
    for r in radii:
        for i, d in enumerate(DIRS):
            a = math.radians(45 * i)  # 0 = ahead, 90 = to the robot's left
            cs.append(dict(cond=f"start:m{round(r * 100)}:{d}", group=f"moved {r * 100:g} cm",
                           name=f"{r * 100:g} cm {d}", dx=round(r * math.cos(a), 4), dy=round(r * math.sin(a), 4),
                           dyaw=0.0, dir=d, r=r))
    for y in yaws:
        for s in (1, -1):
            cs.append(dict(cond=f"start:t{s * y:+g}", group=f"turned {y:g}°",
                           name=f"turned {y:g}° {'left' if s > 0 else 'right'}", dx=0.0, dy=0.0, dyaw=float(s * y)))
    return cs


POINTS = [("p1", "1/3 of the way"), ("p2", "2/3 of the way"), ("p3", "reach point")]


def build_trials(conds: list[dict], n_slots: int, repeats: int, seed: int) -> list[dict]:
    first = {"start:0": None, f"start:m{round(max(c.get('r', 0) for c in conds) * 100)}:forward": None}
    turns = [c["cond"] for c in conds if c["dyaw"] > 0]
    if turns:
        first[turns[-1]] = None
    passes = []
    for rep in range(repeats):
        items = []
        for ci, c in enumerate(conds):
            items.append(dict(kind="start", cond=c["cond"], slot=(ci + rep) % n_slots, rep=rep))
        for s in range(n_slots):
            for pt, _ in POINTS:
                items.append(dict(kind="path", cond=f"path:{pt}", point=pt, slot=s, rep=rep))
        order = np.random.default_rng([seed, 17, rep]).permutation(len(items))
        items = [items[i] for i in order]
        if rep == 0:  # one of each kind first, so a smoke test of a few trials sees every code path
            head = []
            for want in [*first, "path:p1", "path:p3"]:
                j = next((j for j, it in enumerate(items) if it["cond"] == want and it not in head), None)
                if j is not None:
                    head.append(items[j])
            items = head + [it for it in items if it not in head]
        passes += items
    by_cond = {c["cond"]: c for c in conds}
    trials = []
    for i, it in enumerate(passes):
        t = dict(trial=i, label=f"t{i:03d}", **it)
        if it["kind"] == "start":
            c = by_cond[it["cond"]]
            t.update(dx=c["dx"], dy=c["dy"], dyaw=c["dyaw"], group=c["group"], name=c["name"])
        else:
            t.update(group=dict(POINTS)[it["point"]], name=dict(POINTS)[it["point"]])
        trials.append(t)
    return trials


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_dir = resolve_run_dir(args)
    train_dir = ensure_train_pack(args)
    raw_dir = ensure_raw_pack(args)
    eps = training_paths(train_dir, raw_dir)
    given = [int(x) for x in args.episodes.split(",") if x.strip()]
    k = len(given) or args.n_episodes
    slots = choose(eps, k, args.min_walk, given)
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    yaws = [float(x) for x in args.yaws.split(",") if x.strip()]
    conds = conditions(radii, yaws)

    settings = dict(task=args.task, run_dir=str(run_dir), ckpt_step=args.ckpt_step, instruction=args.instruction,
                    radii=radii, yaws=yaws, repeats=args.repeats, reach_lead=args.reach_lead,
                    min_walk=args.min_walk, n_episodes=k, episodes_given=given, seed=args.seed,
                    train_dir=str(train_dir), raw_dir=str(raw_dir))
    plan_path = out / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        diff = {k_: (old["settings"].get(k_), v) for k_, v in settings.items() if old["settings"].get(k_) != v}
        if diff:
            raise SystemExit(f"{out} holds a sweep with other settings: "
                             + "; ".join(f"{k_}: {a} -> {b}" for k_, (a, b) in diff.items())
                             + "\nDelete it or set OUT to another directory.")
    trials = build_trials(conds, k, args.repeats, args.seed)
    n_run = min(args.trials, len(trials)) if args.trials > 0 else len(trials)
    plan_path.write_text(json.dumps(dict(settings=settings, slots=slots, conditions=conds,
                                         points=[dict(point=p, name=n) for p, n in POINTS], trials=trials), indent=1))
    # the table (fixed in the world) and every training episode, for the map
    conf0 = json.loads(read_episodes(train_dir / "meta")[0]["environment_config"])
    table = conf0.get("layout", {}).get("actors", {}).get("table")
    (out / "train_paths.json").write_text(json.dumps(dict(
        table=None if table is None else dict(position=table["pose"]["position"][:2],
                                              yaw=yaw_of(table["pose"]["quaternion"]), size=table["size"][:2]),
        episodes=[{k_: v for k_, v in e.items() if not k_.startswith("_")} for e in eps])))
    (out / "run_dir.txt").write_text(str(run_dir) + "\n")
    (out / "n_trials.txt").write_text(f"{n_run}\n")

    n_start = sum(t["kind"] == "start" for t in trials)
    print(f"[prepare] {args.task}: {len(eps)} training episodes; prompt {args.instruction!r}")
    for s in slots:
        print(f"[prepare]   slot {s['slot']}: episode {s['episode']:3d}  walk {100 * s['walk']:4.1f} cm  faucet "
              f"{100 * s['faucet'][0]:.0f} cm ahead {100 * s['faucet'][1]:+.0f} cm left  nearest other pick "
              f"{100 * (s['min_dist_to_others'] or 0):.1f} cm   backups {s['backups']}")
    print(f"[prepare]   {len(conds)} start conditions x {args.repeats} = {n_start} start trials; {k} episodes x "
          f"{len(POINTS)} path points x {args.repeats} = {len(trials) - n_start} path trials; {len(trials)} in all"
          + (f" (running the first {n_run})" if n_run < len(trials) else ""))
    print(f"[prepare]   checkpoint: {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
