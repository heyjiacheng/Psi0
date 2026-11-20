"""Stage 0a of the tabletop edge sweep: the checkpoint and the scenes the trials start from. Psi0 venv, CPU only.

Fetches like the target-pose sweep (../target_pose_sweep/prepare_target_sweep.py, imported, not changed), downloading
what is missing: the task's released checkpoint (or --run-dir), its training episodes' meta and SIMPLE's eval pack at
--dr-level (the base scenes: the robot spawn). Writes into --out:

  run_dir.txt          the checkpoint run dir
  base_scenes.txt      the eval pack dir
  train_pack.txt       the training pack dir (meta/ and data/: demo_reach.py replays its joints)
  train_geometry.json  every training episode's robot spawn, target pose and visible table edge, in the robot's start
                       frame, and their ranges

The released TabletopGraspMP checkpoint carries the older name of its training data (G1WholebodyTabletopGrasp-v0).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "target_pose_sweep"))
import prepare_target_sweep as P  # noqa: E402

CKPT_ALIASES = {"G1WholebodyTabletopGraspMP-v0": "G1WholebodyTabletopGrasp-v0"}


def ensure_train_data(args, train_dir: Path) -> None:
    """The training pack's data/ (the joint states demo_reach.py replays; ~10 MB of parquet), next to its meta/."""
    if any((train_dir / "data").glob("chunk-*/episode_*.parquet")):
        return
    import shutil
    import zipfile
    stage = Path(args.simple_dir) / "data" / "psi-data" / ".download"
    zp = P.fetch_zip(f"simple/{args.task}.zip", stage)
    with zipfile.ZipFile(zp) as z:
        for m in z.namelist():
            if "/data/" in m and m.endswith(".parquet"):
                dst = train_dir / "data" / m.split("/data/", 1)[1]
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
    shutil.rmtree(stage, ignore_errors=True)


def counter_edge_x(conf: dict) -> float:
    """World x of the visible table's near edge: the room's own counter (the scene's "surface", which SIMPLE moves
    so that its top centre lands on center_offset), table_size[0] long along x (room yaw 90)."""
    sc = conf["dr_state_dict"]["scene"]
    return float(sc["center_offset"][0]) - 0.5 * float(sc["conf"]["table_size"][0])


def geometry(conf: dict) -> dict:
    ds = conf["dr_state_dict"]
    sp = ds["spatial"]
    r = sp[conf["robot_cfg"]["uid"]]
    t = sp[str(ds["target"]["uid"])]
    ryaw = P.yaw_of(r["quaternion"])
    rel = P.to_robot_frame(r["position"][:2], ryaw, t["position"][:2])
    tab = conf["layout"]["actors"]["table"] if "layout" in conf else ds["scene"]["table"]
    box_edge = tab["pose"]["position"][0] - 0.5 * tab["size"][0]
    edge = counter_edge_x(conf)
    return dict(robot=[r["position"][0], r["position"][1], math.degrees(ryaw)],
                target=[float(rel[0]), float(rel[1]), math.degrees(P.yaw_of(t["quaternion"]) - ryaw)],
                edge_from_robot=edge - r["position"][0], target_from_edge=t["position"][0] - edge,
                box_edge_from_robot=box_edge - r["position"][0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--psi0-dir", required=True)
    p.add_argument("--simple-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-dir", default="")
    p.add_argument("--ckpt-step", default="40000")
    p.add_argument("--dr-level", type=int, default=2)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    P.CKPT_ALIASES = {**P.CKPT_ALIASES, **CKPT_ALIASES}  # this process only
    run_dir = P.resolve_run_dir(args)
    train_dir = P.ensure_train_meta(args)
    ensure_train_data(args, train_dir)
    eval_dir = P.ensure_eval_pack(args)
    base = P.load_confs(eval_dir / "meta")
    train = P.load_confs(train_dir / "meta")
    rooms = {(c["dr_state_dict"]["scene"]["uid"], tuple(c["dr_state_dict"]["scene"]["center_offset"][:2]),
              c["dr_state_dict"]["scene"]["center_orientation"][2]) for c in base + train}
    if len(rooms) != 1:
        raise SystemExit(f"the scenes put their room in {len(rooms)} different ways ({sorted(rooms)}); this sweep "
                         "needs one")
    rows = [geometry(c) for c in train]
    rng = lambda k, i=None: [min(r[k] if i is None else r[k][i] for r in rows),
                             max(r[k] if i is None else r[k][i] for r in rows)]
    geo = dict(n=len(rows), episodes=rows,
               range=dict(target_dx=rng("target", 0), target_dy=rng("target", 1), target_dyaw=rng("target", 2),
                          edge_from_robot=rng("edge_from_robot"), target_from_edge=rng("target_from_edge")),
               target=train[0]["dr_state_dict"]["target"].get("label"),
               instruction=train[0]["dr_state_dict"].get("language"),
               table_size=train[0]["dr_state_dict"]["scene"]["conf"]["table_size"])
    (out / "run_dir.txt").write_text(f"{run_dir}\n")
    (out / "base_scenes.txt").write_text(f"{eval_dir}\n")
    (out / "train_pack.txt").write_text(f"{train_dir.resolve()}\n")
    (out / "train_geometry.json").write_text(json.dumps(geo, indent=1))
    g = geo["range"]
    print(f"[prepare] {args.task}: checkpoint {run_dir}")
    print(f"[prepare]   base scenes: {eval_dir} ({len(base)}), room {rooms.pop()}")
    print(f"[prepare]   training ({len(rows)} episodes, robot frame): {geo['target']} dx {g['target_dx'][0]:+.3f}.."
          f"{g['target_dx'][1]:+.3f} m, dy {g['target_dy'][0]:+.3f}..{g['target_dy'][1]:+.3f} m; visible table edge "
          f"{g['edge_from_robot'][0]:.3f}..{g['edge_from_robot'][1]:.3f} m ahead of the robot, target "
          f"{g['target_from_edge'][0]:.3f}..{g['target_from_edge'][1]:.3f} m behind it")
    a = np.array([[r["edge_from_robot"], r["target"][0]] for r in rows])
    print(f"[prepare]   corr(edge distance, target distance) over the demos: {np.corrcoef(a.T)[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
