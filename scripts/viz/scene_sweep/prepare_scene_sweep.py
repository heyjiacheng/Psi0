"""Stage 0a of the scene sweep: the checkpoint and the scenes the trials start from. Psi0 venv, CPU only.

Fetches like the target-pose sweep (../target_pose_sweep/prepare_target_sweep.py), downloading what is missing:
the task's released checkpoint (or --run-dir), its training episodes' meta (where the demos had the faucet) and
SIMPLE's eval pack at --dr-level (the base scenes: robot spawn, room, table). Writes into --out:

  run_dir.txt         the checkpoint run dir
  base_scenes.txt     the eval pack dir
  room.json           the base scenes' room and where SIMPLE puts it (for room_map.py)
  train_targets.json  every training episode's faucet pose relative to the robot, and their range
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "target_pose_sweep"))
import prepare_target_sweep as P  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="SIMPLE env id without 'simple/', e.g. G1WholebodyOpenFaucetTeleop-v0")
    p.add_argument("--psi0-dir", required=True)
    p.add_argument("--simple-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-dir", default="", help="a local checkpoint run dir; default: the task's released one")
    p.add_argument("--ckpt-step", default="40000")
    p.add_argument("--dr-level", type=int, default=2, help="which eval pack the base scenes come from")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_dir = P.resolve_run_dir(args)
    train_dir = P.ensure_train_meta(args)
    eval_dir = P.ensure_eval_pack(args)
    base = P.load_confs(eval_dir / "meta")
    rooms = {(c["dr_state_dict"]["scene"]["uid"], tuple(c["dr_state_dict"]["scene"]["center_offset"][:2]),
              c["dr_state_dict"]["scene"]["center_orientation"][2]) for c in base}
    if len(rooms) != 1:
        raise SystemExit(f"the base scenes put their room in {len(rooms)} different ways ({sorted(rooms)}); "
                         "the scene sweep needs one")
    uid, offset, yaw = rooms.pop()
    tt = P.train_targets(P.load_confs(train_dir / "meta"), "articulated")

    (out / "run_dir.txt").write_text(f"{run_dir}\n")
    (out / "base_scenes.txt").write_text(f"{eval_dir}\n")
    (out / "room.json").write_text(json.dumps(dict(room=uid.split(":")[-1], offset=list(offset), yaw=yaw)))
    (out / "train_targets.json").write_text(json.dumps(tt, indent=1))
    box = tt["box"]
    print(f"[prepare] {args.task}: checkpoint {run_dir}")
    print(f"[prepare]   base scenes: {eval_dir} ({len(base)}), room {uid} at offset {list(offset)}, yaw {yaw}")
    print(f"[prepare]   faucet training range (robot frame): dx {box['dx'][0]:+.3f}..{box['dx'][1]:+.3f} m, "
          f"dy {box['dy'][0]:+.3f}..{box['dy'][1]:+.3f} m, yaw {tt['yaw_deg'][0]:+.1f}..{tt['yaw_deg'][1]:+.1f} deg "
          f"({tt['n']} episodes)")


if __name__ == "__main__":
    main()
