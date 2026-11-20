"""Where the training demos' right hand goes against the can: MuJoCo forward kinematics of the recorded joints.
SIMPLE's venv, from the SIMPLE root, CPU only (seconds).

For every training episode the robot is put where the eval's stand stabilization leaves its pelvis (PELVIS, in the
robot's start frame, upright, at MuJoCo's height) with the recorded legs, waist, arms and right hand (the pack's
observation.leg_joints / arm_joints / hand_joints); the right fingers' centre (index_1, middle_1, thumb_2 links, as in
eval_edge_sweep.py) is then read at fixed control steps. Writes --out (JSON): per step the demos' mean hand position
and how it moves with the can (slope of hand x on can x, hand y on can y, bootstrap 95% intervals).

    python demo_reach.py --pack data/psi-data/simple/<TASK> --out OUT/demo_reach.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

import simple.robots.g1_wholebody as G

PELVIS = (-0.038, 0.013, -0.011)  # m, the eval's pelvis after the 60 stand steps (robot start frame x, y; world z)
STEPS = (1, 21, 31, 41, 51, 61, 71, 81)
FINGERS = ("right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link")


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def slope(x, y, boot=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(0)
    bs = [np.polyfit(x[j], y[j], 1)[0] for j in (rng.integers(0, len(x), len(x)) for _ in range(boot)) if np.ptp(x[j]) > 0]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(slope=round(b, 3), ci=[round(float(lo), 3), round(float(hi), 3)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", required=True, help="the training pack (meta/ and data/)")
    p.add_argument("--mjcf", default="data/robots/g1/g1_29dof_wholebody_dex3.xml")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    pack = Path(args.pack)
    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)
    root = m.jnt_qposadr[m.joint("pelvis").id]
    qadr = lambda name: m.jnt_qposadr[m.joint(name).id]
    fid = [m.body(b).id for b in FINGERS]
    lines = [json.loads(l) for l in (pack / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for e in lines:
        conf = json.loads(e["environment_config"])
        t = pq.read_table(pack / f"data/chunk-000/episode_{e['episode_index']:06d}.parquet")
        arms = np.array(t.column("observation.arm_joints").to_pylist())
        hands = np.array(t.column("observation.hand_joints").to_pylist())
        legs = np.array(t.column("observation.leg_joints").to_pylist())
        sp = conf["dr_state_dict"]["spatial"]
        r, c = sp[conf["robot_cfg"]["uid"]], sp[str(conf["dr_state_dict"]["target"]["uid"])]
        ryaw = yaw_of(r["quaternion"])
        cs, sn = math.cos(ryaw), math.sin(ryaw)
        rel = lambda q: [cs * (q[0] - r["position"][0]) + sn * (q[1] - r["position"][1]),
                         -sn * (q[0] - r["position"][0]) + cs * (q[1] - r["position"][1])]
        hand = {}
        for s in STEPS:
            if s >= len(arms):
                continue
            d.qpos[:] = 0
            d.qpos[root:root + 3] = [r["position"][0] + cs * PELVIS[0] - sn * PELVIS[1],
                                     r["position"][1] + sn * PELVIS[0] + cs * PELVIS[1], PELVIS[2]]
            d.qpos[root + 3:root + 7] = [math.cos(ryaw / 2), 0, 0, math.sin(ryaw / 2)]
            for names, vals in ((G.LEFT_LEG_JOINTS + G.RIGHT_LEFT_JOINTS, legs[s][:12]), (G.WAIST_JOINTS, legs[s][12:15]),
                                (G.LEFT_ARM_JOINTS, arms[s][:7]), (G.RIGHT_ARM_JOINTS, arms[s][7:14]),
                                (G.RIGHT_HAND_JOINTS, hands[s][7:14])):
                for n, v in zip(names, vals):
                    d.qpos[qadr(n)] = v
            mujoco.mj_kinematics(m, d)
            hand[s] = rel(d.xpos[fid].mean(axis=0))
        rows.append(dict(episode=e["episode_index"], can=rel(c["position"]), hand=hand))
    res = dict(n=len(rows), pelvis=PELVIS, can_x=[min(r["can"][0] for r in rows), max(r["can"][0] for r in rows)],
               steps={})
    for s in STEPS:
        sel = [r for r in rows if s in r["hand"]]
        hx = np.array([r["hand"][s][0] for r in sel])
        hy = np.array([r["hand"][s][1] for r in sel])
        cx = np.array([r["can"][0] for r in sel])
        cy = np.array([r["can"][1] for r in sel])
        res["steps"][str(s)] = dict(n=len(sel), hand_x=round(float(hx.mean()), 4), hand_x_sd=round(float(hx.std()), 4),
                                    hand_minus_can_x=round(float((hx - cx).mean()), 4),
                                    x_on_can_x=slope(cx, hx), y_on_can_y=slope(cy, hy))
    Path(args.out).write_text(json.dumps(res, indent=1))
    for s, v in res["steps"].items():
        print(f"[demo_reach] step {s:>3}: hand x {v['hand_x'] * 100:.1f} cm, minus can {v['hand_minus_can_x'] * 100:+.1f}; "
              f"follows can x {v['x_on_can_x']['slope']:+.2f} {v['x_on_can_x']['ci']}, can y {v['y_on_can_y']['slope']:+.2f}")


if __name__ == "__main__":
    main()
