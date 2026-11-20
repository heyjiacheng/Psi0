"""Stage 0b of the tabletop edge sweep: every trial's scene. SIMPLE's venv, CPU only.

Trial i of every set starts from one scene: a base scene of the task's eval pack (the robot spawn) randomized the way
SIMPLE's highest replay level does it (DRManager.load_state_dict dr_level=2: distractors, materials, lighting and the
poses of the target and the distractors, within the task's ranges), seeded per trial, as the scene sweep does
(../scene_sweep/make_scene_trials.py). Both sets draw the table the way the training images show it: the room's own
counter, not SIMPLE's randomly textured slab (which stays, invisible, as the robot's and the objects' table in both
simulators). Each set changes one thing:

  c0_baseline   nothing
  e1_edge_x     the counter's near edge drawn FARTHER farther (into the table, the counter shorter) to CLOSER closer
                (toward the robot, the counter longer), spread over the trials (Latin hypercube); its far end, the
                target, the distractors and the robot stay, the texture keeps its scale, the physics is unchanged
                (the robot and the objects collide only with the recorded table)

Writes into --out:

  scenes/<set>/<i>.json   the scene (a SIMPLE environment_config; its "edge_sweep" entry says what was changed)
  plan.json               the trials eval_edge_sweep.py runs, ordered trial index by trial index (every set's trial 0,
                          then 1, ...) so a partial run covers every set alike
  n_trials.txt            how many trials

    python make_edge_trials.py --out OUT --simple-dir SIMPLE --base-scenes DIR --trials 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np

SETS = {
    "c0_baseline": "no change",
    "e1_edge_x": "table's near edge drawn closer to or farther from the robot; target, robot and physics unchanged",
}
OBJECT_MARGIN = 0.02  # m, the target's footprint kept this far inside the drawn edge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--simple-dir", required=True)
    p.add_argument("--base-scenes", required=True)
    p.add_argument("--trials", type=int, default=50, help="trials per set")
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--closer", type=float, default=0.10, help="m, the largest shift toward the robot")
    p.add_argument("--farther", type=float, default=0.15, help="m, the largest shift away from the robot")
    p.add_argument("--settings", default="{}", help="JSON of the run's settings, kept in plan.json")
    return p.parse_args()


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_robot_frame(robot_xy, robot_yaw, xy) -> list[float]:
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    dx, dy = xy[0] - robot_xy[0], xy[1] - robot_xy[1]
    return [c * dx + s * dy, -s * dx + c * dy]


def lhs(rng: np.random.Generator, n: int) -> np.ndarray:
    """n values in [0, 1), one in each of n equal bins, in random order."""
    return (rng.permutation(n) + rng.random(n)) / n


def counter(conf: dict) -> dict:
    """The visible table (the room's counter) in world x/y: SIMPLE moves the room so that the counter's top centre
    lands on center_offset; table_size is its extent (room yaw 90: length along x)."""
    sc = conf["dr_state_dict"]["scene"]
    (cx, cy), (lx, ly) = sc["center_offset"][:2], sc["conf"]["table_size"][:2]
    return dict(x=[cx - lx / 2, cx + lx / 2], y=[cy - ly / 2, cy + ly / 2])


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    import simple.tasks  # noqa: F401  registers the tasks
    from simple.tasks.registry import TaskRegistry
    from simple.utils import NumpyArrayEncoder

    sets = [s for s in args.sets.split(",") if s]
    unknown = set(sets) - set(SETS)
    if unknown:
        raise SystemExit(f"unknown sets {sorted(unknown)}; known: {', '.join(SETS)}")
    base = [json.loads(json.loads(line)["environment_config"])
            for line in (Path(args.base_scenes) / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if {b["dr_state_dict"]["scene"]["center_orientation"][2] for b in base} != {90.0}:
        raise SystemExit("the base scenes must put the room at yaw 90 (the counter along the robot's heading)")
    task = TaskRegistry.make(base[0]["uid"], render_hz=50)

    rng = np.random.default_rng(args.seed)
    base_order = rng.permutation(len(base))
    seeds = [int(s) for s in np.random.SeedSequence(args.seed).generate_state(args.trials)]
    frac = lhs(np.random.default_rng([args.seed, 11]), args.trials)
    shifts = -args.farther + frac * (args.farther + args.closer)  # + = closer to the robot

    def scene0(i: int) -> dict:
        random.seed(seeds[i])
        np.random.seed(seeds[i] % 2**32)
        task.reset(options={"state_dict": base[base_order[i % len(base)]], "dr_level": 2})
        return json.loads(json.dumps(task.state_dict(), cls=NumpyArrayEncoder))

    (out / "scenes").mkdir(parents=True, exist_ok=True)
    trials, warnings = [], 0
    for i in range(args.trials):
        c0 = scene0(i)
        ds = c0["dr_state_dict"]
        robot = ds["spatial"][c0["robot_cfg"]["uid"]]
        tuid = str(ds["target"]["uid"])
        target = ds["spatial"][tuid]
        ryaw = yaw_of(robot["quaternion"])
        rel = to_robot_frame(robot["position"][:2], ryaw, target["position"][:2])
        top = counter(c0)
        radius = float(target["position"][2])  # the can lies on its side: its centre is its radius above the top
        for s in sets:
            conf = deepcopy(c0)
            closer = 0.0 if s == "c0_baseline" else float(shifts[i])
            edge = top["x"][0] - closer
            margin = target["position"][0] - radius - edge
            if margin < OBJECT_MARGIN:
                warnings += 1
                print(f"[trials] WARNING trial {i} {s}: the target is {margin * 100:.1f} cm inside the drawn edge",
                      flush=True)
            conf["edge_sweep"] = dict(set=s, edge_closer=round(closer, 4))
            path = out / "scenes" / s / f"{i}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(conf))
            trials.append(dict(
                set=s, index=i, seed=seeds[i], base_episode=int(base_order[i % len(base)]),
                conf=str(path.relative_to(out)), value=dict(edge_closer=round(closer, 4)),
                geom=dict(robot=[round(v, 4) for v in [*robot["position"][:2], math.degrees(ryaw)]],
                          target=[round(v, 4) for v in [*rel, math.degrees(yaw_of(target["quaternion"]) - ryaw)]],
                          target_world=[round(v, 4) for v in target["position"]],
                          counter=dict(x=[round(edge, 4), round(top["x"][1], 4)], y=[round(v, 4) for v in top["y"]]),
                          edge_from_robot=round(edge - robot["position"][0], 4),
                          target_from_edge=round(target["position"][0] - edge, 4),
                          distractors=[[round(v, 3) for v in ds["spatial"][k]["position"][:2]]
                                       for k in ds.get("distractors", {}) if k in ds["spatial"]])))
    order = {s: k for k, s in enumerate(sets)}
    trials.sort(key=lambda t: (t["index"], order[t["set"]]))
    for k, t in enumerate(trials):
        t["trial"] = k
    settings = json.loads(args.settings)
    settings.update(task=args.task, base_scenes=str(Path(args.base_scenes).resolve()), sets=sets, trials=args.trials,
                    seed=args.seed, closer=args.closer, farther=args.farther, set_titles={s: SETS[s] for s in sets},
                    target=ds["target"].get("label"), instruction=ds.get("language"))
    (out / "plan.json").write_text(json.dumps(dict(settings=settings, trials=trials), indent=1))
    (out / "n_trials.txt").write_text(f"{len(trials)}\n")
    e = [t["value"]["edge_closer"] for t in trials if t["set"] == "e1_edge_x"]
    print(f"[trials] {len(trials)} trials ({', '.join(f'{s} {args.trials}' for s in sets)}), seed {args.seed}"
          + (f"; e1 edge shifts {min(e) * 100:+.1f}..{max(e) * 100:+.1f} cm (+ = closer)" if e else "")
          + (f"; {warnings} WARNING(s)" if warnings else ""), flush=True)


if __name__ == "__main__":
    main()
