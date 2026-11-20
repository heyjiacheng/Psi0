"""Stage 0c of the scene sweep: take c0_baseline's trials from an earlier sweep instead of running them again.

make_scene_trials.py is deterministic, so with the same task, base scenes, seed and checkpoint trial i of c0_baseline
is the same scene in every sweep (same seed, base episode and faucet pose; checked here for every trial before
anything is written, and nothing is imported on a mismatch). The policy settings that change what the policy does
(action horizon, RTC, step budget, instruction) must match too; a sweep that did not record one ran the default.
Reads the earlier sweep's report folder (results.json, index.html for the hands' paths, videos/) and writes into --out:

  results.jsonl     one line per imported trial, renumbered to this plan, with imported_from (the eval then skips
                    them; the report counts them as from an earlier run and keeps their clips)
  report/videos/    their clips, as they are

    python import_baseline.py --out OUT --source EARLIER_OUT/report
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

POLICY = {"action_exec_horizon": "24", "rtc": "1", "budget_steps": "0", "instruction": ""}  # the runner's defaults


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--source", required=True, help="the earlier sweep's report folder")
    args = p.parse_args()
    out, src = Path(args.out), Path(args.source)
    plan = json.loads((out / "plan.json").read_text())
    st = plan["settings"]
    old = json.loads((src / "results.json").read_text())
    ost = old["settings"]
    for key in ("task", "run_dir", "ckpt_step", "seed", "base_scenes"):
        if str(ost.get(key)) != str(st.get(key)):
            raise SystemExit(f"{src}: {key} is {ost.get(key)!r}, this sweep has {st.get(key)!r}; not importing")
    for key, default in POLICY.items():
        if str(ost.get(key, default)) != str(st.get(key, default)):
            raise SystemExit(f"{src}: {key} is {ost.get(key, default)!r}, this sweep has {st.get(key, default)!r}; "
                             "not importing")
    paths = {}  # earlier trial number -> the hands' paths its report embedded
    m = re.search(r'<script id="sweep-data" type="application/json">(.*?)</script>',
                  (src / "index.html").read_text(), re.S) if (src / "index.html").exists() else None
    if m:
        for t in json.loads(m.group(1))["trials"]:
            if t.get("path_r") or t.get("path_l"):
                paths[t["trial"]] = dict(r=t.get("path_r"), l=t.get("path_l"))
    earlier = {r["index"]: r for r in old["trials"] if r.get("set") == "c0_baseline"}
    mine = {t["index"]: t for t in plan["trials"] if t["set"] == "c0_baseline"}
    results = out / "results.jsonl"
    have = {json.loads(line)["trial"] for line in results.read_text().splitlines() if line.strip()} \
        if results.exists() else set()
    todo = [(t, earlier[i]) for i, t in sorted(mine.items()) if i in earlier and t["trial"] not in have]
    for t, r in todo:  # every scene is checked before anything is written
        if (r["seed"], r["base_episode"]) != (t["seed"], t["base_episode"]) or \
                r["plan"] != dict(dx=t["dx"], dy=t["dy"], dyaw=t["dyaw"]):
            raise SystemExit(f"trial index {t['index']}: the earlier c0 scene differs (seed {r['seed']} vs {t['seed']}, "
                             f"base {r['base_episode']} vs {t['base_episode']}, faucet {r['plan']} vs "
                             f"{dict(dx=t['dx'], dy=t['dy'], dyaw=t['dyaw'])}); not importing")
    (out / "report" / "videos").mkdir(parents=True, exist_ok=True)
    lines = []
    for t, r in todo:
        label = f"t{t['trial']:03d}"
        video = src / "videos" / f"{r['label']}.mp4"
        if video.exists():
            shutil.copy2(video, out / "report" / "videos" / f"{label}.mp4")
        rec = dict(r, trial=t["trial"], label=label, video_dir=None,
                   imported_from=dict(report=str(src.resolve()), trial=r["trial"], label=r["label"]))
        if r["trial"] in paths:
            rec["hand_paths"] = paths[r["trial"]]
        lines.append(json.dumps(rec))
    if lines:
        with open(results, "a") as f:
            f.write("\n".join(lines) + "\n")
    done = have | {t["trial"] for t, _ in todo}
    print(f"[baseline] {len(lines)} c0_baseline trials from {src}; "
          f"{sum(t['trial'] not in done for t in mine.values())} of {len(mine)} left to run")


if __name__ == "__main__":
    main()
