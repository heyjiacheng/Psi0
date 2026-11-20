"""Stage 3 of the tabletop edge sweep: the report. Psi0 venv (numpy), CPU only.

Reads OUT/results.jsonl and OUT/plan.json and writes OUT/report/: index.html (self-contained but for the videos),
results.json (every trial's record and the summary), videos/<label>.mp4 (the head camera, re-encoded small) and
frames/<label>.jpg (the policy's first image).

    python plot_edge_sweep.py --results OUT
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BINS = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10]  # m, edge_closer bins of e1 (last bin closed)
HAND_STEP = 61  # control step (1.2 s) at which the right hand's position is read: the demos then follow the can 0.76


def hand_at(trace: dict, step: int) -> list | None:
    """The right fingers' position (robot start frame x, y) at a control step, from the 10 Hz trace."""
    rows = np.array(trace["rows"], dtype=float)
    if rows.size == 0:
        return None
    k = int(np.argmin(np.abs(rows[:, 0] - step)))
    if abs(rows[k, 0] - step) > 3:
        return None
    c = {n: i for i, n in enumerate(trace["columns"])}
    return [float(rows[k, c["rh_x"]]), float(rows[k, c["rh_y"]])]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def slope(x: np.ndarray, y: np.ndarray, boot: int = 2000, seed: int = 0) -> dict | None:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.ptp(x) == 0:
        return None
    b = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0:
            bs.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(slope=round(b, 3), ci=[round(float(lo), 3), round(float(hi), 3)], n=int(len(x)))


def web_video(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return True
    tmp = dst.with_name(dst.stem + ".part.mp4")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "medium", "-crf", "30",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    if shutil.which("ffmpeg") and subprocess.run(cmd).returncode == 0:
        tmp.replace(dst)
        return True
    tmp.unlink(missing_ok=True)
    shutil.copy(src, dst)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, help="the sweep directory (OUT)")
    args = p.parse_args()
    out = Path(args.results).resolve()
    rep = out / "report"
    (rep / "videos").mkdir(parents=True, exist_ok=True)
    (rep / "frames").mkdir(parents=True, exist_ok=True)
    plan = json.loads((out / "plan.json").read_text()) if (out / "plan.json").exists() else None
    old = json.loads((rep / "results.json").read_text()) if (rep / "results.json").exists() else None
    settings = (plan or old)["settings"]
    n_planned = len(plan["trials"]) if plan else old.get("n_planned")
    recs = {}
    if (out / "results.jsonl").exists():
        for line in (out / "results.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["trial"]] = r
    elif old:
        recs = {r["trial"]: r for r in old["trials"]}
    trials = [recs[k] for k in sorted(recs)]

    (rep / "traces").mkdir(exist_ok=True)
    for r in trials:  # media: the head camera, small, and the first frame; the 10 Hz trace
        lab = r["label"]
        src = sorted((out / "videos" / lab).glob("head_stereo_left*.mp4"))  # SIMPLE adds _success / _failed
        if src:
            web_video(src[0], rep / "videos" / f"{lab}.mp4")
        tr = out / "traces" / f"{lab}.json"
        if tr.exists():
            shutil.copy(tr, rep / "traces" / f"{lab}.json")
        tr = rep / "traces" / f"{lab}.json"
        h = hand_at(json.loads(tr.read_text()), HAND_STEP) if tr.exists() else None
        can = (r.get("placed") or {}).get("target_rel")
        r["hand"] = None if h is None or can is None else dict(
            x=round(h[0], 4), y=round(h[1], 4), minus_can_x=round(h[0] - can[0], 4), minus_can_y=round(h[1] - can[1], 4))
        fr = out / "frames" / f"{lab}.png"
        dst = rep / "frames" / f"{lab}.jpg"
        if fr.exists() and not dst.exists():
            try:
                from PIL import Image
                Image.open(fr).convert("RGB").resize((320, 180)).save(dst, quality=85)
            except Exception:
                shutil.copy(fr, dst.with_suffix(".png"))
        r["video"] = f"videos/{lab}.mp4" if (rep / "videos" / f"{lab}.mp4").exists() else None
        r["frame"] = f"frames/{lab}.jpg" if dst.exists() else None
        rm = r.get("reach") or {}
        r["reach_err"] = rm.get("minus_target")

    sets = settings["sets"]
    by = {s: [r for r in trials if r["set"] == s] for s in sets}
    summary = {}
    for s in sets:
        k, n = sum(r["success"] for r in by[s]), len(by[s])
        summary[s] = dict(k=k, n=n, rate=(k / n if n else None), ci=wilson(k, n),
                          outcomes={o: sum(r["outcome"] == o for r in by[s]) for o in sorted({r["outcome"] for r in by[s]})})
    pairs = None
    if "c0_baseline" in by and "e1_edge_x" in by:
        c0 = {r["index"]: r for r in by["c0_baseline"]}
        pr = [(c0[r["index"]], r) for r in by["e1_edge_x"] if r["index"] in c0]
        b = sum(a["success"] and not e["success"] for a, e in pr)
        c = sum(e["success"] and not a["success"] for a, e in pr)
        pairs = dict(n=len(pr), both=sum(a["success"] and e["success"] for a, e in pr), only_c0=b, only_e1=c,
                     neither=sum(not a["success"] and not e["success"] for a, e in pr), p=mcnemar(b, c))
        bins = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            inb = [(a, e) for a, e in pr if lo <= e["value"]["edge_closer"] < hi or (hi == BINS[-1] and
                                                                                     e["value"]["edge_closer"] == hi)]
            bins.append(dict(lo=lo, hi=hi, n=len(inb), e1=sum(e["success"] for _, e in inb),
                             c0=sum(a["success"] for a, _ in inb)))
        pairs["bins"] = bins
        x = np.array([e["value"]["edge_closer"] for _, e in pr], dtype=float)
        dy = np.array([np.nan if e["reach_err"] is None or a["reach_err"] is None else e["reach_err"] - a["reach_err"]
                       for a, e in pr], dtype=float)
        ye = np.array([np.nan if e["reach_err"] is None else e["reach_err"] for _, e in pr], dtype=float)
        pairs["reach_slope"] = slope(x, ye)            # reach error per unit of edge_closer (following the edge: -1)
        pairs["reach_slope_paired"] = slope(x, dy)     # the same, minus the baseline trial's reach
        flip = lambda s: None if s is None else dict(slope=-s["slope"], ci=[-s["ci"][1], -s["ci"][0]], n=s["n"])
        pairs["reach_follow"] = flip(pairs["reach_slope"])  # cm the reach comes closer per cm the edge does: 1 edge, 0 can
        pairs["reach_follow_paired"] = flip(pairs["reach_slope_paired"])
        # the hand at HAND_STEP, same scene with and without the edge moved
        hd = np.array([np.nan if not (a.get("hand") and e.get("hand")) else e["hand"]["x"] - a["hand"]["x"]
                       for a, e in pr], dtype=float)
        pairs["hand_follow_edge"] = flip(slope(x, hd))
        pairs["hand_diff_abs_median"] = (round(float(np.nanmedian(np.abs(hd))), 4) if np.isfinite(hd).any() else None)
    hx = [(r["placed"]["target_rel"][0], r["hand"]["x"]) for r in trials if r.get("hand") and r.get("placed")]
    can_follow = slope(np.array([a for a, _ in hx]), np.array([b for _, b in hx])) if len(hx) >= 5 else None
    demo = None
    if (out / "demo_reach.json").exists():
        demo = json.loads((out / "demo_reach.json").read_text())
    elif old and old.get("demo"):
        demo = old["demo"]
    c0r = np.array([r["hand"]["minus_can_x"] for r in by.get("c0_baseline", []) if r.get("hand")], dtype=float)
    data = dict(settings=settings, n_planned=n_planned, n_done=len(trials), summary=summary, pairs=pairs,
                hand_step=HAND_STEP, can_follow=can_follow, demo=demo,
                demo_follow=(demo or {}).get("steps", {}).get(str(HAND_STEP), {}).get("x_on_can_x"),
                c0_reach=dict(median=float(np.median(c0r)) if c0r.size else None,
                              q=[float(v) for v in np.percentile(c0r, [10, 90])] if c0r.size else None),
                train=json.loads((out / "train_geometry.json").read_text())["range"]
                if (out / "train_geometry.json").exists() else (old or {}).get("train"),
                trials=trials)
    (rep / "results.json").write_text(json.dumps(data, indent=1))
    html = (HERE / "report_template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
    tmp = rep / "index.html.part"
    tmp.write_text(html)
    tmp.replace(rep / "index.html")
    line = ", ".join(f"{s} {summary[s]['k']}/{summary[s]['n']}" for s in sets)
    extra = ""
    if pairs:
        extra = f"; pairs {pairs['n']}: only c0 {pairs['only_c0']}, only e1 {pairs['only_e1']}, p {pairs['p']:.3f}"
        if pairs.get("hand_follow_edge"):
            rs = pairs["hand_follow_edge"]
            extra += f"; hand follows the edge {rs['slope']:+.2f} cm/cm [{rs['ci'][0]:+.2f}, {rs['ci'][1]:+.2f}]"
    if can_follow:
        extra += f"; follows the can {can_follow['slope']:+.2f} [{can_follow['ci'][0]:+.2f}, {can_follow['ci'][1]:+.2f}]"
        if data["demo_follow"]:
            extra += f" (demos {data['demo_follow']['slope']:+.2f})"
    print(f"[report] {len(trials)}/{n_planned} trials: {line}{extra} -> {rep / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
