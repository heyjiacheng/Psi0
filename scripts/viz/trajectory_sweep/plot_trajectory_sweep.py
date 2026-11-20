"""The trajectory sweep's report (last stage). Runs in Psi0's venv, CPU only.

Reads what prepare_trajectory_sweep.py and eval_trajectory_sweep.py wrote (plan.json, train_paths.json, calib.json,
results.jsonl, traces/, takeover/, videos/) and the train pack, and writes

report/ (self-contained: open report/index.html, or share the folder; kept when the rest of the sweep is deleted)
  index.html    report_template.html filled in: success rates; the top-down map in the robot's start frame (the 99
                training paths, the chosen episodes' replayed paths and faucets, every start condition and path point
                coloured by its success rate; click one to play its trials); success by condition; the condition list
  results.json  every number behind the page: settings, the chosen episodes and their path points, every trial (how
                close its takeover was to the training frame: joint error, image difference, pelvis offset)
  videos/       one head-camera clip per trial, re-encoded small, the replayed part labelled "replay"
  img/          per trial what the camera saw when the policy took over; the matching training frames

Safe to run while the sweep is still going (the runner does, every REPORT_EVERY s): a half-written last line of
results.jsonl is skipped, and index.html / results.json are replaced in one step.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TEMPLATE = Path(__file__).with_name("report_template.html")
IMG_W, IMG_H = 320, 180
FONT = next((f for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(f).is_file()), None)
GROUPS = ["training start", "moved 5 cm", "moved 10 cm", "turned 20°", "turned 40°",
          "1/3 of the way", "2/3 of the way", "reach point"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="the sweep's output directory")
    return p.parse_args()


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def read_jsonl(path: Path) -> list[dict]:
    """The finished trials; a line still being written (the sweep is running) is skipped."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def group_of(t: dict) -> str:
    if t["kind"] == "path":
        return {"p1": "1/3 of the way", "p2": "2/3 of the way", "p3": "reach point"}[t["point"]]
    return "training start" if t["cond"] == "start:0" else t["group"]


class TrainFrames:
    """Frames of the training videos (the policy's input size), decoded once per episode."""

    def __init__(self, train_dir: Path):
        self.dir = train_dir
        self.cache: dict[int, dict[int, np.ndarray]] = {}

    def get(self, ep: int, frames: set[int]) -> dict[int, np.ndarray]:
        have = self.cache.setdefault(ep, {})
        want = sorted(f for f in frames if f not in have and f >= 0)
        if want:
            vids = glob.glob(str(self.dir / "videos" / "chunk-*" / "*" / f"episode_{ep:06d}.mp4"))
            if vids:
                cap = cv2.VideoCapture(vids[0])
                i, wi = 0, 0
                while wi < len(want):
                    ok, fr = cap.read()
                    if not ok:
                        break
                    if i == want[wi]:
                        have[i] = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (IMG_W, IMG_H),
                                             interpolation=cv2.INTER_AREA)
                        wi += 1
                    i += 1
                cap.release()
        return {f: have[f] for f in frames if f in have}


def joint_err(state, train: np.ndarray, f: int) -> float | None:
    """Largest joint difference (deg, arms, hands and waist) to the training state of frame f or f-1, whichever is
    nearer: the train pack labels frame t with the state after raw command t, the replay is one command behind."""
    errs = [float(np.degrees(np.abs(np.asarray(state)[:31] - train[g][:31]).max())) for g in (f - 1, f)
            if 0 <= g < len(train)]
    return round(min(errs), 2) if errs else None


def read_png(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    img = cv2.imread(str(path))
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_jpg(path: Path, img: np.ndarray) -> None:
    if not path.exists():
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])


def web_video(src: Path, dst: Path, policy_from: int | None) -> None:
    """The clip re-encoded small for the browser (H.264, fast start), the replayed frames labelled "replay" and the
    policy's "policy"; a plain copy if ffmpeg is missing."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    tmp = dst.with_name(dst.stem + ".part.mp4")
    vf = []
    if FONT and policy_from is not None:
        style = (f"fontfile={FONT}:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6:x=12:y=12")
        if policy_from > 0:
            vf.append(f"drawtext=text='replay':{style}:enable='lt(n\\,{policy_from})'")
        vf.append(f"drawtext=text='policy':{style}:enable='gte(n\\,{policy_from})'")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src)] + (["-vf", ",".join(vf)] if vf else []) + [
        "-c:v", "libx264", "-preset", "medium", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        str(tmp)]
    if shutil.which("ffmpeg") and subprocess.run(cmd).returncode == 0:
        tmp.replace(dst)
    else:
        tmp.unlink(missing_ok=True)
        shutil.copyfile(src, dst)


def main() -> None:
    args = parse_args()
    root = Path(args.results)
    plan = json.loads((root / "plan.json").read_text())
    st, slots, conds = plan["settings"], plan["slots"], {c["cond"]: c for c in plan["conditions"]}
    calib = json.loads((root / "calib.json").read_text()) if (root / "calib.json").exists() else {"slots": {}}
    tp = json.loads((root / "train_paths.json").read_text())
    n_file = root / "n_trials.txt"
    planned = min(int(n_file.read_text()), len(plan["trials"])) if n_file.exists() else len(plan["trials"])
    plan_trials = plan["trials"][:planned]
    rows = sorted(read_jsonl(root / "results.jsonl"), key=lambda r: r["trial"])
    rows = [r for r in rows if r["trial"] < planned]
    train_dir = Path(st["train_dir"])
    frames = TrainFrames(train_dir)
    report = root / "report"
    (report / "videos").mkdir(parents=True, exist_ok=True)
    (report / "img").mkdir(parents=True, exist_ok=True)
    si = {}
    try:
        si = json.loads((root / "server_info.json").read_text())
    except Exception:
        pass

    # ---------------------------------------------------------------- the chosen episodes, as replayed
    slot_info = []
    states_cache: dict[int, np.ndarray] = {}

    def train_states(ep: int) -> np.ndarray:
        if ep not in states_cache:
            df = pd.read_parquet(train_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet", columns=["states"])
            states_cache[ep] = np.stack(df["states"].values)
        return states_cache[ep]

    by_ep = {e["episode"]: e for e in tp["episodes"]}
    for s in slots:
        c = calib["slots"].get(str(s["slot"]), {})
        ep = c.get("episode", s["episode"]) if c.get("ok") else s["episode"]
        e = by_ep[ep]
        info = dict(slot=s["slot"], episode=ep, ok=bool(c.get("ok")), faucet=e["faucet"]["rel"],
                    train_path=[p[1:3] for p in e["path"]], length=e["length"],
                    attempts=c.get("attempts", []))
        if c.get("ok"):
            info.update(path=[p[1:3] for p in c["path"]], frames=c["frames"], turn_start=c["turn_start"],
                        points={k: v for k, v in c["points"].items()}, path_length=c["path_length"])
            # how the replay matched the training data at its path points: joints, and the camera image
            tr = train_states(ep)
            match = {}
            want = {f for f in [0, *c["frames"].values()]}
            want |= {f + 1 for f in want} | {f - 1 for f in want if f > 0}
            tf = frames.get(ep, want)
            for p, f in [("p0", 0), *c["frames"].items()]:
                d = dict(frame=f, drift=c.get("drift", {}).get(p))
                if p in c.get("states", {}):
                    d["joint_err_deg"] = joint_err(c["states"][p], tr, f)
                img = read_png(root / "calib" / f"s{s['slot']}_ep{ep}_f{f:04d}.png")
                if img is not None:
                    diffs = {g: float(np.abs(img.astype(float) - tf[g].astype(float)).mean())
                             for g in (f - 1, f, f + 1) if g in tf}
                    if diffs:
                        d["img_diff"] = round(min(diffs.values()), 2)
                if f in tf:
                    name = f"train_ep{ep}_f{f:04d}.jpg"
                    save_jpg(report / "img" / name, tf[f])
                    d["train_img"] = f"img/{name}"
                match[p] = d
            info["match"] = match
        slot_info.append(info)
    slot_by = {s["slot"]: s for s in slot_info}

    # ---------------------------------------------------------------- trials
    trials = []
    for r in rows:
        ep, f = r["episode"], int(r.get("frame") or 0)
        to = r.get("takeover") or {}
        d = dict(trial=r["trial"], label=r["label"], kind=r["kind"], cond=r["cond"], group=group_of(r),
                 name=r["name"], slot=r["slot"], episode=ep, frame=f, point=r.get("point"),
                 dx=r.get("dx", 0.0), dy=r.get("dy", 0.0), dyaw=r.get("dyaw", 0.0), success=bool(r["success"]),
                 outcome=r["outcome"], steps=r["steps"], policy_from=to.get("call"), video=None, img=None,
                 train_img=None, pose=to.get("pose"), final=r.get("final_pose"), first_turn=r.get("first_turn_step"),
                 contacts=to.get("contacts") or [], min_hand=r.get("min_hand_dist"))
        # how close the takeover was to the training frame
        tr = train_states(ep)
        if to.get("state") is not None:
            d["joint_err_deg"] = joint_err(to["state"], tr, f)
        img = read_png(root / "takeover" / f"{r['label']}.png")
        tf = frames.get(ep, {f - 1, f, f + 1})
        if img is not None:
            save_jpg(report / "img" / f"{r['label']}.jpg", img)
            d["img"] = f"img/{r['label']}.jpg"
            diffs = [float(np.abs(img.astype(float) - tf[g].astype(float)).mean()) for g in (f - 1, f, f + 1) if g in tf]
            if diffs:
                d["img_diff"] = round(min(diffs), 2)
        if f in tf:
            name = f"train_ep{ep}_f{f:04d}.jpg"
            save_jpg(report / "img" / name, tf[f])
            d["train_img"] = f"img/{name}"
        s = slot_by.get(r["slot"], {})
        raw = {q[0]: q[1:] for q in by_ep[ep]["path"]}  # the teleop recording's pelvis, every 5 frames
        if to.get("pose") and raw:  # its frame t is raw frame t+1; the replay at call f has run raw commands 0..f-1
            q = raw[min(raw, key=lambda g: abs(g - (f - 1)))]
            d["train_dpos_cm"] = round(100 * math.hypot(to["pose"][0] - q[0], to["pose"][1] - q[1]), 1)
            d["train_dyaw_deg"] = round(to["pose"][2] - q[2], 1)
        if r["kind"] == "path" and to.get("pose") and s.get("points"):
            ref = s["points"][r["point"]]
            d["pose_err_cm"] = round(100 * math.hypot(to["pose"][0] - ref[0], to["pose"][1] - ref[1]), 1)
        if r["kind"] == "start" and r.get("dyaw") and r.get("final_pose"):
            d["end_heading"] = round(r["final_pose"][2], 1)
            d["turned_back"] = bool(abs(r["final_pose"][2]) < 0.4 * abs(r["dyaw"]))
        vids = sorted(glob.glob(str(root / (r.get("video_dir") or "-") / "head_stereo_left*.mp4")))
        dst = report / "videos" / f"{r['label']}.mp4"
        if vids:
            web_video(Path(vids[0]), dst, to.get("call"))
        d["video"] = f"videos/{dst.name}" if dst.exists() else None
        trials.append(d)
    by_trial = {t["trial"]: i for i, t in enumerate(trials)}

    # ---------------------------------------------------------------- conditions (map markers) and groups
    p0s = [s["points"]["p0"] for s in slot_info if s.get("points")]
    p0 = np.mean(p0s, axis=0).tolist() if p0s else [-0.064, 0.0, 0.0]
    markers = []
    for c in plan["conditions"]:
        ts = [t for t in plan_trials if t["cond"] == c["cond"]]
        m = dict(id=c["cond"], kind="start", name=c["name"], group=group_of(dict(c, kind="start")),
                 dx=p0[0] + c["dx"], dy=p0[1] + c["dy"],
                 dyaw=c["dyaw"], planned=len(ts), trials=[by_trial[t["trial"]] for t in ts if t["trial"] in by_trial])
        markers.append(m)
    for s in slot_info:
        if not s.get("points"):
            continue
        for p, name in (("p1", "1/3 of the way"), ("p2", "2/3 of the way"), ("p3", "reach point")):
            ts = [t for t in plan_trials if t["kind"] == "path" and t["slot"] == s["slot"] and t["point"] == p]
            x, y, yaw = s["points"][p]
            markers.append(dict(id=f"path:{p}:s{s['slot']}", kind="path", name=f"{name}, episode {s['episode']}",
                                group=name, dx=x, dy=y, dyaw=yaw, slot=s["slot"], episode=s["episode"], point=p,
                                frame=s["frames"][p], planned=len(ts),
                                trials=[by_trial[t["trial"]] for t in ts if t["trial"] in by_trial]))
    for m in markers:
        m["n"] = len(m["trials"])
        m["k"] = sum(trials[i]["success"] for i in m["trials"])
    groups = []
    for g in GROUPS:
        ts = [t for t in trials if t["group"] == g]
        planned_g = sum(group_of(t) == g for t in plan_trials)
        if planned_g:
            k = sum(t["success"] for t in ts)
            groups.append(dict(group=g, n=len(ts), k=k, planned=planned_g, ci95=wilson(k, len(ts))))

    # ---------------------------------------------------------------- map frame extras
    table = None
    if tp.get("table"):
        t0 = tp["table"]
        spawns = [by_ep[s["episode"]]["spawn"] for s in slot_info]
        sx, sy, syaw = np.mean(spawns, axis=0)
        c, s_ = math.cos(syaw), math.sin(syaw)
        dx, dy = t0["position"][0] - sx, t0["position"][1] - sy
        table = dict(center=[c * dx + s_ * dy, -s_ * dx + c * dy], size=t0["size"],
                     yaw_deg=math.degrees(t0["yaw"] - syaw))
    done = len(rows)
    rem = max(0, planned - done)
    wall_s = [r.get("wall_seconds") for r in rows if r.get("wall_seconds")]
    eta_h = round(float(np.mean(wall_s)) * rem / 3600, 1) if wall_s and rem else 0.0
    updated = time.strftime("%Y-%m-%d %H:%M")
    short = st["task"].replace("G1Wholebody", "").replace("Teleop", "").removesuffix("-v0")
    title = re.sub(r"(?<!^)(?=[A-Z])", " ", short) + " near the training trajectories"
    data = dict(task=st["task"], title=title, checkpoint=str(si.get("run_dir", st["run_dir"])),
                ckpt_step=str(st["ckpt_step"]), instruction=st["instruction"], repeats=st["repeats"],
                radii=st["radii"], yaws=st["yaws"], reach_lead=st["reach_lead"], p0=p0, table=table,
                train_paths=[[[round(p[1], 3), round(p[2], 3)] for p in e["path"][::2]] for e in tp["episodes"]],
                slots=[dict(slot=s["slot"], episode=s["episode"], ok=s["ok"], faucet=s["faucet"],
                            path=[[round(a, 3), round(b, 3)] for a, b in s.get("path", s["train_path"])],
                            frames=s.get("frames"), turn_start=s.get("turn_start"), match=s.get("match", {}))
                       for s in slot_info],
                markers=markers, groups=groups, trials=trials, planned=planned, done=done, eta_h=eta_h,
                updated=updated)
    page = (TEMPLATE.read_text().replace("__TITLE__", html.escape(title))
            .replace("__DATA_JSON__", json.dumps(data).replace("</", "<\\/")))
    # line 1 makes the file a standalone page; drop it when the page goes into a host that adds its own skeleton
    write_atomic(report / "index.html", '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
                                        'content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + page)
    write_atomic(report / "results.json", json.dumps(dict(
        task=st["task"], checkpoint=st["run_dir"], ckpt_step=st["ckpt_step"], instruction=st["instruction"],
        progress=dict(done=done, planned=planned, eta_hours=eta_h, updated=updated),
        successes=sum(t["success"] for t in trials), groups=groups, settings=st,
        episodes=[{k: v for k, v in s.items() if k not in ("train_path",)} for s in slot_info],
        conditions=[{k: v for k, v in m.items()} for m in markers],
        trials_detail=[dict(t, **{k: v for k, v in r.items() if k not in t and k != "takeover"},
                            takeover=r.get("takeover")) for t, r in zip(trials, rows)]), indent=1))
    print(f"report: {report / 'index.html'}   data: {report / 'results.json'}")
    print(f"  {done} trials" + (f" of {planned} ({eta_h:.1f} h to go)" if rem else "") + ":  "
          + "   ".join(f"{g['group']} {g['k']}/{g['n']}" for g in groups if g["n"]))
    for s in slot_info:
        mt = s.get("match", {})
        print(f"  episode {s['episode']}: " + ("replay ok, " + ", ".join(
            f"{p} frame {v['frame']} joints {v.get('joint_err_deg', '-')}° image {v.get('img_diff', '-')}"
            for p, v in mt.items()) if s["ok"] else "not calibrated"))


if __name__ == "__main__":
    main()
