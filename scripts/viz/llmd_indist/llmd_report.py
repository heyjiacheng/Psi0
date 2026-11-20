"""Stage 4 of the LLMD in-distribution experiment: the result page (Psi0 venv, CPU, no network).

    llmd_report.py --out <experiment dir> --exp3 <experiment 3 dir> [--fragment <file>] [--scores-suffix <s>]

Reads what fit_llmd.py wrote (<out>/scores<s>/summary.json and scores.npz), the observation tables
(<out>/feats<s>/<set>/obs.jsonl), the renders (grid: <exp3>/exp3_indist_signal/<file>; extra: <out>/renders/<file>)
and experiment 3's floor plan (<exp3>/prep/plan.json, train_index.json, room_map.npz), and writes <out>/index.html:
one self-contained page (no network; inline CSS, JS and SVG; the camera views as small JPEG data URIs).
--fragment <file> also writes the same page without its document head and without links to local files, the form a
page host that supplies its own head expects.

The page follows experiment 3's page (vlm_camera_sweep/indist_report.py): one column of findings, each a headline
that is the conclusion, one chart under it, one or two short lines of text. Every sentence that states a result is
chosen here from the numbers by a small pure function (h1_sentence, bands_headline, where_headline, line_headline,
turned_headline, h264_headline, trials_headline, ...), so another run gets its own words. The every-variant tables
are in the collapsed "Also checked" blocks at the end.

Colour means the same in every chart. Decisions: blue = accepted, orange = a far pose wrongly accepted (a zone of
such poses, a decision that flips), grey = rejected / context. Detectors: violet = paper LLMD, green and magenta =
the two start-referenced forms, neutral grey (dashed as a line) = token match, experiment 3's reference.

The page is robust to what a run lacks: a detector, an observation set or a section that summary.json does not
have is left out, and a number that is not finite never reaches the page (a short "not available" note instead).
A detector without a finite limit has no verdict: its card says "No limit", it is in no chart that needs a limit and
the details tables show "no limit" / n/a for it (the fit's placeholder verdict and counts are not shown). The section
headlines speak about the paper's detector; when that one has no limit, about the first headline detector with one.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HEADLINE = ("paper", "start_tied", "start_lw", "token_match")  # the cards and every main chart, in this order
NAME = dict(paper="Paper LLMD", start_tied="Start-referenced, tied covariance", start_lw="Start-referenced, Ledoit-Wolf",
            token_match="Token match, experiment 3")
SHORT = dict(paper="Paper LLMD", start_tied="Start-ref. tied", start_lw="Start-ref. LW", token_match="Token match")
COLOR = dict(paper="d1", start_tied="d2", start_lw="d3", token_match="d4")
VERDICT = {"pass": "Passes", "ranks": "Ranks only", "fail": "Fails"}
# the earlier closed-loop trials shown in the main view: (observation set, part, label)
TRIAL_GROUPS = (("target", "standing", "Faucet moved"),
                ("target_tf", "standing", "Faucet moved, training prompt"),
                ("traj", "standing", "Robot moved or turned"),
                ("traj", "takeover", "Taken over mid-demo"))
TRIAL_SETS = dict(traj="robot moved, turned or taken over mid-demo", traj_ms="the same, median training-start state",
                  target="faucet moved", target_tf="faucet moved, training prompt", target_ms="faucet moved, median training-start state")
CODECS = (("h264", "the writer's settings (crf 23)"), ("h264s", "matched to the training start frames"),
          ("h264c", "lossless: the colour conversion alone"))
THUMB_W, THUMB_Q = 160, 70
NA = "n/a"
MINUS = "−"
NBSP = "\u00a0"  # keeps a number and its unit on one line


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the experiment directory (scores<s>/, feats<s>/, renders/ are read; index.html written)")
    p.add_argument("--exp3", required=True, help="experiment 3's directory (read-only): prep/ and exp3_indist_signal/")
    p.add_argument("--fragment", default=None, help="also write the page without its document head to this file")
    p.add_argument("--scores-suffix", default="", help="read scores<s>/ and feats<s>/ (a pilot)")
    return p.parse_args()


# ------------------------------------------------------------------------------------------------ small helpers
def fin(v) -> bool:
    """A real, finite number."""
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) and math.isfinite(float(v))


def dig(o, *path, default=None):
    """o[path[0]][path[1]]... or `default` when a step is missing (or the value is None)."""
    for k in path:
        if isinstance(o, dict) and k in o:
            o = o[k]
        elif isinstance(o, (list, tuple)) and isinstance(k, int) and -len(o) <= k < len(o):
            o = o[k]
        else:
            return default
    return default if o is None else o


def clean(o):
    """JSON-safe copy: numpy types unpacked, every NaN / infinity replaced by None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def sig(v, n: int = 4):
    """v rounded to n significant digits, None when it is not a finite number."""
    if not fin(v):
        return None
    v = float(v)
    return 0.0 if v == 0 else round(v, n - 1 - int(math.floor(math.log10(abs(v)))))


def num(v, n: int = 3) -> str:
    """A number for a table cell: n significant digits, thousands separators, powers of ten only at the extremes."""
    if not fin(v):
        return NA
    v = float(v)
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1e6 or a < 1e-3:
        m, e = f"{v:.{n - 1}e}".split("e")
        m = m.rstrip("0").rstrip(".") if "." in m else m
        return f"{m}e{int(e)}"
    return f"{v:,.0f}" if a >= 100 else f"{v:.{n}g}"


def dec3(v) -> str:
    """A value between -1 and 1 (AUROC, rank correlation) for a table cell: always three decimals."""
    return f"{float(v):.3f}" if fin(v) else NA


def pair(p):
    """(k, n) of a [k, n] count with n > 0, else None."""
    if isinstance(p, (list, tuple)) and len(p) == 2 and fin(p[0]) and fin(p[1]) and p[1] > 0:
        return int(p[0]), int(p[1])
    return None


def frac(p) -> str:
    q = pair(p)
    return f"{q[0]}/{q[1]}" if q else NA


def rate(p):
    q = pair(p)
    return q[0] / q[1] if q else None


def pct(v) -> str:
    """A share (0..1, or a [k, n] count) as a percentage."""
    if isinstance(v, (list, tuple)):
        v = rate(v)
    return f"{round(float(v) * 100)}%" if fin(v) else NA


def times(v) -> str:
    """A ratio: 0.48×, 1.05×, 23×."""
    if not fin(v):
        return NA
    v = float(v)
    return f"{v:,.0f}×" if v >= 10 else f"{v:.2f}×" if v >= 0.095 or v <= 0 else num(v, 2) + "×"


def dist(m) -> str:
    """A length in metres, in the unit a person would say: 30 cm, 1.5 m, 3 m."""
    if not fin(m):
        return NA
    m = abs(float(m))
    return f"{m * 100:.0f}{NBSP}cm" if m < 0.995 else f"{m:.1f}".rstrip("0").rstrip(".") + f"{NBSP}m"


def degs(v, sign: bool = True) -> str:
    if not fin(v):
        return NA
    v = float(v)
    return "0°" if abs(v) < 0.05 else f"{('+' if v > 0 else MINUS) if sign else ''}{abs(v):g}°"


def change(log_ratio) -> str:
    """A median log ratio in words: '12% higher', '8% lower', '2.30× as high', 'about the same'."""
    if not fin(log_ratio):
        return NA
    f = math.exp(max(-50.0, min(50.0, float(log_ratio))))
    if abs(f - 1) < 0.01:
        return "about the same"
    if f >= 2:
        return f"{times(f)} as high"
    return f"{abs(f - 1) * 100:.0f}% {'higher' if f > 1 else 'lower'}"


def listing(items: list) -> str:
    items = [str(i) for i in items]
    return "" if not items else items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def of_the(k: int, n: int, what: str) -> str:
    """'all 13 poses', 'none of the 13 poses', '5 of the 13 poses'."""
    return f"all {n} {what}" if k == n else f"none of the {n} {what}" if k == 0 else f"{k} of the {n} {what}"


def cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def esc(t) -> str:
    return html.escape(str(t), quote=True)


def plain(t) -> str:
    """Text taken over from a file as it is, without the words a non-finite number prints as."""
    return re.sub(r"(?<![\w.])-?(?:nan|NaN|inf|Infinity)(?![\w.])", "not available", str(t))


def where_text(r: dict) -> str:
    parts = [f"{abs(r[k]):.2f} m {pos if r[k] > 0 else neg}"
             for k, pos, neg in (("fx", "ahead", "back"), ("fy", "left", "right")) if fin(r.get(k)) and abs(r[k]) > 0.004]
    if fin(r.get("dyaw")) and abs(r["dyaw"]) > 0.5:
        parts.append(f"turned {abs(r['dyaw']):.0f}° {'left' if r['dyaw'] > 0 else 'right'}")
    return cap(", ".join(parts)) if parts else "At the training start"


def outside_text(r: dict) -> str:
    d = r.get("pose_dist")
    if r.get("in_range") or not fin(d):
        return "in the training range"
    return f"{dist(d)} outside the training range"


# ------------------------------------------------------------------------------------------------ the sentences
def h1_sentence(paper: str | None, starts: list, token: str | None, no_limit: bool = False) -> str:
    """The page's main result from the far-vs-close verdicts ('pass' | 'ranks' | 'fail') of the paper's detector,
    of the start-referenced forms that exist, and of token match (None = no verdict; no_limit says why the paper's
    detector has none)."""
    tell = "tells far poses from training-like ones"
    no_tell = "does not tell far poses from training-like ones"
    ranks = "ranks far poses above training-like ones but cannot set a limit"
    if paper not in VERDICT:
        return "This run has no verdict for the paper's LLMD" + (": no limit could be set." if no_limit else ".")
    starts = [v for v in starts if v in VERDICT]
    n, n_pass, n_rank = len(starts), sum(v == "pass" for v in starts), sum(v == "ranks" for v in starts)
    either = "in the paper's form or referenced to the training starts"
    forms = "its start-referenced form" if n == 1 else "its start-referenced forms"
    if paper == "pass":
        kind = "pass"
        a = (f"The paper's LLMD {tell}" if not n else
             f"LLMD {tell}, in the paper's form and referenced to the training starts" if n_pass == n else
             f"LLMD {tell} in the paper's form and in one of {forms}" if n_pass else
             f"The paper's LLMD {tell}, but {forms} {'does' if n == 1 else 'do'} not")
    elif n_pass:
        kind = "pass"
        a = (f"LLMD {tell} only when it is referenced to the training starts" if n_pass == n else
             f"LLMD {tell} only in one of {forms}")
    elif paper == "ranks":
        kind = "ranks"
        a = (f"The paper's LLMD {ranks}" if not n else f"LLMD {ranks}, {either}" if n_rank == n else
             f"The paper's LLMD {ranks}, and referencing it to the training starts does not help")
    else:
        kind = "mixed" if n_rank else "fail"
        a = (f"The paper's LLMD {no_tell}" if not n else
             f"The paper's LLMD {no_tell}, and referenced to the training starts it at best ranks them" if n_rank else
             f"LLMD {no_tell}, {either}")
    b = {"pass": "; token match does too" if kind == "pass" else "; token match tells them apart",
         "ranks": {"pass": "; token match only ranks them", "fail": "; token match at least ranks them"}.get(kind, "; token match also only ranks them"),
         "fail": "; token match fails this test" if kind in ("pass", "ranks") else "; token match fails this test too"}.get(token, "")
    return a + b + "."


def lede_sentence(counts: dict) -> str:
    n = dig(counts, "grid", "poses")
    poses = f"{n} standing poses around the table" if fin(n) else "Standing poses around the table"
    return (f"{poses} of one training scene, one camera view each. LLMD, the paper's detector, scores a view by how far the "
            "action expert's last-layer features lie from the training data; a view is accepted when its score is at or "
            "under the limit.")


def rule_sentence(rule: dict, counts: dict) -> str:
    """The line under the cards: the heart of the rule fixed in advance (rule_items has all of it)."""
    nl, ns = dig(counts, "grid", "in_range_lattice"), dig(counts, "sets", "start")
    lat = math.ceil(rule["in_range_lattice_min"] * nl - 1e-9) if fin(rule.get("in_range_lattice_min")) and fin(nl) else None
    parts = []
    if fin(rule.get("far_accept_max")):
        far = f" (far = more than {dist(rule['band_m'])} outside the training range)" if fin(rule.get("band_m")) else ""
        near = f" and at least {lat} of the {nl} in-range grid poses" if lat is not None else ""
        # the ranking conditions in a few plain words (their numbers are in the first "Also checked" block), so that
        # no verdict on the cards contradicts this line
        rank = ", and scores nearly every far pose above the in-range ones" if fin(rule.get("auroc_min")) else ""
        fails = (f"; it fails when it accepts {pct(rule['fail_far_accept'])} or more of the far poses"
                 + (" or ranks them poorly" if fin(rule.get("fail_auroc")) else "")) if fin(rule.get("fail_far_accept")) else ""
        parts.append(f"Rule fixed before any pose was scored, in short: a detector passes when it accepts at most "
                     f"{pct(rule['far_accept_max'])} of the far poses{far}{near}{rank}{fails}.")
    if fin(rule.get("alpha")):
        parts.append(f"The limit is the score that {pct(1 - rule['alpha'])} of the{' ' + str(ns) if fin(ns) else ''} re-rendered "
                     "training starts stay under.")
    return " ".join(parts)


def rule_items(rule: dict, counts: dict) -> list:
    """The rule fixed in advance as (term, plain words) pairs; a part whose numbers are missing is left out."""
    g = dig(counts, "grid", default={})
    nl, nt, ns = g.get("in_range_lattice"), g.get("turned"), dig(counts, "sets", "start")
    need = lambda share, n: math.ceil(share * n - 1e-9) if fin(share) and fin(n) else None
    lat, rot = need(rule.get("in_range_lattice_min"), nl), need(rule.get("turned_rejected_min"), nt)
    out = []
    if fin(rule.get("alpha")):
        out.append(("Limit", f"the score that {pct(1 - rule['alpha'])} of the{' ' + str(ns) if fin(ns) else ''} re-rendered "
                             "training starts stay under"))
    if fin(rule.get("band_m")):
        out.append(("Far", f"more than {dist(rule['band_m'])} outside the training range"))
    if fin(rule.get("far_accept_max")) and lat is not None and fin(rule.get("auroc_min")):
        out.append(("Passes", f"at most {pct(rule['far_accept_max'])} of the far poses accepted, at least {lat} of the {nl} "
                              f"in-range poses of the position grid accepted, and a far pose scored above an in-range one "
                              f"in at least {pct(rule['auroc_min'])} of pairs"))
    if fin(rule.get("fail_far_accept")) and fin(rule.get("fail_auroc")):
        out.append(("Fails", f"{pct(rule['fail_far_accept'])} or more of the far poses accepted, or under "
                             f"{pct(rule['fail_auroc'])} of pairs; in between it ranks only"))
    if rot is not None and fin(rule.get("turned_deg")):
        out.append(("Turning passes", f"at least {rot} of the {nt} poses turned {rule['turned_deg']:g}° or more rejected, "
                                      "and +5° accepted"))
    return out


def notice_sentence(names: list, n_start, alpha=0.05) -> str:
    """Shown only when a headline detector has no finite limit; says why when the reference set is too small for
    the quantile the limit is."""
    if not names:
        return ""
    few = fin(n_start) and fin(alpha) and math.ceil((n_start + 1) * (1 - alpha)) > n_start
    why = f" (only {n_start} re-rendered training start{'s' if n_start != 1 else ''})" if few else ""
    return (f"No limit could be set for {listing(names)}{why}, so {'it has' if len(names) == 1 else 'they have'} no verdict "
            f"and {'is' if len(names) == 1 else 'are'} left out of the charts that need a limit.")


def bands_headline(name: str, in_range, far) -> str:
    """Far vs close from the detector's counts: in_range, far = [accepted, n]."""
    a, f = pair(in_range), pair(far)
    if not a or not f:
        return ""
    pa, pf = a[0] / a[1], f[0] / f[1]
    near = f"{of_the(a[0], a[1], 'poses')} in the training range"
    if a[0] == 0 and f[0] == 0:
        return f"{name} rejects every pose, also the {a[1]} in the training range."
    if f[0] == 0:
        return (f"{name} accepts {near} and none of the {f[1]} far ones." if pa >= 0.75 else
                f"{name} accepts no far pose, but only {a[0]} of the {a[1]} poses in the training range.")
    far_n = f"all {f[1]} far ones" if f[0] == f[1] else f"{f[0]} of the {f[1]} far ones"
    share = "" if f[0] == f[1] else f" ({pct(pf)})"
    if a[0] == 0:
        return f"{name} accepts none of the {a[1]} poses in the training range, but {far_n}{share}."
    if pa < 0.75:
        return f"{name} accepts only {a[0]} of the {a[1]} poses in the training range, and {far_n}{share}."
    if pf <= 0.05:
        return f"{name} accepts {near} and only {far_n}."
    if pf >= 0.5:
        return f"{name} accepts {near}, but also {far_n}{share}."
    return f"{name} accepts {near}, and still {far_n}{share}."


def why_headline(name: str, rise, room) -> str:
    """Why a limit works or not, inside one scene: how much standing far away raises the score (the median far pose
    against this scene's own training start), against how far the limit lies above that start. The limit comes from
    the training starts of all scenes, so it lies far above the start of a scene that scores low. A far pose is
    accepted exactly when its own rise is at most that factor."""
    if not (fin(rise) and fin(room) and room > 0):
        return ""
    if abs(rise / room - 1) < 0.05:
        return (f"Standing far away raises {name}'s score {times(rise)}, about as far as the limit lies above this "
                f"scene's own start ({times(room)}).")
    if rise <= 1.02 and rise < room:
        return f"Standing far away does not raise {name}'s score ({times(rise)} this scene's own start); the limit is {times(room)} that start."
    if rise < room:
        return (f"Standing far away raises {name}'s score {times(rise)}, but the limit, set by the starts of all scenes, "
                f"is {times(room)} this scene's own start.")
    return f"Standing far away raises {name}'s score {times(rise)}, and the limit is only {times(room)} this scene's own start."


def why_sub(spread) -> str:
    """The line under the chart: where the limit comes from, and how much the starts differ between scenes."""
    line = "The limit is set so that 95% of the training starts, each in its own scene, stay under it"
    return line + (f"; between scenes their scores differ {times(spread)} (5th to 95th percentile)." if fin(spread) else ".")


def where_headline(name: str, n_acc: int, in_range, far, far_reach, near_reach) -> str:
    """Where the accepted poses are: far = [accepted far poses, n], far_reach / near_reach = how far outside the
    training range the farthest accepted far pose / accepted pose lies (m)."""
    a, f = pair(in_range), pair(far)
    if not f:
        return ""
    if n_acc == 0:
        return f"{name} accepts no pose in the room."
    if f[0] == 0:
        short = f", and only {a[0]} of the {a[1]} in it" if a and a[0] < a[1] else ""
        if fin(near_reach) and near_reach > 0.004:
            return f"{name} accepts only poses within {dist(near_reach)} of the training range{short}."
        return f"{name} accepts only poses in the training range{short}."
    reach = f", up to {dist(far_reach)} from the training range" if fin(far_reach) else ""
    if f[0] >= 0.5 * f[1]:
        return f"{name} accepts poses all over the room{reach}."
    if f[0] == 1:
        return f"{name} accepts one far pose" + (f", {dist(far_reach)} from the training range." if fin(far_reach) else ".")
    return f"{name} accepts {f[0]} far poses" + (f", the farthest {dist(far_reach)} from the training range." if fin(far_reach) else ".")


def line_story(pts: list, lo: float, hi: float) -> dict | None:
    """What a score does along a line of poses. pts = [(offset, score / limit, far)] sorted by offset; lo, hi = the
    training range on that line. Each side lists its FAR poses in outward order: n of them, `over` the limit, the
    outermost at `far`, the nearest one over the limit at `first` (None: never), `clean` when every pose from there
    on is over it too, the outermost one under the limit at `last_under`."""
    inside = [v for x, v, _ in pts if lo - 1e-9 <= x <= hi + 1e-9] or [v for x, v, _ in pts if abs(x) < 1e-9]
    if not inside:
        return None

    def side(seq: list):
        seq = [(abs(x), v) for x, v, far in seq if far]
        if not seq:
            return None
        over = [v > 1 for _, v in seq]
        k = over.index(True) if any(over) else None
        return dict(n=len(seq), over=sum(over), far=seq[-1][0], first=None if k is None else seq[k][0],
                    clean=k is not None and all(over[k:]), last_under=max((x for (x, _), o in zip(seq, over) if not o), default=None))

    return dict(r_in=float(np.median(inside)), neg=side([q for q in reversed(pts) if q[0] < lo - 1e-9]),
                pos=side([q for q in pts if q[0] > hi + 1e-9]))


def line_headline(name: str, story: dict | None, neg_word: str, pos_word: str) -> str:
    """Approach / sideways: does the score leave the accepted side when the robot leaves the training range?"""
    if story is None:
        return ""
    subject = f"{name}'s score"
    if story["r_in"] > 1:
        return f"{subject} is over the limit even at the training start: {times(story['r_in'])} the limit."
    sides = [(s, w) for s, w in ((story["neg"], neg_word), (story["pos"], pos_word)) if s]
    if not sides:
        return f"{subject} is under the limit at the training start; this line has no far pose."
    n, over = sum(s["n"] for s, _ in sides), sum(s["over"] for s, _ in sides)
    if over == 0:
        return f"{subject} never crosses the limit on this line: it is still under it " + " and ".join(f"{dist(s['far'])} {w}" for s, w in sides) + "."
    clean = [(s, w) for s, w in sides if s["clean"]]
    never = [(s, w) for s, w in sides if s["over"] == 0]
    if len(clean) == len(sides):
        return f"{subject} crosses the limit " + " and ".join(f"{dist(s['first'])} {w}" for s, w in sides) + "."
    if len(clean) == 1 and len(never) == 1:
        (s, w), (o, ow) = clean[0], never[0]
        return f"{subject} crosses the limit {dist(s['first'])} {w}, but {dist(o['far'])} {ow} it is still under it."
    if n - over >= over:
        s, w = max(((s, w) for s, w in sides if s["last_under"] is not None), key=lambda q: q[0]["last_under"])
        return f"{subject} stays under the limit at {n - over} of the {n} far poses on this line, even {dist(s['last_under'])} {w}."
    # a side that crossed the limit and came back under it farther out (there is one: not every side is clean or never)
    s, w = max(((s, w) for s, w in sides if s["over"] and not s["clean"]), key=lambda q: q[0]["last_under"])
    return f"{subject} is over the limit at {over} of the {n} far poses on this line, but under it again {dist(s['last_under'])} {w}."


def beyond_line(name: str, pts: list, what: str = "the table") -> str:
    """The poses of a line that lie past an obstacle: pts = [(offset, score / limit)]."""
    pts = [(x, v) for x, v in pts if fin(v)]
    if not pts:
        return ""
    where = listing([dist(x) for x, _ in pts]) + " ahead"
    under = sum(v <= 1 for _, v in pts)
    if under == len(pts):
        return f"Past {what} ({where}) {name}'s score is under the limit again."
    if under == 0:
        return f"Past {what} ({where}) {name}'s score is over the limit."
    return f"Past {what} ({where}) {name}'s score is under the limit at {under} of {len(pts)} poses."


def no_limit_headline(name: str, what: str = "its score cannot be judged along this line") -> str:
    """The headline of a section whose chart needs a limit, for a detector that has none."""
    return f"No limit could be set for {name}, so {what}."


def turned_headline(name: str, accepted: list, rejected, turned_deg: float = 20.0) -> str:
    """Turned on the spot: accepted = the headings accepted (deg), rejected = [rejected, n] of those turned by
    turned_deg or more."""
    r = pair(rejected)
    if not r:
        return ""
    k, n = r
    acc = sorted(float(a) for a in accepted if fin(a))
    t = f"{turned_deg:g}°"
    wide = sorted((a for a in acc if abs(a) >= turned_deg - 1e-9), key=abs)
    if k == n and not acc:
        return f"{name} rejects every turned pose, also +5°, which the training starts cover."
    if k == n:
        span = listing([degs(a) for a in acc]) if len(acc) <= 3 else f"{degs(acc[0])} to {degs(acc[-1])}"
        return f"{name} rejects all {n} poses turned {t} or more; it accepts only {span}."
    away = ", facing away from the table" if wide and abs(wide[-1]) >= 135 else ""
    if k == 0:
        return f"{name} accepts every heading{', even' + away[1:] if away else ''}: none of the {n} poses turned {t} or more is rejected."
    if n - k == 1 and wide:
        return f"{name} rejects {k} of the {n} poses turned {t} or more; the one it accepts is turned {degs(wide[-1], sign=False)}{away}."
    if 2 * k >= n:
        which = listing([degs(a) for a in sorted(wide)]) if 0 < len(wide) <= 3 else f"{n - k} of them"
        return f"{name} rejects {k} of the {n} poses turned {t} or more; it still accepts {which}."
    still = f"; it accepts even {degs(wide[-1], sign=False)}{away}" if wide else ""
    return f"{name} rejects only {k} of the {n} poses turned {t} or more{still}."


def h264_headline(name: str, h: dict, raw: dict) -> str:
    """Does the H.264 round trip change the answer? h = detectors[d].h264 (decisions_changed = [k, n], grid = the
    verdict after the round trip), raw = detectors[d].grid."""
    c = pair(h.get("decisions_changed"))
    if not c:
        return f"The H.264 round trip could not be judged for {name}."
    k, n = c
    v0, v1 = dig(raw, "verdict", "far_vs_close"), dig(h, "grid", "verdict", "far_vs_close")
    far = pair(dig(h, "grid", "far"))
    if v0 in VERDICT and v1 in VERDICT and v0 != v1:
        return (f"The H.264 round trip changes the verdict for {name}, from “{VERDICT[v0].lower()}” to “{VERDICT[v1].lower()}”: "
                f"{k} of {n} decisions flip.")
    if k == 0:
        return f"The H.264 round trip changes none of {name}'s {n} decisions."
    flips = f"The H.264 round trip flips {k} of {name}'s {n} decisions ({pct(k / n)})"
    if v1 == "fail" and far:
        return f"{flips} but not the verdict: {far[0]} of the {far[1]} far poses are still accepted."
    if v1 == "pass" and far:
        left = "none" if far[0] == 0 else f"only {far[0]}"
        return f"{flips} but not the verdict: {left} of the {far[1]} far poses {'is' if far[0] == 1 else 'are'} accepted after it."
    if v1 == "ranks":
        return f"{flips} but not the verdict: it still only ranks the poses."
    return flips + "."


def h264_no_limit_headline(name: str, h: dict) -> str:
    """The round trip for a detector without a limit: only how the scores move."""
    move = change(h.get("median_log_ratio"))
    if move == NA:
        return no_limit_headline(name, "the H.264 round trip cannot be judged")
    after = "its scores stay about the same" if move == "about the same" else f"its scores come out {move}"
    return f"No limit could be set for {name}; after the H.264 round trip {after}."


def h264_scores_line(h: dict) -> str:
    """How the scores move: the median shift and whether the order of the poses is kept."""
    move, rho = change(h.get("median_log_ratio")), h.get("spearman")
    a = "" if move == NA else "A pose scores about the same after the round trip" if move == "about the same" else f"A pose scores {move} after the round trip"
    if not fin(rho):
        return a + "." if a else ""
    kept = "is kept" if rho >= 0.95 else "is mostly kept" if rho >= 0.8 else "is only loosely kept"
    b = f"the order of the poses {kept} (rank correlation {rho:.2f})"
    return f"{a}; {b}." if a else cap(b) + "."


def h264_starts_line(st: dict) -> str:
    """Training video frame vs its re-render vs the round-tripped re-render (median log ratios to the video frame)."""
    a, b = st.get("median_log_render_over_video"), st.get("median_log_codec_over_video")
    if not fin(a) or not fin(b):
        return ""
    say = lambda v: "the same as" if change(v) == "about the same" else f"{change(v)} than"
    first = f"A re-rendered training start scores {say(a)} its training video frame"
    if abs(b) < abs(a) - 0.01:
        return f"{first}; the round trip brings it closer ({say(b)} the video frame)."
    if abs(b) > abs(a) + 0.01:
        return f"{first}; the round trip moves it away ({say(b)} the video frame)."
    return f"{first}; the round trip changes little."


def h264s_line(h: dict) -> str:
    """The round trip with the compression matched to the training start frames (nothing without its limit)."""
    c, far = pair(h.get("decisions_changed")), pair(dig(h, "grid", "far"))
    if not c or not fin(h.get("tau")):
        return ""
    tail = f", and {of_the(far[0], far[1], 'far poses')} {'is' if far[0] == 1 else 'are'} accepted" if far else ""
    return f"With the compression matched to the training start frames, {c[0]} of {c[1]} decisions flip{tail}."


def trial_link(auroc, thirds) -> str:
    """'yes' = a low score goes with success, 'opposite' = with failure, 'no' = neither."""
    lo, hi = rate(dig(thirds, 0)), rate(dig(thirds, -1))
    if not fin(auroc) or lo is None or hi is None:
        return "no"
    return "yes" if auroc >= 0.6 and lo > hi else "opposite" if auroc <= 0.4 and lo < hi else "no"


def trials_headline(name: str, groups: list) -> str:
    """groups = [dict(label, n, thirds, link)], one per set of earlier trials."""
    if not groups:
        return ""
    n = len(groups)
    yes, opp = [g for g in groups if g["link"] == "yes"], [g for g in groups if g["link"] == "opposite"]
    big = max(groups, key=lambda g: g["n"])
    lo, hi = pct(big["thirds"][0]), pct(big["thirds"][-1])
    numbers = f"{lo} of the lowest third succeed, {hi} of the highest"
    if len(yes) == n:
        return (f"A low {name} score goes with success: {numbers}." if n == 1 else
                f"A low {name} score goes with success in {'both' if n == 2 else f'all {n}'} trial sets: in the largest, {numbers}.")
    labels = listing([g["label"].lower() for g in yes])
    if yes and opp:
        return (f"A low {name} score goes with success in only {len(yes)} of {n} trial sets ({labels}); "
                f"in {len(opp)} it goes with failure.")
    if yes:
        return f"A low {name} score goes with success in only {len(yes)} of {n} trial sets ({labels})."
    if opp:
        some = "these trials" if n == 1 else "both trial sets" if len(opp) == n == 2 else f"{len(opp)} of {n} trial sets"
        return f"A low {name} score does not go with success; in {some} it even goes with failure."
    return (f"A low {name} score does not go with success: {numbers}" + ("." if n == 1 else " in the largest trial set."))


# ------------------------------------------------------------------------------------------------ floor plan, images
def heading_xy(plan: dict, xy) -> np.ndarray:
    """World points (..., 2) as plot points (left y, forward x) of the start pose's heading frame."""
    x0, y0, yaw0 = plan["anchor"]["pelvis"]
    c, s = math.cos(yaw0), math.sin(yaw0)
    d = np.asarray(xy, dtype=float) - np.array([x0, y0])
    return np.stack([-s * d[..., 0] + c * d[..., 1], c * d[..., 0] + s * d[..., 1]], axis=-1)


def train_box(plan: dict) -> tuple[float, float, float, float]:
    """(fx_lo, fx_hi, fy_lo, fy_hi): base positions, at the start heading, that see the faucet where a training
    start had it."""
    f0, tr = plan["anchor"]["faucet_rel"], plan["train_range"]
    return f0[0] - tr["dx"][1], f0[0] - tr["dx"][0], f0[1] - tr["dy"][1], f0[1] - tr["dy"][0]


def train_headings(plan: dict) -> tuple[float, float] | None:
    """(lo, hi) in degrees: headings at the start position that keep the faucet's bearing inside the training range."""
    f0, tr = dig(plan, "anchor", "faucet_rel"), dig(plan, "train_range", "dyaw")
    if not f0 or len(f0) < 3 or not tr:
        return None
    return math.degrees(f0[2] - tr[1]), math.degrees(f0[2] - tr[0])


def wall_rects(room, plan: dict, view: dict, res: float = 0.04) -> list[list[float]]:
    """What stands on the floor inside `view` (plot metres: l, r = left edge and right edge as metres to the left of
    the start, t, b = ahead), as rectangles [left edge, top edge, width, height], one per run of a row."""
    x0, y0, yaw0 = plan["anchor"]["pelvis"]
    c, s = math.cos(yaw0), math.sin(yaw0)
    lefts, aheads = np.arange(view["l"], view["r"], -res), np.arange(view["t"], view["b"], -res)
    L, A = np.meshgrid(lefts - res / 2, aheads - res / 2)  # cell centres; rows run down the page
    occ, lo, r = room["robot"], room["lo"], float(room["res"])
    ix = np.floor((x0 + c * A - s * L - lo[0]) / r).astype(int)
    iy = np.floor((y0 + s * A + c * L - lo[1]) / r).astype(int)
    inside = (ix >= 0) & (ix < occ.shape[0]) & (iy >= 0) & (iy < occ.shape[1])
    full = np.zeros(L.shape, bool)
    full[inside] = occ[ix[inside], iy[inside]]
    rects = []
    for k, row in enumerate(full):
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
        rects += [[round(float(lefts[a]), 3), round(float(aheads[k]), 3), round(float((b - a) * res), 3), res]
                  for a, b in zip(edges[::2], edges[1::2])]
    return rects


def table_polygon(plan: dict) -> np.ndarray:
    tb = plan["anchor"]["table"]
    half, ct, st = np.asarray(tb["size"][:2]) / 2, math.cos(tb["yaw"]), math.sin(tb["yaw"])
    return heading_xy(plan, [[tb["position"][0] + ct * a * half[0] - st * b * half[1],
                              tb["position"][1] + st * a * half[0] + ct * b * half[1]]
                             for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def crossing(poly: np.ndarray, axis: int) -> tuple[float, float] | None:
    """Where the polygon (plot points: left, ahead) covers the line left = 0 (axis 1: the interval of `ahead`) or
    ahead = 0 (axis 0: the interval of `left`)."""
    other, hits = 1 - axis, []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        if (a[other] > 0) != (b[other] > 0):
            t = a[other] / (a[other] - b[other])
            hits.append(float(a[axis] + t * (b[axis] - a[axis])))
    return (min(hits), max(hits)) if len(hits) >= 2 else None


def thumbnail(path: Path) -> str | None:
    try:
        im = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    im = im.resize((THUMB_W, max(1, round(THUMB_W * im.height / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=THUMB_Q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------------ "Also checked"
def table(head: list, rows: list, wrap_last: bool = False) -> str:
    if not rows:
        return ""
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    wrap = ' class="wrap"'
    body = "".join("<tr>" + "".join(f"<td{wrap if wrap_last and j == len(r) - 1 else ''}>{esc(c)}</td>"
                                    for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><tr>{th}</tr>{body}</table></div>'


def bullets(items: list) -> str:
    items = [i for i in items if i]
    return '<ul class="defs">' + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>" if items else ""


def note(text: str) -> str:
    return f'<p class="cap">{esc(text)}</p>' if text else ""


def block(title: str, *parts: str) -> str:
    """A collapsed block; nothing at all when it has no table and no list."""
    body = "".join(p for p in parts if p)
    return f"<details><summary>{esc(title)}</summary>{body}</details>" if "<table" in body or "<ul" in body else ""


def judged(g) -> bool:
    """A grid evaluation made with a finite limit: only then do its verdict and its accept counts mean something
    (the fit writes placeholders otherwise)."""
    return fin(dig(g, "tau"))


def verdict_word(g: dict | None, key: str = "far_vs_close") -> str:
    if isinstance(g, dict) and "tau" in g and not judged(g):
        return "no limit"
    v = dig(g, "verdict", key)
    return {"pass": "passes", "ranks": "ranks only", "fail": "fails"}.get(v, NA)


def count(g: dict | None, key: str) -> str:
    """'k/n' of a grid evaluation's accept count, n/a when it was made without a finite limit."""
    return frac(dig(g, key)) if judged(g) else NA


def span(vals: list, fmt=num) -> str:
    v = [float(x) for x in vals if fin(x)]
    return NA if not v else fmt(min(v)) if min(v) == max(v) else f"{fmt(min(v))} to {fmt(max(v))}"


def exact(v, digits: int = 4) -> str:
    return f"{float(v):,.{digits}f}" if fin(v) else NA


def variant_name(v: str) -> str:
    m = re.fullmatch(r"s(\d+)(_bf16)?", v)
    if not m:
        return v
    return f"noise seed {m.group(1)}" + (", action expert in bf16 as served" if m.group(2) else "")


def also_checked(S: dict, links: list) -> str:
    D = S.get("detectors") or {}
    label = lambda d: dig(S, "labels", d) or dig(D, d, "label") or d
    tau_of = lambda d: dig(D, d, "grid", "tau")
    limited = lambda d: fin(tau_of(d))  # the detector has a limit: its accept counts are not placeholders
    rel = lambda v, d: times(v / tau_of(d)) if fin(v) and fin(tau_of(d)) and tau_of(d) > 0 else NA
    lifted = lambda v: times(math.exp(v)) if fin(v) else NA
    main = [d for d in HEADLINE if d in D]
    out = []

    # every detector
    rows, rows_v, rows_b = [], [], []
    for d, e in D.items():
        g = e.get("grid") or {}
        rows.append([label(d), verdict_word(g), verdict_word(g, "rotation"), count(g, "in_range"), count(g, "far"),
                     count(g, "turned_rejected"), dec3(g.get("auroc_far")), num(g.get("tau")),
                     {"render": "re-rendered starts", "video": "training video frames"}.get(e.get("tau_used"), NA)])
        gv = e.get("grid_video_tau")
        if gv:
            rows_v.append([label(d), verdict_word(gv), verdict_word(gv, "rotation"), count(gv, "in_range"), count(gv, "far"),
                           count(gv, "turned_rejected"), dec3(gv.get("auroc_far")), num(gv.get("tau"))])
        if isinstance(g.get("by_band"), list):
            rows_b.append([label(d)] + [frac(p) if judged(g) else NA for p in g["by_band"]])
    bands = list((dig(S, "counts", "grid", "bands") or {}).keys())
    x3 = dig(S, "exp3", "token_best", default={})
    x3_line = (f"Experiment 3 reported token match (layer {x3.get('layer', NA)}) with {pct(x3.get('accept_in_range'))} of the in-range "
               f"poses and {pct(x3.get('accept_far'))} of the far poses accepted, AUROC {dec3(x3.get('auroc'))}, limit "
               f"{num(x3.get('tau'))} from the training video frames.") if x3 else ""
    head = ["detector", "far vs close", "turning", "in range accepted", "far accepted", "turned rejected", "AUROC, in range vs far", "limit"]
    full_rule = [f"{k}: {v}." for k, v in rule_items(S.get("rule") or {}, S.get("counts") or {})]
    out.append(block("Every detector, and the rule in full",
                     table(head + ["limit from"], rows),
                     note("The rule, fixed before any pose was scored (“pairs” = AUROC, in range vs far):") + bullets(full_rule) if full_rule and rows else "",
                     note("The limit is the split-conformal 95% quantile of the reference scores. The same with the limit taken "
                          "from the training video frames:") + table(head, rows_v) if rows_v else "",
                     note("Poses accepted per distance band:") + table(["detector"] + bands, rows_b)
                     if rows_b and all(len(r) == len(bands) + 1 for r in rows_b) else "",
                     bullets([x3_line])))

    # ridge and eigenvalues
    R = S.get("ridge") or {}
    if R:
        idn, cv, dim = R.get("identity") or {}, R.get("cv") or {}, dig(S, "feature", "dim")
        of = f" of {dim}" if fin(dim) else ""
        items = [f"Ridge {num(R.get('value'))} added to the diagonal, statistics and factorisation in {R.get('dtype', NA)}"
                 f" ({R.get('factorisation', 'factorisation not recorded')}).",
                 (f"Correlation form (every channel at unit variance, the form used): {span(R.get('corr_eigs_below_ridge') or [])}{of} "
                  f"eigenvalues per token lie below the ridge, {span(R.get('corr_eigs_below_1e_4') or [])} below 1e-4; smallest "
                  f"{num(R.get('corr_eig_min'))}, median {num(R.get('corr_eig_median'))}.") if R.get("corr_eigs_below_ridge") else "",
                 (f"Raw-unit form (the literal recipe): {span(R.get('raw_eigs_below_ridge') or [])}{of} eigenvalues per token lie below the "
                  f"ridge, and {span(R.get('raw_channels_var_below_ridge') or [])} channels have a variance below it; smallest "
                  f"{num(R.get('raw_eig_min'))}, median {num(R.get('raw_eig_median'))}.") if R.get("raw_eigs_below_ridge") else "",
                 (f"In-sample identity check: mean squared Mahalanobis distance {exact(idn.get('mean_in_sample_d2'))} against the closed form "
                  f"{exact(idn.get('expected'))} (n = {num(idn.get('n'))}, D = {num(idn.get('D'))}, covariance {idn.get('cov_dtype', NA)}, "
                  f"factor {idn.get('factor_dtype', NA)}).") if idn else "",
                 (f"Held-out frames: mean squared distance per token {num(R.get('heldout_mean_d2_per_token'))}, "
                  f"{times(R.get('heldout_over_in_sample'))} the in-sample value.") if fin(R.get("heldout_mean_d2_per_token")) else "",
                 (f"Cross-validated ridge (5 folds of training episodes, tokens {listing(cv.get('tokens') or [])}): "
                  f"{num(cv.get('chosen'))}.") if cv else "",
                 (f"Ledoit-Wolf shrinkage weight: {num(dig(R, 'lw_shrinkage', 'video'))} on the training start frames, "
                  f"{num(dig(R, 'lw_shrinkage', 'render'))} on the re-rendered starts.") if R.get("lw_shrinkage") else ""]
        cv_t = table(["ridge", "held-out negative log-likelihood per frame"],
                     [[num(a), num(b, 5)] for a, b in zip(cv.get("grid") or [], cv.get("nll_per_frame") or [])])
        sw = S.get("ridge_sweep") or {}
        sw_dets = [d for d in sw if sw[d]]
        sw_rows = [[num(dig(sw, sw_dets[0], j, "ridge"))] + [c for d in sw_dets for c in (dec3(dig(sw, d, j, "auroc_far")), frac(dig(sw, d, j, "far_at_oracle")))]
                   for j in range(len(sw[sw_dets[0]]))] if sw_dets else []
        sw_t = table(["ridge"] + [c for d in sw_dets for c in (f"{label(d)}: AUROC", "far accepted, limit set to accept all in-range poses but one")], sw_rows)
        out.append(block("The ridge and the eigenvalues", bullets(items), cv_t, sw_t))

    # other noise seeds, served precision
    rows = []
    for v, per in (S.get("replicates") or {}).items():
        for d, e in (per or {}).items():
            g = e.get("grid") or {}
            rows.append([variant_name(v), label(d), verdict_word(g), verdict_word(g, "rotation"), count(g, "in_range"),
                         count(g, "far"), dec3(g.get("auroc_far")), dec3(e.get("spearman_with_primary"))])
    out.append(block("Other noise seeds and the served precision",
                     table(["variant", "detector", "far vs close", "turning", "in range accepted", "far accepted", "AUROC",
                            "rank correlation with the main run"], rows)))

    # what the feature needs
    rows = []
    g0 = dig(D, "paper", "grid")
    if g0 and S.get("ablations"):
        rows.append(["as used: fixed noise, timestep 1000, one statistic per token", dec3(g0.get("auroc_far")), frac(g0.get("far_at_oracle")), dec3(g0.get("rho"))])
    for e in (S.get("ablations") or {}).values():
        g = e.get("grid") or {}
        rows.append([e.get("label", NA), dec3(g.get("auroc_far")), frac(g.get("far_at_oracle")), dec3(g.get("rho"))])
    agg = S.get("aggregation") or {}
    rows_a = [[t, verdict_word(agg[k]), count(agg[k], "in_range"), count(agg[k], "far"), dec3(agg[k].get("auroc_far"))]
              for k, t in (("max", "largest of the 30 token distances (the paper's rule)"), ("mean", "mean of the 30 token distances")) if agg.get(k)]
    hist = [h for h in agg.get("argmax_token_hist") or [] if fin(h)]
    hist_line = (f"The deciding token: token {int(np.argmax(hist))} most often ({max(hist)} of {sum(hist)} poses); "
                 f"{sum(1 for h in hist if h)} of {len(hist)} tokens decide at least once.") if hist and sum(hist) else ""
    out.append(block("What the feature needs",
                     table(["feature", "AUROC, in range vs far", "far accepted, limit set to accept all in-range poses but one",
                            "rank correlation with distance"], rows),
                     table(["over the tokens", "far vs close", "in range accepted", "far accepted", "AUROC"], rows_a), bullets([hist_line])))

    # batch size
    rows = [[label(d), frac(dig(e, "batch16", "decisions_changed")) if limited(d) else NA, change(dig(e, "batch16", "median_log_ratio")),
             dec3(dig(e, "batch16", "spearman")), verdict_word(dig(e, "batch16", "grid"))] for d, e in D.items() if e.get("batch16")]
    out.append(block("Batch size (16 views per VLM pass instead of 1)",
                     table(["detector", "decisions changed", "median score", "rank correlation", "far vs close"], rows)))

    # repeat renders
    rows = []
    for d, e in D.items():
        sc = [v for v in dig(e, "grid", "anchor_scores", default=[]) if fin(v)]
        if not sc:
            continue
        t = tau_of(d)
        ok = fin(t) and t > 0
        rows.append([label(d), count(e.get("grid"), "anchors"), f"{times(min(sc) / t)} to {times(max(sc) / t)}" if ok else NA,
                     times(max(sc) / min(sc)) if min(sc) > 0 else NA])
    # (newer fits) the same scene with the robot moved, and the training starts in their own scenes
    rows_m, rows_q = [], []
    for d, e in D.items():
        rs, sr = dig(e, "grid", "rel_start"), e.get("start_render") or {}
        if isinstance(rs, dict) and isinstance(rs.get("by_band"), list) and len(rs["by_band"]) == len(bands):
            rows_m.append([label(d)] + [times(v) for v in rs["by_band"]])
        if fin(sr.get("q05")) and fin(sr.get("q95")):
            rows_q.append([label(d), rel(sr.get("q05"), d), rel(sr.get("median"), d), rel(sr.get("q95"), d), times(sr.get("spread_q95_over_q05"))])
    out.append(block("Repeat renders of the same pose",
                     table(["detector", "accepted", "score ÷ limit, lowest to highest", "highest ÷ lowest"], rows),
                     note("The same scene with the robot moved: median score ÷ the score at this scene’s own training start, per "
                          "distance band:") + table(["detector"] + bands, rows_m) if rows_m else "",
                     note("The same kind of pose in different scenes: the re-rendered training starts, each in its own scene "
                          "(score ÷ limit):") + table(["detector", "5th percentile", "median", "95th percentile", "95th ÷ 5th"], rows_q)
                     if rows_q else ""))

    # new scenes
    rows = []
    for d, e in D.items():
        u = e.get("unused")
        if not u:
            continue
        ci = u.get("ci") or []
        rows.append([label(d), frac(u.get("accepted")), pct(u.get("accepted")),
                     f"{pct(ci[0])} to {pct(ci[1])}" if len(ci) == 2 and fin(ci[0]) and fin(ci[1]) else NA,
                     rel(u.get("median"), d)] if limited(d) else [label(d), NA, NA, NA, NA])
    out.append(block("New scenes (starts of recordings that were not trained on)",
                     table(["detector", "accepted", "share", "95% interval", "median score ÷ limit"], rows)))

    # the extra poses
    heads = [d for d in main if dig(D, d, "extra")]
    items: dict = {}
    for d in heads:
        for kind, e in D[d]["extra"].items():
            for it in e.get("items") or []:
                items.setdefault(it.get("id"), dict(kind=kind, it=it, per={}))["per"][d] = it
    rows = [[where_text(v["it"]), v["kind"], outside_text(v["it"])]
            + [NA if d not in v["per"] or not limited(d) else ("accepted" if v["per"][d].get("accepted") else "rejected")
               + ("" if rel(v["per"][d].get("score"), d) == NA else f", {rel(v['per'][d].get('score'), d)}") for d in heads]
            for _, v in sorted(items.items(), key=lambda kv: str(kv[0]))]
    out.append(block(f"The {len(rows)} extra poses",
                     table(["pose", "kind", "where"] + [f"{SHORT[d]} (score ÷ limit)" for d in heads], rows)))

    # other tasks, swap
    short = lambda p: re.sub(r"^G1Wholebody|-v\d+$", "", str(p))
    video = lambda d: fin(dig(D, d, "tau", "video", default=tau_of(d)))  # the limit from the training video frames exists
    rows = [[label(d), frac(dig(e, "other_task", "accepted_video_tau")) if video(d) else NA,
             frac(dig(e, "other_task", "accepted_render_tau")) if limited(d) else NA,
             ("; ".join(f"{short(k)} {frac(v)}" for k, v in dig(e, "other_task", "by_pack", default={}).items()) or NA) if video(d) else NA]
            for d, e in D.items() if e.get("other_task")]
    rows_s = [[label(d), num(dig(e, "swap", "n")), lifted(dig(e, "swap", "log_rise_state_swapped")),
               lifted(dig(e, "swap", "log_rise_image_swapped")), lifted(dig(e, "swap", "log_rise_both"))]
              for d, e in D.items() if e.get("swap")]
    out.append(block("Other tasks' frames, and image and state swapped",
                     note("Frames of other tasks' datasets with a faucet start state; all of them should be rejected.") if rows else "",
                     table(["detector", "accepted, limit from the training video frames", "accepted, limit from the re-rendered starts",
                            "by task (video-frame limit)"], rows),
                     note("A start frame’s score when one input is replaced by the one from late in the same demo (median, relative to "
                          "the start frame):") + table(["detector", "episodes", "late state, start image", "late image, start state", "both late"], rows_s)
                     if rows_s else ""))

    # frames during a demo
    rows, n_ph = [], 0
    for d, e in D.items():
        ph = dig(e, "train", "accepted_by_phase")
        if not ph:
            continue
        n_ph = max(n_ph, len(ph))
        g = e.get("grid_demo_tau") or {}
        rows.append([label(d)] + [pct(p) if fin(dig(e, "tau", "video")) else NA for p in ph]
                    + [num(dig(e, "train", "demo_tau")), count(g, "in_range"), count(g, "far")])
    rows = [r for r in rows if len(r) == n_ph + 4]
    out.append(block("Frames during a demo",
                     note("Training frames accepted (each scored with its own episode left out, limit from the training start frames), "
                          "by tenth of the demo; then the grid judged with a limit that covers whole demos:") if rows else "",
                     table(["detector"] + [f"{k * 100 // n_ph}–{(k + 1) * 100 // n_ph}%" for k in range(n_ph)]
                           + ["whole-demo limit", "in range accepted", "far accepted"], rows)))

    # the three H.264 round trips
    rows, rows_s = [], []
    for d in main:
        for key, what in CODECS:
            h = dig(D, d, key)
            if not isinstance(h, dict):
                continue
            rows.append([label(d), what, frac(h.get("decisions_changed")) if limited(d) and fin(h.get("tau")) else NA,
                         frac(h.get("decisions_changed_at_raw_tau")) if limited(d) else NA,
                         change(h.get("median_log_ratio")), dec3(h.get("spearman")), count(h.get("grid"), "far"), verdict_word(h.get("grid"))])
            if h.get("starts"):
                rows_s.append([label(d), what, rel(dig(h, "starts", "median_video"), d), rel(dig(h, "starts", "median_render"), d),
                               rel(dig(h, "starts", "median_render_codec"), d)])
    out.append(block("Video compression: the three round trips",
                     table(["detector", "round trip", "decisions flipped", "flipped at the raw limit", "median score", "rank correlation",
                            "far accepted after", "far vs close after"], rows),
                     note("The training starts, median score ÷ the raw limit:") + table(["detector", "round trip", "training video frame",
                                                                                           "re-render", "re-render after the round trip"], rows_s)
                     if rows_s else ""))

    # earlier trials, every set and part
    rows, within = [], False
    for d in main:
        for name, parts in (dig(D, d, "trials") or {}).items():
            for part, t in (parts or {}).items():
                if not isinstance(t, dict):
                    continue
                inside = [dec3(dig(t, k, "auroc")) for k in ("auroc_within_scene", "auroc_within_condition")]  # newer fits
                within = within or any(v != NA for v in inside)
                rows.append([label(d), TRIAL_SETS.get(name, name), part, num(t.get("n")), num(t.get("successes")),
                             " · ".join(frac(x) for x in t.get("thirds") or []) or NA, dec3(t.get("auroc"))] + inside)
    head = ["detector", "set", "starts", "trials", "successes", "successes in the lowest · middle · highest third of the score",
            "AUROC (a success scores below a failure)"]
    out.append(block("Earlier trials: every set and part",
                     table(head + ["AUROC inside one scene", "AUROC inside one start condition"], rows) if within
                     else table(head, [r[:len(head)] for r in rows])))

    # hook check
    hk = S.get("hook_check") or {}
    rows = [[plain(c.get("check", NA)), "ok" if c.get("ok") else "FAILED", plain(c.get("detail", ""))] for c in hk.get("hard") or [] if isinstance(c, dict)]
    out.append(block("The hook check", table(["check", "result", "detail"], rows, wrap_last=True), bullets([plain(n) for n in hk.get("notes") or []])))

    body = "".join(out)
    if links and body:
        body += "<p class=\"files\">Files: " + " · ".join(f'<a href="{esc(h)}">{esc(h)}</a>' for h in links) + "</p>"
    return body


# ------------------------------------------------------------------------------------------------ the page
HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')

CSS = r"""<style>
/* layout: one column of findings, each a one-line conclusion over one simple chart
   colour, decisions: blue = accepted, orange = a far pose wrongly accepted, grey = rejected, context
   colour, detectors: violet = paper LLMD, green / magenta = start-referenced, neutral grey (dashed line) = token match */
:root{
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --ring:rgba(11,11,11,.10); --shadow:0 6px 24px rgba(11,11,11,.14);
  --accept:#2a78d6; --wrong:#eb6834; --rej:#8e8c84; --context:#9a988f;
  --floor:#fcfcfb; --wall:#d5d3c9; --table:#efeee9;
  --d1:#6250d6; --d2:#008300; --d3:#e87ba4; --d4:#77756e;
  --t1:#1c5cab; --t2:#5598e7; --t3:#86b6ef;
  --pass-bg:#dbe9fb; --pass-ink:#184f95; --fail-bg:#fbe3d8; --fail-ink:#9c3a16; --mid-bg:#eceae4; --mid-ink:#52514e;
  --font:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --ring:rgba(255,255,255,.10); --shadow:0 6px 24px rgba(0,0,0,.5);
  --accept:#3987e5; --wrong:#d95926; --rej:#6f6e68; --context:#77766f;
  --floor:#1a1a19; --wall:#3d3d3a; --table:#262624;
  --d1:#9085e9; --d2:#0ca30c; --d3:#d55181; --d4:#a3a199;
  --t1:#b7d3f6; --t2:#5598e7; --t3:#1c5cab;
  --pass-bg:#17335a; --pass-ink:#b7d3f6; --fail-bg:#4a2313; --fail-ink:#f3a583; --mid-bg:#2c2c2a; --mid-ink:#c3c2b7;
  color-scheme:dark } }
:root[data-theme="dark"]{
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --ring:rgba(255,255,255,.10); --shadow:0 6px 24px rgba(0,0,0,.5);
  --accept:#3987e5; --wrong:#d95926; --rej:#6f6e68; --context:#77766f;
  --floor:#1a1a19; --wall:#3d3d3a; --table:#262624;
  --d1:#9085e9; --d2:#0ca30c; --d3:#d55181; --d4:#a3a199;
  --t1:#b7d3f6; --t2:#5598e7; --t3:#1c5cab;
  --pass-bg:#17335a; --pass-ink:#b7d3f6; --fail-bg:#4a2313; --fail-ink:#f3a583; --mid-bg:#2c2c2a; --mid-ink:#c3c2b7;
  color-scheme:dark }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--font);-webkit-font-smoothing:antialiased}
.page{max-width:1120px;margin:0 auto;padding-block:36px 72px;padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));display:flex;flex-direction:column;gap:60px}
.page>*{min-width:0}
p{margin:0}
.eyebrow{font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:10px}
h1{font-size:clamp(25px,3.7vw,36px);line-height:1.13;letter-spacing:-.02em;font-weight:680;margin:0;text-wrap:balance;max-width:34ch}
.lede{color:var(--ink2);max-width:86ch;margin-top:16px}
.cards{display:grid;grid-template-columns:repeat(var(--n,4),minmax(0,1fr));gap:14px;margin-top:28px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 14px 13px;min-width:0;
  display:grid;grid-template-rows:subgrid;grid-row:span 4;gap:10px}
.cn{font-weight:650;font-size:14px;line-height:1.3;display:flex;gap:8px;align-items:baseline}
.cn .k{transform:translateY(1px)}
.cd{font-size:12.5px;color:var(--ink2);line-height:1.35;margin-top:-5px}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.vchip{display:inline-flex;align-items:center;font-size:12.5px;font-weight:650;padding:3px 10px;border-radius:999px;white-space:nowrap}
.vchip.sm{font-weight:550;font-size:12px;padding:2px 9px}
.v-pass{background:var(--pass-bg);color:var(--pass-ink)} .v-fail{background:var(--fail-bg);color:var(--fail-ink)}
.v-ranks,.v-na{background:var(--mid-bg);color:var(--mid-ink)}
.cs{margin:0;display:grid;gap:6px;align-content:start}
.cs div{display:grid;grid-template-columns:minmax(78px,auto) minmax(0,1fr);gap:10px;align-items:baseline}
.cs dt{font-size:19px;font-weight:680;letter-spacing:-.01em;white-space:nowrap;line-height:1.2}
.cs dt small{font-size:13px;font-weight:520;color:var(--ink2);letter-spacing:0}
.cs dd{margin:0;font-size:12.5px;color:var(--ink2);line-height:1.3}
.rule{font-size:12.5px;color:var(--ink2);margin-top:16px;max-width:118ch}
.notice{font-size:13px;color:var(--fail-ink);background:var(--fail-bg);border-radius:8px;padding:9px 12px;margin-top:14px;max-width:86ch}
h2{font-size:clamp(20px,2.7vw,27px);line-height:1.2;letter-spacing:-.016em;margin:0;font-weight:660;text-wrap:balance;max-width:44ch}
.figs{display:grid;gap:14px;margin-top:22px}
.figs.c3.lead{grid-template-columns:minmax(0,1.26fr) minmax(0,1fr) minmax(0,1fr)}
.figs.c1{grid-template-columns:minmax(0,880px)}
.fig{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 14px 12px;min-width:0;
  display:flex;flex-direction:column}
.fig h3{font-size:13px;font-weight:600;margin:0 0 10px;line-height:1.35;text-wrap:balance}
.split{display:grid;grid-template-columns:minmax(0,440px) minmax(0,1fr);gap:8px 26px;align-items:start}
.side{display:grid;gap:12px;font-size:13px;color:var(--ink2);align-content:start;padding-top:6px}
.side .legend{margin-top:0;flex-direction:column;align-items:flex-start;gap:7px}
.side p:empty{display:none}
.sub{font-size:12.5px;color:var(--ink2);margin-top:12px;max-width:100ch}
.sub+.sub{margin-top:4px}
.na{font-size:13px;color:var(--ink2);padding:22px 2px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--ink2);margin-top:12px;align-items:center}
.legend span{display:inline-flex;align-items:center;gap:7px}
.legend[hidden]{display:none}
.k{display:inline-block;flex:none;--c:var(--ink2)}
.k.sq{width:11px;height:11px;border-radius:2.5px;background:var(--c)}
.k.dot{width:5px;height:5px;border-radius:50%;background:var(--rej);margin:0 3px}
.k.pt{width:8px;height:8px;border-radius:50%;background:var(--c);margin:0 1.5px}
.k.ring{width:8px;height:8px;border-radius:50%;border:1.4px solid var(--context);margin:0 1.5px}
.k.vdash{width:0;height:12px;border-left:1.5px dashed var(--ink2);margin:0 5px}
.k.box{width:13px;height:10px;border:1.6px solid var(--ink)}
.k.ln{width:18px;height:0;border-top:2px solid var(--c)}
.k.ln.hero{border-top-width:3px}
.k.ln.ref{border-top-style:dashed}
.k.zone{width:14px;height:11px;border-radius:2px;background:var(--wrong);opacity:.3}
.k.fau{width:9px;height:9px;border-radius:50%;border:1.6px solid var(--ink);background:var(--surface)}
.seg{display:flex;flex-wrap:wrap;gap:6px;margin-top:16px}
.seg button{font:inherit;font-size:13px;line-height:1.2;color:var(--ink2);background:transparent;border:1px solid var(--axis);border-radius:999px;
  padding:6px 13px 6px 10px;display:inline-flex;align-items:center;gap:7px;cursor:pointer}
.seg button:hover{color:var(--ink);border-color:var(--ink2)}
.seg button[aria-pressed="true"]{color:var(--ink);font-weight:600;border-color:var(--ink);background:var(--surface);box-shadow:inset 0 0 0 1px var(--ink)}
.seg button:focus-visible{outline:2px solid var(--accept);outline-offset:2px}
svg{display:block;max-width:100%;overflow:visible}
svg text{font-family:var(--font);fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
svg.map{border-radius:6px;cursor:crosshair;touch-action:pan-y}
svg:focus-visible{outline:2px solid var(--accept);outline-offset:3px;border-radius:6px}
.m-floor{fill:var(--floor)} .m-wall{fill:var(--wall)} .m-table{fill:var(--table);stroke:var(--axis);stroke-width:1}
.m-ring{fill:none;stroke:var(--grid);stroke-width:1}
.m-box{fill:none;stroke:var(--ink);stroke-width:1.6}
.m-fau{fill:var(--surface);stroke:var(--ink);stroke-width:1.6}
.c-acc{fill:var(--accept)} .c-wrong{fill:var(--wrong)} .c-rej{fill:var(--rej)}
.c-ctx{fill:none;stroke:var(--context);stroke-width:1.3}
.startl{stroke:var(--ink2);stroke-width:1.3;stroke-dasharray:4 3}
.a-acc{stroke:var(--accept);fill:var(--accept);stroke-width:2.4} .a-rej{stroke:var(--rej);fill:var(--rej);stroke-width:1.5}
.a-wrong{stroke:var(--wrong);fill:var(--wrong);stroke-width:2.4}
.hl{fill:none;stroke:var(--ink);stroke-width:1.8;pointer-events:none}
.gridl{stroke:var(--grid);stroke-width:1} .axisl{stroke:var(--axis);stroke-width:1}
.limit{stroke:var(--ink);stroke-width:1.5}
.diag{stroke:var(--muted);stroke-width:1}
.trainband{fill:var(--ink);opacity:.07}
.zone{fill:var(--wrong);opacity:.13}
.blocked{fill:var(--table);stroke:var(--axis);stroke-width:1}
.brace{fill:none;stroke:var(--ink2);stroke-width:1}
.ln{fill:none;stroke-width:1.6;stroke-linejoin:round;stroke-linecap:round;opacity:.85}
.ln.hero{stroke-width:2.6;opacity:1} .ln.ref{stroke-dasharray:5 4;opacity:1}
.pt{stroke:var(--surface);stroke-width:2}
.pt.sm{stroke-width:1.2}
.cross{stroke:var(--ink2);stroke-width:1;pointer-events:none}
.tick-strong{fill:var(--ink)} .tick-muted{fill:var(--muted)}
.halo{paint-order:stroke;stroke:var(--surface);stroke-width:3.5px;stroke-linejoin:round}
.b-d1{fill:var(--d1)} .b-d2{fill:var(--d2)} .b-d3{fill:var(--d3)} .b-d4{fill:var(--d4)}
.b-t1{fill:var(--t1)} .b-t2{fill:var(--t2)} .b-t3{fill:var(--t3)}
.also{margin-top:20px}
details{border-top:1px solid var(--grid);padding:13px 0;font-size:13.5px;color:var(--ink2);min-width:0}
details:last-of-type{border-bottom:1px solid var(--grid)}
summary{cursor:pointer;color:var(--ink);font-weight:600;font-size:14px}
summary:focus-visible{outline:2px solid var(--accept);outline-offset:3px;border-radius:4px}
details .defs{display:grid;gap:7px;margin:12px 0 0;padding-left:18px;max-width:100ch}
details .cap{margin-top:14px;max-width:100ch}
.tw{overflow-x:auto;margin-top:12px;max-width:100%}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--grid);padding:5px 16px 5px 0;text-align:left;white-space:nowrap;vertical-align:top}
th{color:var(--muted);font-weight:500;white-space:normal;min-width:64px;max-width:170px;vertical-align:bottom}
td.wrap{white-space:normal;min-width:260px;max-width:620px;overflow-wrap:anywhere}
.files{margin-top:16px;font-size:13px;color:var(--ink2)}
.files a{color:var(--ink);text-underline-offset:3px}
.tip{position:fixed;z-index:20;pointer-events:none;background:var(--surface);border:1px solid var(--ring);border-radius:9px;
  box-shadow:var(--shadow);overflow:hidden;max-width:232px;font-size:12.5px;line-height:1.4}
.tip img{display:block;width:232px;height:auto;aspect-ratio:16/9}
.tip .tb{padding:9px 11px 10px}
.tip .tt{font-weight:600;color:var(--ink)}
.tip .ts{color:var(--muted)}
.tip .tr{display:flex;align-items:center;gap:7px;color:var(--ink2);margin-top:3px}
.tip .tr b{color:var(--ink);font-weight:600;white-space:nowrap}
@media (max-width:980px){ .cards{grid-template-columns:repeat(2,minmax(0,1fr));max-width:none!important} .figs.c3.lead{grid-template-columns:minmax(0,1fr)} }
@media (max-width:620px){ .split{grid-template-columns:minmax(0,1fr)} .page{gap:48px;padding-block:24px 56px}
  td:first-child{white-space:normal;min-width:128px} th,td{padding-right:12px} }
@media (max-width:520px){ .cards{grid-template-columns:minmax(0,1fr)} }
</style>
"""

JS = r"""<script>
(() => {
'use strict';
const D = JSON.parse(document.getElementById('D').textContent), P = D.poses || [], NS = 'http://www.w3.org/2000/svg';
const S = (tag, at, par) => { const e = document.createElementNS(NS, tag); for (const k in at || {}) e.setAttribute(k, at[k]); if (par) par.appendChild(e); return e; };
const H = (tag, cls, txt, par) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; if (par) par.appendChild(e); return e; };
const f1 = v => v.toFixed(1), pct = v => Math.round(v * 100) + '%', cap = t => t.charAt(0).toUpperCase() + t.slice(1);
const DET = {}; for (const d of D.dets) DET[d.k] = d;
const xr = r => (r >= 100 ? Math.round(r).toLocaleString('en-US') : r >= 10 ? r.toFixed(0) : r.toFixed(2)) + ' ×';
const sel = Object.assign({map: D.hero, why: D.hero, h264: D.hero, trials: D.hero}, D.first || {});
const key = (par, shape, c, mod) => { const i = H('i', 'k ' + shape + (mod ? ' ' + mod : ''), null, par); if (c) i.style.setProperty('--c', `var(--${c})`); return i; };
const say = (par, t) => par.appendChild(document.createTextNode(t));
const wrap = (s, n) => { const out = []; let cur = ''; for (const w of s.split(' ')) { if (cur && (cur + ' ' + w).length > n) { out.push(cur); cur = w; } else cur = cur ? cur + ' ' + w : w; } if (cur) out.push(cur); return out; };
const setText = (id, t) => { const e = document.getElementById(id); if (e && t != null) e.textContent = t; };
const lineMod = d => d.k === D.hero ? 'hero' : d.ref ? 'ref' : '';

/* ---- ticks of a log axis from lo to hi: decades, 1-2-5, or even steps when the span is small */
function logTicks(lo, hi) {
  const dec = Math.log10(hi / lo), out = [];
  if (dec < .45) { const step = [.01, .02, .05, .1, .2, .5, 1, 2, 5].find(s => (hi - lo) / s <= 6) || 10;
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi * (1 + 1e-9); v += step) out.push(+v.toPrecision(6)); }
  else { const mult = dec > 2.4 ? [1] : dec > 1 ? [1, 2, 5] : [1, 1.5, 2, 3, 5, 7], every = dec > 7 ? 2 : 1;
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const q of mult) {
      const v = +(q * Math.pow(10, e)).toPrecision(6); if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9) && !(every === 2 && e % 2)) out.push(v); } }
  if (lo <= 1 && hi >= 1 && !out.includes(1)) out.push(1);
  return out;
}
const tickText = v => (v >= 1e5 || v < 1e-4) ? '1e' + Math.round(Math.log10(v)) : String(+v.toPrecision(3));

/* ---- one tooltip for the page */
const tip = H('div', 'tip', null, document.body); tip.hidden = true;
function showTip(x, y, c) {
  tip.textContent = '';
  if (c.img) { const im = H('img', null, null, tip); im.src = c.img; im.alt = ''; }
  const b = H('div', 'tb', null, tip);
  if (c.title) H('div', 'tt', c.title, b);
  if (c.sub) H('div', 'ts', c.sub, b);
  for (const r of c.rows || []) {
    const d = H('div', 'tr', null, b);
    if (r.c) key(d, r.shape || 'sq', r.c, r.mod);
    H('b', null, r.b, d); if (r.t) H('span', null, r.t, d);
  }
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
  let l = x + 16, t = y + 16;
  if (l + w > vw - 8) l = x - w - 16; if (l < 8) l = 8;
  if (t + h > vh - 8) t = y - h - 16; if (t < 8) t = 8;
  tip.style.left = l + 'px'; tip.style.top = t + 'px';
}
const hideTip = () => { tip.hidden = true; };
window.addEventListener('scroll', hideTip, {passive: true});

/* ---- what a pose says for a detector */
const clsOf = (i, k) => D.a[k][i] ? (P[i].far ? 'wrong' : 'acc') : 'rej';
const verdict = (i, k) => { const c = clsOf(i, k); return c === 'wrong' ? {c: 'wrong', b: 'accepted', t: 'but ' + D.meta.far_text} : c === 'acc' ? {c: 'accept', b: 'accepted'} : {c: 'rej', b: 'rejected'}; };
const ratio = (i, k) => { const r = D.r[k][i]; return r == null ? {b: 'no limit', t: 'for this score'} : {b: xr(r), t: 'the limit'}; };
const poseTip = (i, k) => ({img: D.thumbs[P[i].id], title: P[i].where, sub: P[i].dist, rows: [verdict(i, k), ratio(i, k)]});

/* ---- a detector switch */
function seg(name, keys, onPick) {
  const host = document.querySelector(`[data-seg="${name}"]`); if (!host) return;
  if (keys.length && !keys.includes(sel[name])) sel[name] = keys[0];
  if (keys.length < 2) { host.remove(); return; }
  const btns = keys.map(k => { const b = H('button', null, null, host); b.type = 'button'; key(b, 'sq', DET[k].c); say(b, DET[k].name);
    b.setAttribute('aria-pressed', String(k === sel[name]));
    b.addEventListener('click', () => { sel[name] = k; btns.forEach((x, j) => x.setAttribute('aria-pressed', String(keys[j] === k))); onPick(k); });
    return b; });
}

/* ---- bars with a rounded top on a baseline */
const bar = (svg, x, base, w, h, cls) => { const hh = Math.max(2, h), top = base - hh, rr = Math.min(4, hh, w / 2);
  S('path', {d: `M${f1(x)} ${f1(base)}V${f1(top + rr)}q0 ${-rr} ${rr} ${-rr}H${f1(x + w - rr)}q${rr} 0 ${rr} ${rr}V${f1(base)}z`, class: cls}, svg); return top; };

/* ---- far vs close: share accepted per distance band, one bar per detector */
function drawBands(host) {
  const W = host.clientWidth; if (!W) return;
  const B = D.bands, K = D.dets.filter(d => B.by[d.k]), G = B.names.length, m = {l: 38, r: 4, t: 54, b: 0};
  const gw = (W - m.l - m.r) / G, lines = B.names.map(n => wrap(n, Math.max(7, Math.floor((gw - 8) / 6))));
  m.b = 20 + 13 * Math.max(...lines.map(l => l.length)) + 18;
  const ph = Math.round(Math.max(170, Math.min(240, W * .3))), Hh = m.t + ph + m.b, base = m.t + ph, y = v => base - ph * v;
  const bw = Math.max(5, Math.min(24, (gw - 20) / K.length - 2)), gx = g => m.l + gw * g + (gw - K.length * (bw + 2) + 2) / 2;
  host.textContent = '';
  const svg = S('svg', {width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: 'img', tabindex: 0,
                        'aria-label': 'Share of poses accepted, by distance outside the training range, one bar per detector. Arrow keys move between the groups.'}, host);
  const ideal = (g0, g1, text) => { const xa = m.l + gw * g0 + 6, xb = m.l + gw * g1 - 6, tw = text.length * 5.7;
    S('path', {d: `M${f1(xa)} ${m.t - 22}v-4H${f1(xb)}v4`, class: 'brace'}, svg);
    S('text', {x: Math.max(tw / 2 + 2, Math.min(W - tw / 2 - 2, (xa + xb) / 2)), y: m.t - 33, 'text-anchor': 'middle', class: 'tick-strong'}, svg).textContent = text; };
  ideal(0, 1, 'should be full'); if (B.far_from < G) ideal(B.far_from, G, gw * (G - B.far_from) > 130 ? 'far poses: should be empty' : 'should be empty');
  for (const v of [0, .5, 1]) { S('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: v ? 'gridl' : 'axisl'}, svg);
    S('text', {x: m.l - 7, y: y(v) + 4, 'text-anchor': 'end'}, svg).textContent = pct(v); }
  const labels = [], hi = K.findIndex(d => d.k === D.hero);
  B.names.forEach((n, g) => {
    K.forEach((d, j) => { const kn = B.by[d.k][g]; if (!kn || !kn[1]) return;
      const v = kn[0] / kn[1], x = gx(g) + j * (bw + 2), top = bar(svg, x, base, bw, ph * v, 'b-' + d.c);
      if (d.k === D.hero || (d.ref && Math.abs(j - hi) * (bw + 2) >= 34)) labels.push([x + bw / 2, top - 5, pct(v), d.k === D.hero]); });
    lines[g].forEach((t, j) => { S('text', {x: m.l + gw * (g + .5), y: base + 17 + 13 * j, 'text-anchor': 'middle'}, svg).textContent = t; });
    S('text', {x: m.l + gw * (g + .5), y: base + 19 + 13 * lines[g].length, 'text-anchor': 'middle', class: 'tick-muted'}, svg).textContent = B.n[g] + ' poses';
  });
  for (const [x, yy, t, hero] of labels) S('text', {x, y: yy, 'text-anchor': 'middle', class: 'halo' + (hero ? ' tick-strong' : ''), 'font-weight': hero ? 650 : 400}, svg).textContent = t;
  const hl = S('rect', {class: 'hl', visibility: 'hidden', rx: 4}, svg);
  let cur = -1;
  const show = (g, px, py) => { cur = g;
    for (const [a, val] of Object.entries({x: m.l + gw * g + 1, y: m.t - 14, width: gw - 2, height: ph + 16, visibility: 'visible'})) hl.setAttribute(a, val);
    showTip(px, py, {title: cap(B.names[g]), sub: B.n[g] + ' poses, accepted:', rows: K.map(d => { const kn = B.by[d.k][g];
      return {c: d.c, b: kn && kn[1] ? `${kn[0]} of ${kn[1]}` : 'n/a', t: d.name}; })}); };
  const off = () => { cur = -1; hl.setAttribute('visibility', 'hidden'); hideTip(); };
  const move = e => { const b = svg.getBoundingClientRect(); show(Math.max(0, Math.min(G - 1, Math.floor((e.clientX - b.left - m.l) / gw))), e.clientX, e.clientY); };
  svg.addEventListener('pointermove', move); svg.addEventListener('pointerdown', move); svg.addEventListener('pointerleave', off); svg.addEventListener('blur', off);
  svg.addEventListener('keydown', e => { const d = {ArrowLeft: -1, ArrowRight: 1}[e.key]; if (e.key === 'Escape') return off(); if (!d) return; e.preventDefault();
    const g = Math.max(0, Math.min(G - 1, cur < 0 ? 0 : cur + d)), b = svg.getBoundingClientRect(); show(g, b.left + m.l + gw * (g + .5), b.top + m.t + 20); });
}

/* ---- top-down map: filled square = accepted, dot = rejected, arrows for the poses that face the faucet */
function drawMap(host) {
  const W = host.clientWidth; if (!W) return;
  const k = sel.map, v = D.map.views[host.dataset.view], set = D.sets[host.dataset.sel];
  const arrows = host.dataset.sel === 'facing', s = W / (v.l - v.r), Hm = s * (v.t - v.b), foot = 26;
  const X = m => (v.l - m) * s, Y = m => (v.t - m) * s, id = 'c' + Math.random().toString(36).slice(2, 8);
  host.textContent = '';
  const svg = S('svg', {width: W, height: Hm + foot, viewBox: `0 0 ${W} ${Hm + foot}`, class: 'map', role: 'img', tabindex: 0,
                        'aria-label': host.parentNode.querySelector('h3').textContent + '. Arrow keys move between poses.'}, host);
  S('rect', {width: W, height: Hm, rx: 6}, S('clipPath', {id}, S('defs', null, svg)));
  const g = S('g', {'clip-path': `url(#${id})`}, svg);
  S('rect', {width: W, height: Hm, class: 'm-floor'}, g);
  let d = '';
  for (const r of D.map.walls) {
    const x = X(r[0]), y = Y(r[1]), w = r[2] * s, h = r[3] * s;
    if (x + w < 0 || x > W || y + h < 0 || y > Hm) continue;
    d += `M${f1(x)} ${f1(y)}h${f1(w)}v${f1(h)}h${f1(-w)}z`;
  }
  if (d) S('path', {d, class: 'm-wall'}, g);
  const tb = D.map.table.map(q => [X(q[0]), Y(q[1])]);
  S('polygon', {points: tb.map(q => f1(q[0]) + ',' + f1(q[1])).join(' '), class: 'm-table'}, g);
  const tx = tb.reduce((a, q) => a + q[0], 0) / 4, ty = tb.reduce((a, q) => a + q[1], 0) / 4;
  S('text', {x: Math.max(24, Math.min(W - 24, tx)), y: Math.max(16, Math.min(Hm - 10, ty)) + 4, 'text-anchor': 'middle'}, g).textContent = 'table';
  const order = set.slice().sort((a, b) => P[b].c - P[a].c);
  if (arrows) {
    for (const i of order) {
      const p = P[i], a = p.yaw * Math.PI / 180, L = Math.max(15, .3 * s), dx = -Math.sin(a), dy = -Math.cos(a), cx = X(p.x), cy = Y(p.y);
      const x2 = cx + dx * L / 2, y2 = cy + dy * L / 2, hd = Math.max(4.5, L * .3), c = clsOf(i, k);
      S('line', {x1: cx - dx * L / 2, y1: cy - dy * L / 2, x2: x2 - dx * hd * .6, y2: y2 - dy * hd * .6, class: 'a-' + c, 'stroke-linecap': 'round'}, g);
      S('polygon', {points: [[x2, y2], [x2 - dx * hd - dy * hd * .5, y2 - dy * hd + dx * hd * .5], [x2 - dx * hd + dy * hd * .5, y2 - dy * hd - dx * hd * .5]]
                    .map(q => f1(q[0]) + ',' + f1(q[1])).join(' '), class: 'a-' + c, 'stroke-width': 0}, g);
    }
  } else {
    for (const pass of ['rej', 'acc', 'wrong']) for (const i of order) {
      const p = P[i], c = clsOf(i, k); if (c !== pass) continue;
      const cs = p.c * s, cx = X(p.x), cy = Y(p.y), gap = Math.min(1.2, cs * .12);
      if (c === 'rej') S('circle', {cx, cy, r: Math.max(.9, Math.min(2.3, cs * .13)), class: 'c-rej'}, g);
      else S('rect', {x: cx - cs / 2 + gap / 2, y: cy - cs / 2 + gap / 2, width: cs - gap, height: cs - gap, rx: Math.min(2.5, cs * .16), class: 'c-' + c}, g);
    }
  }
  const bx = D.map.box;
  S('rect', {x: X(bx.l), y: Y(bx.t), width: (bx.l - bx.r) * s, height: (bx.t - bx.b) * s, class: 'm-box'}, g);
  S('circle', {cx: X(D.map.faucet[0]), cy: Y(D.map.faucet[1]), r: 4.2, class: 'm-fau'}, g);
  S('rect', {x: .5, y: .5, width: W - 1, height: Hm - 1, rx: 6, class: 'm-ring'}, svg);
  const hl = S('rect', {class: 'hl', visibility: 'hidden', rx: 2}, g);
  const sb = v.l - v.r > 3 ? 1 : .5;
  S('line', {x1: 1, x2: 1 + sb * s, y1: Hm + 13, y2: Hm + 13, class: 'axisl', style: 'stroke:var(--ink2)'}, svg);
  S('text', {x: 7 + sb * s, y: Hm + 17}, svg).textContent = sb + ' m';

  let cur = -1;
  const at = (i, cx, cy) => {
    cur = i; const p = P[i], cs = Math.max(arrows ? 18 : p.c * s, 9);
    for (const [a, val] of Object.entries({x: X(p.x) - cs / 2, y: Y(p.y) - cs / 2, width: cs, height: cs, visibility: 'visible'})) hl.setAttribute(a, val);
    showTip(cx, cy, poseTip(i, sel.map));
  };
  const off = () => { cur = -1; hl.setAttribute('visibility', 'hidden'); hideTip(); };
  const pick = e => {
    const r = svg.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    let best = -1, bd = 1e9;
    for (const i of set) { const dd = Math.hypot(X(P[i].x) - px, Y(P[i].y) - py), lim = Math.max(arrows ? 16 : P[i].c * s * .75, 12); if (dd < lim && dd < bd) { bd = dd; best = i; } }
    if (best < 0 || py > Hm) off(); else at(best, e.clientX, e.clientY);
  };
  svg.addEventListener('pointermove', pick); svg.addEventListener('pointerdown', pick);
  svg.addEventListener('pointerleave', off); svg.addEventListener('blur', off);
  svg.addEventListener('keydown', e => {
    const dir = {ArrowLeft: [1, 0], ArrowRight: [-1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1]}[e.key];
    if (e.key === 'Escape') return off();
    if (!dir) return;
    e.preventDefault();
    let nx = -1;
    if (cur < 0) { let bd = 1e9; for (const i of set) { const dd = Math.hypot(P[i].x, P[i].y); if (dd < bd) { bd = dd; nx = i; } } }
    else { let bd = 1e9; for (const i of set) { const ax = P[i].x - P[cur].x, ay = P[i].y - P[cur].y, al = ax * dir[0] + ay * dir[1], dd = Math.hypot(ax, ay);
      if (al > 1e-6 && al > .5 * dd && dd < bd) { bd = dd; nx = i; } } }
    if (nx < 0) return;
    const r = svg.getBoundingClientRect(); at(nx, r.left + X(P[nx].x), r.top + Y(P[nx].y));
  });
}

/* ---- score / limit along a line of poses: log scale, the limit at 1, the zone where a far pose is still accepted in orange */
function drawLine(host) {
  const W = host.clientWidth; if (!W) return;
  const L = D.lines[host.dataset.line], K = D.dets.filter(d => L.pts.some(p => p.r[d.k] != null));
  host.textContent = '';
  const lg = document.getElementById('line-legend-' + host.dataset.line); if (lg) lg.hidden = !K.length;
  if (!K.length) { H('p', 'na', 'Not available: no limit could be set in this run, so the scores cannot be shown against it.', host); return; }
  const Hh = Math.round(Math.max(260, Math.min(336, W * .4))), m = {l: 40, r: 68, t: 38, b: 46}, ph = Hh - m.t - m.b, x1p = W - m.r, yb = Hh - m.b, dot = W < 480 ? 3 : 4;
  const xs = L.pts.map(p => p.x), x0 = Math.min(...xs), x1 = Math.max(...xs), sq = v => L.stretch ? Math.sign(v) * Math.sqrt(Math.abs(v)) : v;
  const X = v => m.l + (x1p - m.l) * (sq(v) - sq(x0)) / ((sq(x1) - sq(x0)) || 1);
  let lo = 1, hi = 1;
  for (const p of L.pts) for (const d of K) { const r = p.r[d.k]; if (r != null && r > 0) { lo = Math.min(lo, r); hi = Math.max(hi, r); } }
  lo /= 1.25; hi *= 1.25; if (hi / lo < 3) { const f = Math.sqrt(3 / (hi / lo)); lo /= f; hi *= f; }
  const Y = v => m.t + ph * (1 - Math.log(v / lo) / Math.log(hi / lo)), yl = Y(1);
  const svg = S('svg', {width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: 'img', tabindex: 0, 'aria-label': L.aria + ' Arrow keys move between the poses.'}, host);
  if (L.zone) { const a = X(Math.max(L.zone[0], x0)), b = X(Math.min(L.zone[1], x1));
    if (L.zone[0] > x0) S('rect', {x: m.l, y: yl, width: a - m.l, height: yb - yl, class: 'zone'}, svg);
    if (L.zone[1] < x1) S('rect', {x: b, y: yl, width: x1p - b, height: yb - yl, class: 'zone'}, svg); }
  if (L.band) { const a = X(Math.max(L.band[0], x0)), b = X(Math.min(L.band[1], x1));
    if (b > a) { S('rect', {x: a, y: m.t, width: b - a, height: ph, class: 'trainband'}, svg);
      S('text', {x: Math.max(m.l + 42, Math.min(x1p - 42, (a + b) / 2)), y: m.t - 8, 'text-anchor': 'middle', class: 'tick-strong'}, svg).textContent = L.band_label; } }
  for (const v of logTicks(lo, hi)) { if (v === 1) continue;
    S('line', {x1: m.l, x2: x1p, y1: Y(v), y2: Y(v), class: 'gridl'}, svg);
    if (Math.abs(Y(v) - yl) > 9) S('text', {x: m.l - 7, y: Y(v) + 4, 'text-anchor': 'end'}, svg).textContent = tickText(v); }
  if (L.blocked) { const a = X(Math.max(L.blocked[0], x0)), b = X(Math.min(L.blocked[1], x1));
    if (b > a) { S('rect', {x: a, y: m.t, width: b - a, height: ph, class: 'blocked'}, svg); S('text', {x: (a + b) / 2, y: m.t + ph / 2 + 4, 'text-anchor': 'middle'}, svg).textContent = 'table'; } }
  S('text', {x: 2, y: 11}, svg).textContent = 'score ÷ limit';
  S('line', {x1: m.l, x2: x1p, y1: yb, y2: yb, class: 'axisl'}, svg);
  const cand = L.unit === 'deg' ? [0, 180, 90, 45, 20, 135, 10, 5] : [0, 3, 1, .3, .1, 2, .5, .2, 1.5, .05], kept = [];
  for (const a of cand) for (const sgn of a ? [-1, 1] : [1]) { const v = sgn * a; if (v < x0 - 1e-9 || v > x1 + 1e-9) continue;
    const px = X(v); if (kept.some(q => Math.abs(q - px) < 36)) continue; kept.push(px);
    S('line', {x1: px, x2: px, y1: yb, y2: yb + 4, class: 'axisl'}, svg);
    S('text', {x: px, y: yb + 17, 'text-anchor': 'middle'}, svg).textContent = (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v) + (L.unit === 'deg' ? '°' : ''); }
  S('text', {x: (x1p + m.l) / 2, y: Hh - 6, 'text-anchor': 'middle'}, svg).textContent = W > 520 ? L.xlabel : L.xshort;
  const gap = (a, b) => L.blocked && a < L.blocked[1] && b > L.blocked[0];
  const order = K.slice().sort((a, b) => (a.k === D.hero) - (b.k === D.hero) || (b.ref ? 1 : 0) - (a.ref ? 1 : 0));
  for (const d of order) { const hero = d.k === D.hero, runs = []; let prev = null;
    for (const p of L.pts) { const r = p.r[d.k]; if (r == null) { prev = null; continue; }
      if (prev == null || gap(prev, p.x)) runs.push([]); runs[runs.length - 1].push([X(p.x), Y(r)]); prev = p.x; }
    S('path', {d: runs.map(q => q.map((c, j) => (j ? 'L' : 'M') + f1(c[0]) + ' ' + f1(c[1])).join('')).join(''),
               class: 'ln' + (hero ? ' hero' : '') + (d.ref ? ' ref' : ''), style: `stroke:var(--${d.c})`}, svg);
    for (const q of runs) if (hero || q.length === 1) for (const c of q)
      S('circle', {cx: c[0], cy: c[1], r: hero ? dot : 2.6, class: 'pt' + (hero && dot > 3 ? '' : ' sm'), style: `fill:var(--${d.c})`}, svg); }
  S('line', {x1: m.l, x2: x1p, y1: yl, y2: yl, class: 'limit'}, svg);
  S('text', {x: m.l - 7, y: yl + 4, 'text-anchor': 'end', class: 'tick-strong', 'font-weight': 650}, svg).textContent = '1';
  S('text', {x: x1p + 7, y: yl + 4, class: 'tick-strong', 'font-weight': 650}, svg).textContent = 'limit';
  S('text', {x: x1p + 7, y: yl - 12}, svg).textContent = '↑ rejected';
  S('text', {x: x1p + 7, y: yl + 20}, svg).textContent = '↓ accepted';
  const cross = S('line', {y1: m.t, y2: yb, class: 'cross', visibility: 'hidden'}, svg);
  let cur = -1;
  const show = (j, cx, cy) => { cur = j; const p = L.pts[j]; cross.setAttribute('x1', X(p.x)); cross.setAttribute('x2', X(p.x)); cross.setAttribute('visibility', 'visible');
    showTip(cx, cy, {img: D.thumbs[P[p.i].id], title: P[p.i].where, sub: P[p.i].dist, rows: K.map(d => { const r = p.r[d.k];
      return {c: d.c, shape: 'ln', mod: lineMod(d), b: r == null ? 'n/a' : xr(r), t: d.name + (r == null ? '' : r > 1 ? ', rejected' : ', accepted')}; })}); };
  const off = () => { cur = -1; cross.setAttribute('visibility', 'hidden'); hideTip(); };
  const move = e => { const b = svg.getBoundingClientRect(), px = e.clientX - b.left; let j = 0; L.pts.forEach((p, q) => { if (Math.abs(X(p.x) - px) < Math.abs(X(L.pts[j].x) - px)) j = q; }); show(j, e.clientX, e.clientY); };
  svg.addEventListener('pointermove', move); svg.addEventListener('pointerdown', move); svg.addEventListener('pointerleave', off); svg.addEventListener('blur', off);
  svg.addEventListener('keydown', e => { const d = {ArrowLeft: -1, ArrowRight: 1}[e.key]; if (e.key === 'Escape') return off(); if (!d) return; e.preventDefault();
    const j = Math.max(0, Math.min(L.pts.length - 1, cur < 0 ? Math.max(0, L.pts.findIndex(p => p.x === 0)) : cur + d)), b = svg.getBoundingClientRect(); show(j, b.left + X(L.pts[j].x), b.top + m.t + 20); });
}

/* ---- H.264: each pose's score from the raw render against the same render after the round trip */
function drawScatter(host) {
  const W = host.clientWidth; if (!W) return;
  const k = sel.h264, Q = D.h264[k], lg = document.getElementById('h264-legend'); host.textContent = '';
  if (lg) lg.hidden = !Q || !Q.pts.length;
  if (!Q || !Q.pts.length) { H('p', 'na', D.r[k] ? 'Not available for this detector: no limit could be set in this run.' : 'Not available: this run has no score per pose.', host); return; }
  const m = {l: 44, r: 12, t: 26, b: 44}, pw = Math.min(W, 440) - m.l - m.r, Hh = m.t + pw + m.b;
  let lo = 1, hi = 1; for (const p of Q.pts) { lo = Math.min(lo, p[0], p[1]); hi = Math.max(hi, p[0], p[1]); }
  if (Q.yl != null) { lo = Math.min(lo, Q.yl); hi = Math.max(hi, Q.yl); }
  lo /= 1.12; hi *= 1.12; if (hi / lo < 1.5) { const f = Math.sqrt(1.5 / (hi / lo)); lo /= f; hi *= f; }
  const T = v => Math.log(v / lo) / Math.log(hi / lo), X = v => m.l + pw * T(v), Y = v => m.t + pw * (1 - T(v));
  const svg = S('svg', {width: m.l + pw + m.r, height: Hh, viewBox: `0 0 ${m.l + pw + m.r} ${Hh}`, role: 'img',
                        'aria-label': 'Score of each pose from the raw render against the same render after the H.264 round trip, both divided by the limit.'}, host);
  let lastX = -99;
  for (const v of logTicks(lo, hi)) { const t = tickText(v);
    if (v !== 1) { S('line', {x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), class: 'gridl'}, svg); S('line', {x1: X(v), x2: X(v), y1: m.t, y2: m.t + pw, class: 'gridl'}, svg); }
    S('text', {x: m.l - 7, y: Y(v) + 4, 'text-anchor': 'end', class: v === 1 ? 'tick-strong' : ''}, svg).textContent = t;
    if (X(v) - lastX > 26) { S('text', {x: X(v), y: m.t + pw + 16, 'text-anchor': 'middle', class: v === 1 ? 'tick-strong' : ''}, svg).textContent = t; lastX = X(v); }
  }
  S('rect', {x: m.l, y: m.t, width: pw, height: pw, fill: 'none', class: 'axisl'}, svg);
  S('line', {x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), class: 'diag'}, svg);
  S('line', {x1: X(1), x2: X(1), y1: m.t, y2: m.t + pw, class: 'limit'}, svg);
  if (Q.yl != null) S('line', {x1: m.l, x2: m.l + pw, y1: Y(Q.yl), y2: Y(Q.yl), class: 'limit'}, svg);
  S('text', {x: 2, y: 12}, svg).textContent = 'after the round trip, score ÷ limit';
  S('text', {x: m.l + pw / 2, y: Hh - 8, 'text-anchor': 'middle'}, svg).textContent = 'raw render, score ÷ limit';
  for (const pass of ['rej', 'acc', 'wrong']) for (const p of Q.pts) if (p[3] === pass)
    S('circle', {cx: X(p[0]), cy: Y(p[1]), r: pass === 'wrong' ? 4.2 : 3, class: 'c-' + pass + ' pt sm'}, svg);
  S('text', {x: X(hi) - 6, y: Y(hi) + 14, 'text-anchor': 'end', class: 'halo'}, svg).textContent = 'no change';
  S('text', {x: X(1) + 5, y: m.t + pw - 7, class: 'tick-strong halo'}, svg).textContent = 'limit';
  if (Q.yl != null) S('text', {x: m.l + 5, y: Y(Q.yl) - 6, class: 'tick-strong halo'}, svg).textContent = 'limit after the round trip';
  const hl = S('circle', {r: 7, class: 'hl', visibility: 'hidden'}, svg);
  const off = () => { hl.setAttribute('visibility', 'hidden'); hideTip(); };
  const pick = e => { const b = svg.getBoundingClientRect(), px = e.clientX - b.left, py = e.clientY - b.top; let best = null, bd = 16;
    for (const p of Q.pts) { const dd = Math.hypot(X(p[0]) - px, Y(p[1]) - py); if (dd < bd) { bd = dd; best = p; } }
    if (!best) return off();
    hl.setAttribute('cx', X(best[0])); hl.setAttribute('cy', Y(best[1])); hl.setAttribute('visibility', 'visible');
    const i = best[2], t = best[3] === 'wrong' ? {c: 'wrong', b: 'decision flips', t: D.a[k][i] ? 'accepted raw, rejected after' : 'rejected raw, accepted after'}
                                             : best[3] === 'acc' ? {c: 'accept', b: 'accepted', t: 'both ways'} : {c: 'rej', b: 'rejected', t: 'both ways'};
    showTip(e.clientX, e.clientY, {img: D.thumbs[P[i].id], title: P[i].where, sub: P[i].dist, rows: [t, {b: xr(best[0]), t: 'the limit, raw render'}, {b: xr(best[1] / (Q.yl || 1)), t: Q.yl ? 'its own limit, after the round trip' : 'the limit, after the round trip'}]}); };
  svg.addEventListener('pointermove', pick); svg.addEventListener('pointerdown', pick); svg.addEventListener('pointerleave', off);
}

/* ---- why: the training starts in their own scenes and the poses of this scene on one axis, score ÷ limit */
function drawStrip(host) {
  const W = host.clientWidth; if (!W) return;
  const k = sel.why, Q = D.why[k]; host.textContent = '';
  if (!Q) { H('p', 'na', 'Not available for this detector.', host); return; }
  const scene = name => D.why_sets[name].filter(i => D.r[k][i] > 0).map(i => ({r: D.r[k][i], i, c: clsOf(i, k)}));
  const rows = [{label: 'Training starts, each in its own scene', pts: Q.cal.map(r => ({r, c: 'ctx'}))},
                {label: 'This scene: poses in the training range', pts: scene('inr')},
                {label: 'This scene: far poses', pts: scene('far')}];
  const narrow = W < 560, m = {l: narrow ? 6 : 232, r: 26, t: 26, b: 40}, rh = narrow ? 74 : 56, spreadY = narrow ? 34 : 36;
  const pw = W - m.l - m.r, yb = m.t + rh * rows.length, Hh = yb + m.b;
  let lo = 1, hi = 1; for (const row of rows) for (const p of row.pts) { lo = Math.min(lo, p.r); hi = Math.max(hi, p.r); }
  lo /= 1.12; hi *= 1.12; if (hi / lo < 1.5) { const f = Math.sqrt(1.5 / (hi / lo)); lo /= f; hi *= f; }
  const X = v => m.l + pw * Math.log(v / lo) / Math.log(hi / lo);
  const svg = S('svg', {width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: 'img',
                        'aria-label': 'Score divided by the limit: the training starts in their own scenes, the poses of this scene in the training range, and its far poses, on one axis.'}, host);
  const shown = [X(1)];
  for (const v of logTicks(lo, hi)) { if (v === 1) continue;
    S('line', {x1: X(v), x2: X(v), y1: m.t, y2: yb, class: 'gridl'}, svg);
    if (shown.every(x => Math.abs(x - X(v)) > 30)) { S('text', {x: X(v), y: yb + 16, 'text-anchor': 'middle'}, svg).textContent = tickText(v); shown.push(X(v)); } }
  S('text', {x: X(1), y: yb + 16, 'text-anchor': 'middle', class: 'tick-strong', 'font-weight': 650}, svg).textContent = '1';
  S('line', {x1: m.l, x2: m.l + pw, y1: yb, y2: yb, class: 'axisl'}, svg);
  S('text', {x: m.l + pw / 2, y: Hh - 6, 'text-anchor': 'middle'}, svg).textContent = 'score ÷ limit';
  const all = [];
  rows.forEach((row, g) => { const top = m.t + rh * g, cy = narrow ? top + 22 + (rh - 22) / 2 : top + rh / 2;
    if (g) S('line', {x1: narrow ? 0 : m.l, x2: m.l + pw, y1: top, y2: top, class: 'gridl'}, svg);
    if (!narrow) { const ls = wrap(row.label, 30), y0 = cy - 7 * ls.length + 3;
      ls.forEach((t, j) => { S('text', {x: m.l - 16, y: y0 + 14 * j, 'text-anchor': 'end', class: 'tick-strong'}, svg).textContent = t; });
      S('text', {x: m.l - 16, y: y0 + 14 * ls.length, 'text-anchor': 'end', class: 'tick-muted'}, svg).textContent = row.pts.length + ' camera views'; }
    row.pts.forEach((p, j) => { all.push([X(p.r), cy + ((j * 0.61803398875) % 1 - .5) * spreadY, p]); });
  });
  S('line', {x1: X(1), x2: X(1), y1: m.t - 8, y2: yb, class: 'limit'}, svg);
  S('text', {x: X(1), y: m.t - 12, 'text-anchor': 'middle', class: 'tick-strong', 'font-weight': 650}, svg).textContent = 'limit';
  if (Q.start > lo && Q.start < hi) S('line', {x1: X(Q.start), x2: X(Q.start), y1: m.t + rh, y2: yb, class: 'startl'}, svg);
  if (narrow) rows.forEach((row, g) => { const y = m.t + rh * g + 16;   // on a patch of the surface: the limit line may cross a label
    const t = S('text', {x: m.l, y, class: 'tick-strong'}, svg); t.textContent = row.label;
    const b = t.getBBox(); svg.insertBefore(S('rect', {x: f1(b.x - 4), y: f1(b.y), width: f1(b.width + 8), height: f1(b.height), style: 'fill:var(--surface)'}), t);
    S('text', {x: W - 4, y, 'text-anchor': 'end', class: 'tick-muted'}, svg).textContent = String(row.pts.length); });
  for (const pass of ['ctx', 'rej', 'acc', 'wrong']) for (const [x, y, p] of all) if (p.c === pass)
    S('circle', {cx: f1(x), cy: f1(y), r: pass === 'ctx' ? 3 : 3.2, class: pass === 'ctx' ? 'c-ctx' : 'c-' + pass + ' pt sm'}, svg);
  const hl = S('circle', {r: 7, class: 'hl', visibility: 'hidden'}, svg);
  const off = () => { hl.setAttribute('visibility', 'hidden'); hideTip(); };
  const pick = e => { const b = svg.getBoundingClientRect(), px = e.clientX - b.left, py = e.clientY - b.top; let best = null, bd = 14;
    for (const q of all) { const dd = Math.hypot(q[0] - px, q[1] - py); if (dd < bd) { bd = dd; best = q; } }
    if (!best) return off();
    hl.setAttribute('cx', best[0]); hl.setAttribute('cy', best[1]); hl.setAttribute('visibility', 'visible');
    const p = best[2];
    showTip(e.clientX, e.clientY, p.i == null ? {title: 'A training start, re-rendered in its own scene', rows: [{b: xr(p.r), t: 'the limit'}]} : poseTip(p.i, k)); };
  svg.addEventListener('pointermove', pick); svg.addEventListener('pointerdown', pick); svg.addEventListener('pointerleave', off);
}

/* ---- earlier trials: success rate in the lowest, middle and highest third of the score */
function drawTrials(host) {
  const W = host.clientWidth; if (!W) return;
  const R = D.trials[sel.trials] || []; host.textContent = '';
  if (!R.length) { H('p', 'na', 'Not available for this detector.', host); return; }
  const G = R.length, m = {l: 38, r: 4, t: 26, b: 0}, gw = (W - m.l - m.r) / G, cols = Math.max(8, Math.floor((gw - 8) / 6));
  const SAYS = {yes: 'low score, more success', opposite: 'low score, less success', no: 'no link'};
  const lines = R.map(r => wrap(r.label, cols)), says = R.map(r => wrap(SAYS[r.link] || '', cols));
  m.b = 20 + 13 * Math.max(...lines.map((l, g) => l.length + says[g].length)) + 22;
  const ph = Math.round(Math.max(170, Math.min(230, W * .28))), Hh = m.t + ph + m.b, base = m.t + ph, y = v => base - ph * v;
  const bw = Math.max(8, Math.min(24, (gw - 22) / 3 - 2)), step = bw + (bw >= 20 ? 8 : 2), gx = g => m.l + gw * g + (gw - 3 * step + (step - bw)) / 2;
  const svg = S('svg', {width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: 'img', tabindex: 0,
                        'aria-label': 'Success rate of earlier trials in the lowest, middle and highest third of the score, per set of trials. Arrow keys move between the sets.'}, host);
  for (const v of [0, .5, 1]) { S('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: v ? 'gridl' : 'axisl'}, svg);
    S('text', {x: m.l - 7, y: y(v) + 4, 'text-anchor': 'end'}, svg).textContent = pct(v); }
  S('text', {x: 2, y: 12}, svg).textContent = 'trials that succeed';
  R.forEach((r, g) => {
    r.thirds.forEach((t, j) => { if (!t || !t[1]) return; const v = t[0] / t[1], x = gx(g) + j * step, top = bar(svg, x, base, bw, ph * v, 'b-t' + (j + 1));
      const wide = step >= 30; if (!wide && j === 1) return;
      S('text', {x: wide ? x + bw / 2 : j ? x : x + bw, y: top - 5, 'text-anchor': wide ? 'middle' : j ? 'start' : 'end', class: 'tick-strong halo'}, svg).textContent = pct(v); });
    const cx = m.l + gw * (g + .5);
    lines[g].forEach((t, j) => { S('text', {x: cx, y: base + 17 + 13 * j, 'text-anchor': 'middle'}, svg).textContent = t; });
    S('text', {x: cx, y: base + 19 + 13 * lines[g].length, 'text-anchor': 'middle', class: 'tick-muted'}, svg).textContent = r.n + ' trials';
    says[g].forEach((t, j) => { S('text', {x: cx, y: base + 37 + 13 * (lines[g].length + j), 'text-anchor': 'middle', class: 'tick-strong', 'font-weight': r.link === 'no' ? 400 : 650}, svg).textContent = t; });
  });
  const hl = S('rect', {class: 'hl', visibility: 'hidden', rx: 4}, svg);
  let cur = -1;
  const show = (g, px, py) => { cur = g; const r = R[g];
    for (const [a, val] of Object.entries({x: m.l + gw * g + 1, y: m.t - 2, width: gw - 2, height: ph + 4, visibility: 'visible'})) hl.setAttribute(a, val);
    const rows = r.thirds.map((t, j) => ({c: 't' + (j + 1), b: t && t[1] ? `${t[0]} of ${t[1]}` : 'n/a', t: 'succeed, ' + ['lowest', 'middle', 'highest'][j] + ' third'}));
    if (r.auroc != null) rows.push({b: pct(r.auroc), t: 'of pairs: the success scores lower'});
    showTip(px, py, {title: r.label, sub: r.n + ' trials, ' + r.s + ' succeed', rows}); };
  const off = () => { cur = -1; hl.setAttribute('visibility', 'hidden'); hideTip(); };
  const move = e => { const b = svg.getBoundingClientRect(); show(Math.max(0, Math.min(G - 1, Math.floor((e.clientX - b.left - m.l) / gw))), e.clientX, e.clientY); };
  svg.addEventListener('pointermove', move); svg.addEventListener('pointerdown', move); svg.addEventListener('pointerleave', off); svg.addEventListener('blur', off);
  svg.addEventListener('keydown', e => { const d = {ArrowLeft: -1, ArrowRight: 1}[e.key]; if (e.key === 'Escape') return off(); if (!d) return; e.preventDefault();
    const g = Math.max(0, Math.min(G - 1, cur < 0 ? 0 : cur + d)), b = svg.getBoundingClientRect(); show(g, b.left + m.l + gw * (g + .5), b.top + m.t + 20); });
}

/* ---- legends */
function legend(id, items) {
  const host = document.getElementById(id); if (!host) return; host.textContent = '';
  for (const [shape, c, label, mod] of items) { const sp = H('span', null, null, host); key(sp, shape, c, mod); say(sp, label); }
}
const present = f => D.dets.filter(f);
if (D.bands) legend('bands-legend', present(d => D.bands.by[d.k]).map(d => ['sq', d.c, d.name]));
if (D.map) legend('map-legend', [['sq', 'accept', 'accepted'], ['sq', 'wrong', 'accepted, but ' + D.meta.far_text], ['dot', null, 'rejected'], ['box', null, 'training range'], ['fau', null, 'faucet']]);
for (const name in D.lines || {}) { const L = D.lines[name];
  legend('line-legend-' + name, present(d => L.pts.some(p => p.r[d.k] != null)).map(d => ['ln', d.c, d.name, lineMod(d)]).concat(L.zone && L.zone_label ? [['zone', null, L.zone_label]] : [])); }
legend('why-legend', [['vdash', null, 'this scene: its own training start'], ['ring', null, 'a training start in its own scene'], ['pt', 'accept', 'this scene: accepted'], ['pt', 'wrong', 'this scene: accepted, but ' + D.meta.far_text], ['pt', 'rej', 'this scene: rejected']]);
legend('h264-legend', [['pt', 'accept', 'accepted both ways'], ['pt', 'rej', 'rejected both ways'], ['pt', 'wrong', 'the decision flips']]);
legend('trials-legend', [['sq', 't1', 'lowest third of the score'], ['sq', 't2', 'middle third'], ['sq', 't3', 'highest third']]);

/* ---- draw, and again when a chart's width changes or a detector is picked */
const charts = [];
const add = (q, fn) => { for (const h of document.querySelectorAll(q)) charts.push([h, fn]); };
if (D.bands) add('#bands', drawBands);
if (D.map) add('[data-map]', drawMap);
if (D.why) add('#why', drawStrip);
add('[data-line]', drawLine);
if (D.h264) add('#h264', drawScatter);
if (D.trials) add('#trials', drawTrials);
const widths = new Map();
const draw = () => { for (const [h, fn] of charts) { const w = h.clientWidth; if (w && widths.get(h) !== w) { widths.set(h, w); fn(h); } } };
const redraw = q => { for (const [h, fn] of charts) if (h.matches(q)) fn(h); };
const T = D.text || {};
if (D.map) seg('map', Object.keys(D.a), k => { setText('h-where', T.where[k]); (T.map_titles[k] || []).forEach((t, j) => setText('map-t' + j, t)); redraw('[data-map]'); });
if (D.why) seg('why', D.dets.map(d => d.k).filter(k => D.why[k]), k => { setText('h-why', T.why[k]); setText('why-sub', (T.why_sub || {})[k] || ''); redraw('#why'); });
if (D.h264) seg('h264', D.dets.map(d => d.k).filter(k => D.h264[k]), k => { setText('h-h264', T.h264[k]); setText('h264-scores', T.h264_scores[k] || ''); setText('h264-starts', T.h264_starts[k] || ''); setText('h264-s', T.h264s[k] || ''); redraw('#h264'); });
if (D.trials) seg('trials', D.dets.map(d => d.k).filter(k => D.trials[k]), k => { setText('h-trials', T.trials[k]); redraw('#trials'); });
draw();
let timer = 0;
if (window.ResizeObserver) new ResizeObserver(() => { clearTimeout(timer); timer = setTimeout(draw, 80); }).observe(document.querySelector('.page'));
else window.addEventListener('resize', draw);
document.addEventListener('pointerdown', e => { if (!e.target.closest('svg')) hideTip(); });
document.addEventListener('pointerleave', e => { if (e.pointerType === 'touch') e.stopPropagation(); }, true);  // a tap keeps its tooltip
})();
</script>
"""


def section(eyebrow: str, h_id: str, headline: str, body: str) -> str:
    if not headline or not body:
        return ""
    return (f'<section>\n  <p class="eyebrow">{esc(eyebrow)}</p>\n  <h2 id="{h_id}">{esc(headline)}</h2>\n{body}\n</section>\n')


def na_fig(title: str, why: str) -> str:
    """A chart's frame with a short note in place of the chart."""
    return f'  <div class="figs c1"><div class="fig"><h3>{esc(title)}</h3><p class="na">{esc(why)}</p></div></div>'


def subs(*lines: str) -> str:
    return "\n".join(f'  <p class="sub">{esc(s)}</p>' for s in lines if s)


def line_section(eyebrow: str, name: str, headline: str, title: str, *sub: str) -> str:
    return section(eyebrow, f"h-{name}", headline,
                   f'  <div class="figs c1"><div class="fig"><h3>{esc(title)}</h3><div data-line="{name}"></div>'
                   f'<div class="legend" id="line-legend-{name}"></div></div></div>\n' + subs(*sub))


def card(d: str, e: dict, definition: str, turned_deg: float, limit: bool = True) -> str:
    """A detector's verdict and its three counts; without a limit there is neither."""
    g = (e.get("grid") or {}) if limit else {}
    fc, rot = dig(g, "verdict", "far_vs_close"), dig(g, "verdict", "rotation")
    chip = f'<span class="vchip v-{fc if fc in VERDICT else "na"}">{esc(VERDICT.get(fc, "No verdict" if limit else "No limit"))}</span>'
    chip += (f'<span class="vchip sm v-{rot}">turning: {"passes" if rot == "pass" else "fails"}</span>' if rot in ("pass", "fail") else "")

    def stat(p, text: str) -> str:
        q = pair(p)
        value = f"{q[0]}<small>/{q[1]}</small>" if q else NA
        return f"<div><dt>{value}</dt><dd>{esc(text)}</dd></div>"

    key = f'<i class="k sq" style="--c:var(--{COLOR[d]})"></i>'
    return (f'<div class="card"><div class="cn">{key}<span>{esc(NAME[d])}</span></div><div class="cd">{esc(definition)}</div>'
            f'<div class="chips">{chip}</div><dl class="cs">{stat(g.get("in_range"), "in-range poses accepted")}'
            f'{stat(g.get("far"), "far poses accepted")}{stat(g.get("turned_rejected"), f"poses turned {turned_deg:g}° or more rejected")}</dl></div>')


def build(args: argparse.Namespace) -> tuple[str, str, dict]:
    """The page body (title to script) with and without the links to local files, and a few facts for the log."""
    out, exp3, sfx = Path(args.out), Path(args.exp3), args.scores_suffix
    scores, feats = out / f"scores{sfx}", out / f"feats{sfx}"
    S = json.loads((scores / "summary.json").read_text())
    Z = np.load(scores / "scores.npz") if (scores / "scores.npz").exists() else None
    DET = {d: e for d, e in (S.get("detectors") or {}).items() if isinstance(e, dict)}
    counts, rule = S.get("counts") or {}, S.get("rule") or {}
    dets = [d for d in HEADLINE if dig(DET, d, "grid")]
    band = float(rule["band_m"]) if fin(rule.get("band_m")) else 0.1
    turned_deg = float(rule["turned_deg"]) if fin(rule.get("turned_deg")) else 20.0
    grid_of = lambda d: dig(DET, d, "grid", default={})
    tau_of = lambda d: dig(DET, d, "grid", "tau")
    has_limit = lambda d: fin(tau_of(d)) and tau_of(d) > 0
    # the detector the section headlines speak about and the charts emphasise: the paper's; when it has no limit,
    # the first headline detector that has one. A detector without a limit is in no chart that needs one.
    limited = [d for d in dets if has_limit(d)]
    hero = "paper" if "paper" in limited else limited[0] if limited else "paper" if "paper" in dets else (dets[0] if dets else None)
    no_limit_note = "Not available: no limit could be set in this run."

    prep = exp3 / "prep"
    plan = json.loads((prep / "plan.json").read_text()) if (prep / "plan.json").exists() else None
    index = json.loads((prep / "train_index.json").read_text()) if (prep / "train_index.json").exists() else {}
    room = np.load(prep / "room_map.npz") if (prep / "room_map.npz").exists() else None
    n_train = dig(counts, "train_episodes") or len(index.get("episodes") or []) or None

    # ---- the poses: experiment 3's grid, then the extra poses of this experiment
    grid_rows, extra_rows = load_rows(feats / "grid" / "obs.jsonl"), load_rows(feats / "extra" / "obs.jsonl")
    cell_by_id = {int(q["id"]): float(q.get("cell") or 0.0) for q in (plan or {}).get("poses", []) if fin(q.get("id"))}
    default_cell = dict(fine=0.05, medium=0.1, coarse=0.3, facing=0.5)

    def entry(r: dict, src: Path, is_grid: bool) -> dict:
        m = re.search(r"(\d+)$", str(r.get("id", "")))
        kind = r.get("kind", "")
        d = float(r["pose_dist"]) if fin(r.get("pose_dist")) else 0.0
        val = lambda k: float(r[k]) if fin(r.get(k)) else 0.0
        return dict(id=str(r.get("id")), x=round(val("fy"), 3), y=round(val("fx"), 3), yaw=round(val("dyaw"), 1), k=kind,
                    far=bool(kind != "anchor" and d > band), inr=bool(r.get("in_range")),
                    c=cell_by_id.get(int(m.group(1)), default_cell.get(kind, 0.0)) if m and is_grid else 0.0,
                    d=round(d, 4), where=where_text(r), dist=outside_text(r), file=str(src / r["file"]) if r.get("file") else None)

    poses = [entry(r, exp3 / "exp3_indist_signal", True) for r in grid_rows] + [entry(r, out / "renders", False) for r in extra_rows]
    n_grid = len(grid_rows)
    kind = np.array([p["k"] for p in poses], dtype=object)
    is_grid = np.arange(len(poses)) < n_grid
    ev = is_grid & (kind != "anchor")
    far = np.array([p["far"] for p in poses], dtype=bool)
    inr = np.array([p["inr"] for p in poses], dtype=bool)
    pdist = np.array([p["d"] for p in poses], dtype=float)
    px, py, pyaw = (np.array([p[k] for p in poses], dtype=float) for k in ("x", "y", "yaw"))

    def scores_of(d: str, name: str, n: int):
        key = f"{d}/{name}"
        if Z is None or key not in Z.files:
            return None
        a = np.asarray(Z[key], dtype=float)
        return a if a.shape == (n,) else None

    score, acc, ratio = {}, {}, {}
    for d in dets:
        sg = scores_of(d, "grid", n_grid) if n_grid else None
        if sg is None:
            continue
        se = scores_of(d, "extra", len(extra_rows))
        s = np.r_[sg, se if se is not None else np.full(len(extra_rows), np.nan)]
        t = tau_of(d)
        score[d] = s
        with np.errstate(invalid="ignore"):
            acc[d] = (s <= t) if isinstance(t, (int, float)) and not math.isnan(t) else np.zeros(len(s), bool)
        ratio[d] = s / t if has_limit(d) else np.full(len(s), np.nan)
    scored = [d for d in dets if d in score]

    # ---- header
    verdict = lambda d: dig(grid_of(d), "verdict", "far_vs_close") if d in dets and has_limit(d) else None
    h1 = h1_sentence(verdict("paper"), [verdict(d) for d in ("start_tied", "start_lw") if d in dets], verdict("token_match"),
                     no_limit="paper" in dets and not has_limit("paper"))
    frames = dig(counts, "train_frames")
    n_starts = f"the {n_train} training starts" if n_train else "the training starts"
    n_tm = dig(counts, "sets", "trainstart") or n_train
    definitions = dict(
        paper=f"distance to all {frames:,} training frames" if fin(frames) else "distance to all training frames",
        start_tied=f"distance to {n_starts}, spread of all training frames",
        start_lw=f"distance to {n_starts}, spread of the starts alone",
        token_match=f"image patches matched to the {n_tm} training starts" if n_tm else "image patches matched to the training starts")
    cards = "".join(card(d, DET[d], definitions[d], turned_deg, has_limit(d)) for d in dets)
    cards_style = f' style="--n:{len(dets)};max-width:{len(dets) * 290}px"' if 0 < len(dets) < 4 else ""
    notice = notice_sentence([SHORT[d] for d in dets if not has_limit(d)], dig(counts, "sets", "start"), rule.get("alpha", 0.05))
    step = S.get("ckpt_step")
    step = f"{int(step):,}" if str(step).isdigit() else step
    eyebrow = " · ".join(["Experiment 4", "Open the faucet"] + ([f"checkpoint step {step}"] if step not in (None, "") else []))
    rule_line = rule_sentence(rule, counts)

    data: dict = dict(dets=[dict(k=d, name=SHORT[d], c=COLOR[d], ref=d == "token_match") for d in dets], hero=hero,
                      meta=dict(far_text=f"more than {dist(band)} outside"))
    text: dict = {}
    sections, heads = [], {}

    # ---- 1 far vs close
    names = list((dig(counts, "grid", "bands") or {}).keys())
    banded = [d for d in dets if isinstance(grid_of(d).get("by_band"), list) and len(grid_of(d)["by_band"]) == len(names)]
    by = {d: grid_of(d)["by_band"] for d in banded if has_limit(d)}
    bands_title = "Share of poses accepted, by distance outside the training range"
    if names and hero in by:
        far_from = next((j for j, n in enumerate(names) if j and not n.startswith("up to")), len(names))
        data["bands"] = dict(names=names, n=[dig(counts, "grid", "bands", n) for n in names], by=by, far_from=far_from)
        heads["Far vs close"] = bands_headline(SHORT[hero], grid_of(hero).get("in_range"), grid_of(hero).get("far"))
        sections.append(section("Far vs close", "h-bands", heads["Far vs close"],
                                f'  <div class="figs c1"><div class="fig"><h3>{bands_title}</h3>'
                                '<div id="bands"></div><div class="legend" id="bands-legend"></div></div></div>\n'
                                + subs(f"Far = more than {dist(band)} outside the training range. Hover or tap a group for the counts.")))
    elif names and hero in banded and not limited:
        heads["Far vs close"] = no_limit_headline(SHORT[hero], "the share of poses it accepts cannot be shown")
        sections.append(section("Far vs close", "h-bands", heads["Far vs close"], na_fig(bands_title, no_limit_note)))

    # ---- why: the training starts' scores in their own scenes, against this scene's poses, on one axis
    data["why"], text["why"], text["why_sub"] = {}, {}, {}
    inr_idx, far_idx = [int(i) for i in np.nonzero(ev & inr)[0]], [int(i) for i in np.nonzero(ev & far)[0]]
    start_idx = [i for i in range(n_grid) if poses[i]["k"] == "anchor"]
    median_of = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")
    for d in scored:
        cal = scores_of(d, "start", int(dig(counts, "sets", "start") or 0))
        if cal is None or not has_limit(d) or not (start_idx and far_idx):
            continue
        cal = cal[np.isfinite(cal) & (cal > 0)]
        base = median_of(score[d][start_idx])
        if len(cal) < 20 or not (fin(base) and base > 0):
            continue
        q05, q95 = (float(v) for v in np.percentile(cal, [5, 95]))
        rise, spread, headroom = median_of(score[d][far_idx]) / base, q95 / q05, tau_of(d) / base
        headline = why_headline(SHORT[d], rise, headroom)
        if not headline:  # no far pose has a score: nothing to say, so no entry in the detector switch either
            continue
        data["why"][d] = dict(cal=[sig(v / tau_of(d)) for v in np.sort(cal)], rise=sig(rise), spread=sig(spread),
                              headroom=sig(headroom), start=sig(base / tau_of(d)))
        text["why"][d], text["why_sub"][d] = headline, why_sub(spread)
    if data["why"]:
        data["why_sets"] = dict(inr=inr_idx, far=far_idx)
        first = hero if hero in data["why"] else next(iter(data["why"]))
        heads["Why"] = text["why"][first]
        sections.append(section("Why", "h-why", heads["Why"],
                                '  <div class="seg" data-seg="why" role="group" aria-label="Detector shown in the chart"></div>\n'
                                '  <div class="figs c1"><div class="fig"><h3>Each camera view: its score divided by the limit</h3>'
                                '<div id="why"></div><div class="legend" id="why-legend"></div></div></div>\n'
                                + f'  <p class="sub" id="why-sub">{esc(text["why_sub"][first])}</p>\n'
                                + subs("Hover or tap a pose for its camera view.")))
    else:
        del data["why"]

    # ---- 2 where
    lattice = [i for i in range(n_grid) if poses[i]["k"] in ("fine", "medium", "coarse")]
    facing = [i for i in range(n_grid) if poses[i]["k"] == "facing"]
    mapped = [d for d in scored if has_limit(d)]
    if plan is not None and scored and lattice and not mapped:
        heads["Where"] = no_limit_headline(SHORT[hero], "the poses it accepts cannot be mapped")
        sections.append(section("Where", "h-where", heads["Where"], na_fig("The poses it accepts, top-down", no_limit_note)))
    if plan is not None and mapped and lattice:
        tbl = table_polygon(plan)
        fx_lo, fx_hi, fy_lo, fy_hi = train_box(plan)
        pts = np.r_[np.stack([px[:n_grid], py[:n_grid]], axis=1), tbl]
        views = dict(zoom=dict(l=0.95, r=-0.95, b=-0.75, t=0.85),
                     room=dict(l=round(float(pts[:, 0].max()) + 0.35, 3), r=round(float(pts[:, 0].min()) - 0.35, 3),
                               b=round(float(pts[:, 1].min()) - 0.35, 3), t=round(float(pts[:, 1].max()) + 0.35, 3)))
        data["map"] = dict(views=views, walls=wall_rects(room, plan, views["room"]) if room is not None else [],
                           table=np.round(tbl, 3).tolist(), faucet=np.round(heading_xy(plan, plan["anchor"]["faucet"][:2]), 3).tolist(),
                           box=dict(l=round(fy_hi, 3), r=round(fy_lo, 3), b=round(fx_lo, 3), t=round(fx_hi, 3)))
        data["sets"] = dict(lattice=lattice, facing=facing)
        text["where"], text["map_titles"] = {}, {}
        lat_m, face_m = np.isin(np.arange(len(poses)), lattice), np.isin(np.arange(len(poses)), facing)
        for d in mapped:
            a = acc[d]
            fa = ev & far & a
            text["where"][d] = where_headline(SHORT[d], int((ev & a).sum()), [int((ev & inr & a).sum()), int((ev & inr).sum())],
                                              [int(fa.sum()), int((ev & far).sum())], float(pdist[fa].max()) if fa.any() else None,
                                              float(pdist[ev & a].max()) if (ev & a).any() else None)
            # each map's title counts the poses that map shows (the position grid; the poses that face the faucet)
            text["map_titles"][d] = [
                f"Around the start: {int((lat_m & inr & a).sum())} of {int((lat_m & inr).sum())} in-range poses accepted",
                f"Whole room: {int((lat_m & far & a).sum())} of {int((lat_m & far).sum())} far poses accepted",
                f"Turned to face the faucet: {int((face_m & a).sum())} of {len(facing)} accepted"]
        first = hero if hero in mapped else mapped[0]
        tt = text["map_titles"][first]
        figs = (f'<div class="fig"><h3 id="map-t0">{esc(tt[0])}</h3><div data-map data-view="zoom" data-sel="lattice"></div></div>'
                f'<div class="fig"><h3 id="map-t1">{esc(tt[1])}</h3><div data-map data-view="room" data-sel="lattice"></div></div>'
                + (f'<div class="fig"><h3 id="map-t2">{esc(tt[2])}</h3><div data-map data-view="room" data-sel="facing"></div></div>' if facing else ""))
        heads["Where"] = text["where"][first]
        sections.append(section("Where", "h-where", heads["Where"],
                                '  <div class="seg" data-seg="map" role="group" aria-label="Detector shown on the maps"></div>\n'
                                f'  <div class="figs c3 lead">{figs}</div>\n  <div class="legend" id="map-legend"></div>\n'
                                + subs("Top-down, the robot faces up. Hover or tap a pose for its camera view and its score against the limit.")))

    # ---- 3-5 the three lines through the training start
    data["lines"] = {}

    def line(idx: list, xs: np.ndarray, at_zero: list | None = None) -> list:
        """One point per offset: the detectors' score / limit (the median where a pose was rendered more than once)."""
        groups: dict = {}
        for i in idx:
            groups.setdefault(round(float(xs[i]), 3), []).append(i)
        if at_zero:
            groups[0.0] = groups.get(0.0, []) + list(at_zero)
        pts = []
        for x in sorted(groups):
            mem = groups[x]
            rep = next((i for i in mem if poses[i]["k"] != "anchor"), mem[0])
            med = lambda a: float(np.nanmedian(a[mem])) if np.isfinite(a[mem]).any() else float("nan")
            pts.append(dict(x=x, i=rep, far=bool(far[rep]), r={d: sig(med(ratio[d])) for d in scored}))
        return pts

    def story_of(pts: list, lo: float, hi: float, neg: str, pos: str) -> str:
        """The headline of a line from the hero's values."""
        if hero not in scored:
            return ""
        if not has_limit(hero):
            return no_limit_headline(SHORT[hero])
        return line_headline(SHORT[hero], line_story([(p["x"], p["r"][hero], p["far"]) for p in pts if p["r"][hero] is not None], lo, hi), neg, pos)

    def put_line(name: str, pts: list, xlabel: str, wide: float, **kw) -> None:
        """A line chart's data; an axis that reaches beyond `wide` is stretched near 0 so the close poses stay apart."""
        stretch = bool(pts) and max(abs(p["x"]) for p in pts) > wide
        kw["xshort"] = kw["xshort"] + (" · stretched near 0" if stretch else "")
        data["lines"][name] = dict(pts=[dict(x=p["x"], i=p["i"], r=p["r"]) for p in pts], stretch=stretch,
                                   xlabel=xlabel + (" · axis stretched near 0" if stretch else ""), **kw)

    on_line = np.isin(kind, ("fine", "medium", "coarse", "anchor")) & is_grid
    straight = np.abs(pyaw) < 1e-6
    anchors = [i for i in range(n_grid) if poses[i]["k"] == "anchor"]
    box = train_box(plan) if plan is not None else None
    tbl = table_polygon(plan) if plan is not None else None
    how = "Each line is a score divided by its own limit. Hover or tap a point for the camera view."
    wrong = "under the limit although far: wrongly accepted"
    if scored:
        idx = [i for i in range(len(poses)) if straight[i] and abs(px[i]) < 1e-6 and (on_line[i] or (not is_grid[i] and poses[i]["k"] in ("approach", "closer")))]
        pts = line(idx, py)
        if len(pts) >= 3:
            lo, hi = (box[0], box[1]) if box else (0.0, 0.0)
            blocked = crossing(tbl, 1) if tbl is not None else None
            if blocked and not (pts[0]["x"] < blocked[0] and pts[-1]["x"] > blocked[1]):
                blocked = None
            near = [p for p in pts if blocked is None or p["x"] < blocked[0]]
            past = [(p["x"], p["r"][hero]) for p in pts if blocked is not None and p["x"] > blocked[1]] if hero in scored and has_limit(hero) else []
            put_line("approach", pts, band=[lo, hi] if box else None, band_label="training range", blocked=blocked, unit="m",
                     zone=[lo - band, hi + band] if box else None, zone_label=wrong,
                     xlabel=f"forward offset from the training start (m): {MINUS} farther back, + closer", wide=0.6,
                     xshort=f"forward offset (m): {MINUS} back, + closer",
                     aria="Score divided by the limit against the forward offset from the training start, one line per detector.")
            heads["Approach"] = story_of(near, lo, hi, "farther back", "closer")
            sections.append(line_section("Approach", "approach", heads["Approach"], "Standing farther back or closer, facing as in training",
                                         beyond_line(SHORT[hero], past) if past else "", how))
        idx = [i for i in range(n_grid) if straight[i] and abs(py[i]) < 1e-6 and on_line[i]]
        pts = line(idx, px)
        if len(pts) >= 3:
            lo, hi = (box[2], box[3]) if box else (0.0, 0.0)
            put_line("sideways", pts, band=[lo, hi] if box else None, band_label="training range", blocked=None, unit="m",
                     zone=[lo - band, hi + band] if box else None, zone_label=wrong,
                     xlabel=f"sideways offset from the training start (m): {MINUS} right, + left", wide=0.6,
                     xshort=f"sideways offset (m): {MINUS} right, + left",
                     aria="Score divided by the limit against the sideways offset from the training start, one line per detector.")
            heads["Sideways"] = story_of(pts, lo, hi, "to the right", "to the left")
            sections.append(line_section("Sideways", "sideways", heads["Sideways"], "Standing to the side, facing as in training", how))
        idx = [i for i in range(n_grid) if poses[i]["k"] == "yaw"]
        pts = line(idx, pyaw, at_zero=anchors) if idx else []
        if len(pts) >= 3 and hero in scored:
            hd = train_headings(plan) if plan is not None else None
            put_line("turned", pts, band=list(hd) if hd else None, band_label="training headings", blocked=None, unit="deg",
                     zone=[-turned_deg, turned_deg], zone_label=f"under the limit although turned {turned_deg:g}° or more: wrongly accepted",
                     xlabel=f"heading at the start position: {MINUS} turned right, + turned left", wide=45.0,
                     xshort=f"heading: {MINUS} right, + left",
                     aria="Score divided by the limit against the heading at the start position, one line per detector.")
            g = grid_of(hero)
            heads["Turned"] = (turned_headline(SHORT[hero], g.get("headings_accepted") or [], g.get("turned_rejected"), turned_deg)
                               if has_limit(hero) else no_limit_headline(SHORT[hero]))
            sections.append(line_section("Turned", "turned", heads["Turned"], "Turned on the spot at the start position", how))

    # ---- 6 video compression
    data["h264"] = {}
    for k in ("h264", "h264_scores", "h264_starts", "h264s"):
        text[k] = {}
    h_dets = [d for d in dets if isinstance(dig(DET, d, "h264"), dict)]
    for d in h_dets:
        h = DET[d]["h264"]
        text["h264"][d] = (h264_headline(SHORT[d], h, grid_of(d)) if has_limit(d) and fin(h.get("tau"))
                           else h264_no_limit_headline(SHORT[d], h))
        text["h264_scores"][d] = h264_scores_line(h)
        text["h264_starts"][d] = h264_starts_line(h.get("starts") or {})
        text["h264s"][d] = h264s_line(dig(DET, d, "h264s", default={})) if has_limit(d) else ""
        cod, th = scores_of(d, "grid_h264", n_grid), h.get("tau")
        pts = []
        if d in score and cod is not None and has_limit(d):
            t = tau_of(d)
            for i in np.nonzero(ev[:n_grid])[0]:
                a, b = score[d][i], cod[i]
                if not (fin(a) and fin(b) and a > 0 and b > 0):
                    continue
                a_raw, a_cod = bool(a <= t), bool(b <= th) if isinstance(th, (int, float)) and not math.isnan(th) else False
                pts.append([sig(a / t), sig(b / t), int(i), "wrong" if a_raw != a_cod else "acc" if a_raw else "rej"])
        data["h264"][d] = dict(pts=pts, yl=sig(th / tau_of(d)) if has_limit(d) and fin(th) and th > 0 else None)
    if h_dets:
        first = hero if hero in h_dets else h_dets[0]
        heads["Video compression"] = text["h264"][first]
        sections.append(section("Video compression", "h-h264", heads["Video compression"],
                                '  <div class="seg" data-seg="h264" role="group" aria-label="Detector shown in the chart"></div>\n'
                                '  <div class="figs c1"><div class="fig"><h3>Each pose: its score from the raw render, and after encoding and decoding '
                                'that render like a training video</h3>'
                                '<div class="split"><div id="h264"></div><div class="side"><div class="legend" id="h264-legend"></div>'
                                f'<p id="h264-scores">{esc(text["h264_scores"][first])}</p><p id="h264-starts">{esc(text["h264_starts"][first])}</p>'
                                f'<p id="h264-s">{esc(text["h264s"][first])}</p></div></div></div></div>'))
    else:
        del data["h264"]

    # ---- 7 success of the earlier trials
    data["trials"], text["trials"] = {}, {}
    for d in dets:
        tr = dig(DET, d, "trials")
        if not isinstance(tr, dict):
            continue
        groups = []
        for name, part, label in TRIAL_GROUPS:
            parts = tr.get(name) or {}
            t = parts.get(part) or (parts.get("all") if part == "standing" and "takeover" not in parts else None)
            th3 = [list(q) for q in map(pair, dig(t, "thirds", default=[])) if q] if t else []
            if not t or len(th3) != 3:
                continue
            groups.append(dict(label=label, n=int(t.get("n", sum(q[1] for q in th3))), s=int(t.get("successes", sum(q[0] for q in th3))),
                               thirds=th3, auroc=sig(t.get("auroc")), link=trial_link(t.get("auroc"), th3)))
        if groups:
            data["trials"][d] = groups
            text["trials"][d] = trials_headline(SHORT[d], groups)
    if data["trials"]:
        first = "paper" if "paper" in data["trials"] else hero if hero in data["trials"] else next(iter(data["trials"]))
        heads["Success"] = text["trials"][first]
        sections.append(section("Success", "h-trials", heads["Success"],
                                '  <div class="seg" data-seg="trials" role="group" aria-label="Detector shown in the chart"></div>\n'
                                '  <div class="figs c1"><div class="fig"><h3>Earlier closed-loop trials: success by the score of the first camera view</h3>'
                                '<div id="trials"></div><div class="legend" id="trials-legend"></div></div></div>\n'
                                + subs("The trial images are re-encoded copies of what the policy saw, so scores are only compared inside one set. "
                                       "No new rollouts were run.")))
    else:
        del data["trials"]

    # ---- per-pose decisions and camera views
    used = set(data.get("sets", {}).get("lattice", [])) | set(data.get("sets", {}).get("facing", []))
    used |= {p["i"] for L in data["lines"].values() for p in L["pts"]} | {p[2] for q in data.get("h264", {}).values() for p in q["pts"]}
    used |= set(data.get("why_sets", {}).get("inr", [])) | set(data.get("why_sets", {}).get("far", []))
    thumbs = {}
    for i in sorted(used):
        t = thumbnail(Path(poses[i]["file"])) if poses[i]["file"] else None
        if t:
            thumbs[poses[i]["id"]] = t
    data["poses"] = [{k: p[k] for k in ("id", "x", "y", "yaw", "c", "far", "where", "dist")} for p in poses]
    data["a"] = {d: acc[d].astype(int).tolist() for d in scored if has_limit(d)}
    data["first"] = dict(trials="paper" if "paper" in data.get("trials", {}) else hero)
    data["r"] = {d: [sig(v) for v in ratio[d]] for d in scored}
    data["text"] = text
    data["thumbs"] = thumbs

    links = [h for h in (f"scores{sfx}/summary.json", f"scores{sfx}/table.csv") if (out / h).exists()]
    head = ("<title>LLMD In-Distribution Test</title>\n"
            '<meta name="claude:viewport-fit-sides" content="cover">\n' + CSS + '\n<div class="page">\n<header>\n'
            f'  <p class="eyebrow">{esc(eyebrow)}</p>\n  <h1>{esc(h1)}</h1>\n  <p class="lede">{esc(lede_sentence(counts))}</p>\n'
            + (f'  <div class="cards"{cards_style}>{cards}</div>\n' if cards else "")
            + (f'  <p class="notice">{esc(notice)}</p>\n' if notice else "")
            + (f'  <p class="rule">{esc(rule_line)}</p>\n' if rule_line else "") + "</header>\n\n"
            + "\n".join(s for s in sections if s))
    tail = ('</div>\n\n<script type="application/json" id="D">'
            + json.dumps(clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/") + "</script>\n" + JS)

    def page(with_links: bool) -> str:
        also = also_checked(S, links if with_links else [])
        return head + (f'\n<section>\n  <p class="eyebrow">Details</p>\n  <h2>Also checked</h2>\n  <div class="also">{also}</div>\n</section>\n'
                       if also else "") + tail

    facts = dict(h1=h1, detectors=dets, headlines=heads, thumbs=len(thumbs))
    return page(True), page(False), facts


def check_text(page: str) -> list:
    """Numbers that are not finite must never reach the page: the words to look for, outside the embedded images."""
    bare = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "", page)
    return sorted(set(re.findall(r"\b(?:NaN|Infinity|nan|inf)\b", bare)))


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    full, bare, facts = build(args)
    (out / "index.html").write_text(HEAD + full, encoding="utf-8")
    print(f"[report] {facts['h1']}")
    for name, headline in facts["headlines"].items():
        print(f"[report] {name}: {headline}")
    print(f"[report] detectors {facts['detectors']}; {facts['thumbs']} camera views")
    bad = check_text(full)
    if bad:
        print(f"[report] WARNING: the page text contains {bad}", file=sys.stderr)
    print(f"wrote {out / 'index.html'} ({(out / 'index.html').stat().st_size / 1e6:.2f} MB)")
    if args.fragment:
        Path(args.fragment).parent.mkdir(parents=True, exist_ok=True)
        Path(args.fragment).write_text(bare, encoding="utf-8")
        print(f"wrote {args.fragment} ({Path(args.fragment).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
