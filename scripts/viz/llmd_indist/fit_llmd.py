"""Stage 3 of the LLMD in-distribution experiment: statistics, scores and verdicts (Psi0 venv, CPU, float64).

Reads the features of extract_llmd_features.py and scores every observation with the detectors below. All
statistics are per action token (30 tokens x 1536 channels) in float64; an observation's score is the largest of its
30 squared Mahalanobis distances (the paper's rule). An observation that belongs to a training episode's scene (a
training frame, a render in that scene) is scored with that episode left out of every statistic.

Detectors, fixed before any test pose was scored:
  paper              the paper's LLMD: mean and covariance over all 35,808 training frames, ridge 1e-6, in
                     coordinates where every channel has unit variance (so the ridge means the same for each)
  paper_raw          the same in the feature's own units (the literal recipe); many channels of this model have a
                     variance below 1e-6, so there the ridge is not negligible
  paper_cv           the same with the ridge chosen on training episodes only (5-fold held-out likelihood)
  start_tied         start-referenced, tied covariance: mean of the 99 training START frames, covariance of all
                     training frames (full rank)
  start_lw           start-referenced, Ledoit-Wolf shrinkage covariance of the 99 start frames alone
  start_phase        start mean, covariance pooled within 10 phases of the demo (the approach motion taken out)
  start_tied_render, start_lw_render   as start_tied / start_lw with the 99 start RE-RENDERS as the reference
  token_match        experiment 3's best signal (layer-8 image tokens matched to the training starts), recomputed
An observation is accepted when its score is at most the detector's threshold: the split-conformal 95 % quantile of
the 99 re-rendered training starts' own leave-one-episode-out scores (same image domain as the test poses). The
re-rendered starts therefore are never counted as accepted cases.

Pass / fail on experiment 3's 281 poses, fixed in advance (see RULE): far vs close passes with at most 5 % of the
far poses accepted, at least 10 of the 12 in-range lattice poses accepted and AUROC (in range vs far) >= 0.95; it
fails with >= 50 % of the far poses accepted or AUROC < 0.80; anything else "ranks but cannot set a limit". Rotation
passes with at least 12 of the 13 headings turned >= 20 deg rejected and the +5 deg heading accepted.

Written to <out>/scores/: summary.json (everything the page shows), scores.npz (every detector's score for every
observation), table.csv (one line per detector).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve_triangular
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from llmd_stats import (LAMBDA, auroc, conformal_tau, corr_eig, corr_factor, cov_factor, cov_from_moments, lw_fit,  # noqa: E402
                        lw_score, maha, wilson)

RULE = dict(far_accept_max=0.05, in_range_lattice_min=10 / 12, auroc_min=0.95, fail_far_accept=0.50, fail_auroc=0.80,
            turned_deg=20.0, turned_rejected_min=12 / 13, alpha=0.05, band_m=0.10)
PHASES = 10
LAM_GRID = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
CV_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CV_TOKENS = (0, 7, 15, 22, 29)
FULL = ("paper", "paper_raw", "paper_cv", "start_tied", "start_tied_render", "start_phase", "start_lw", "start_lw_render")
LIGHT = ("paper", "start_tied", "start_lw")
LABEL = dict(paper="Paper LLMD", paper_raw="Paper LLMD, raw feature units", paper_cv="Paper LLMD, ridge chosen on training data",
             start_tied="Start-referenced, tied covariance", start_lw="Start-referenced, Ledoit-Wolf",
             start_phase="Start-referenced, within-phase covariance",
             start_tied_render="Start-referenced, tied, re-rendered starts as reference",
             start_lw_render="Start-referenced, Ledoit-Wolf, re-rendered starts as reference",
             token_match="Token match, layer 8 (experiment 3)")
G: dict = {}  # a worker's data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the experiment directory (feats/ is read, scores/ written)")
    p.add_argument("--exp3", required=True)
    p.add_argument("--suffix", default="", help="read feats<suffix>/ and write scores<suffix>/ (a pilot)")
    p.add_argument("--workers", type=int, default=5, help="tokens fitted in parallel (each uses the BLAS threads)")
    p.add_argument("--variants", default="s0,s1,s2,s0_bf16", help="feature files h_<variant>.npy; the first is the primary")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--pilot", action="store_true", help="few tokens and keys: checks the plumbing, not the result")
    return p.parse_args()


# ------------------------------------------------------------------------------------------------ workers
def init(cfg: dict) -> None:
    G.clear()
    G.update(cfg)
    feats, v = Path(cfg["feats"]), cfg["variant"]
    G["train"] = np.load(feats / "train" / f"h_{v}.npy", mmap_mode="r")
    G["tests"] = {name: np.load(feats / name / f"h_{v}.npy", mmap_mode="r") for name in cfg["sets"]}


def token_data(s: int):
    """The training features of token s, centred on their mean (float64), and that mean."""
    X = np.asarray(G["train"][:, s, :], dtype=np.float64)
    c = X.mean(0)
    X -= c
    return X, c


def token_job(s: int) -> dict:
    """Every detector's squared distance at action token s, for every observation, each with its own left-out
    episode."""
    t0 = time.perf_counter()
    X, c = token_data(s)
    n, D = X.shape
    S, Pm = X.sum(0), X.T @ X
    bounds, keys, dets, lam = G["bounds"], G["keys"], G["dets"], G["lam"]
    tests = {name: np.asarray(a[:, s, :], dtype=np.float64) - c for name, a in G["tests"].items()}
    X0 = X[[b[0] for b in bounds]]  # (99, D) the training start frames, row = episode
    Rr = tests["start"][G["start_order"]] if "start" in tests and G.get("start_order") is not None else None
    ph = G["phase"]
    Sb = np.stack([X[ph == b].sum(0) for b in range(PHASES)])
    nb = np.array([(ph == b).sum() for b in range(PHASES)], dtype=np.float64)
    a_scale = G["scale"]
    out = {d: {name: np.full(len(a), np.nan) for name, a in tests.items()} for d in dets}
    for d in dets:
        out[d]["train"] = np.full(n, np.nan)
    extra: dict = {}

    def without(M: np.ndarray, e: int) -> np.ndarray:
        return M if e < 0 else np.delete(M, e, axis=0)

    for key in keys:
        # an episode the features do not cover (a pilot on a slice of the training set) is scored with nothing left out
        e = key if key < len(bounds) else -1
        if e >= 0:
            a, b = bounds[e]
            Xe = X[a:b]
            ne, Se, Pe = b - a, Xe.sum(0), Xe.T @ Xe
            Sbe = np.stack([Xe[ph[a:b] == k].sum(0) for k in range(PHASES)])
            nbe = np.array([(ph[a:b] == k).sum() for k in range(PHASES)], dtype=np.float64)
        else:
            ne, Se, Pe, Sbe, nbe = 0, 0.0, 0.0, 0.0, 0.0
        n1 = n - ne
        mu, C = cov_from_moments(n1, S - Se, Pm - Pe)
        L1, sd = corr_factor(C, lam)
        blocks = [(name, idx, tests[name][idx]) for name, idx in G["by_key"].get(key, [])]
        if e >= 0 and G["score_train"]:
            blocks.append(("train", np.arange(a, b), Xe))
        mu0 = without(X0, e).mean(0)
        mur = without(Rr, e).mean(0) if Rr is not None else None
        for d in dets:
            if d == "paper":
                f = lambda Xt: maha(L1, (Xt - mu) / sd)
            elif d == "start_tied":
                f = lambda Xt: maha(L1, (Xt - mu0) / sd)
            elif d == "start_tied_render":
                if mur is None:
                    continue
                f = lambda Xt: maha(L1, (Xt - mur) / sd)
            elif d == "paper_raw":  # f = h * scale + shift: the shift cancels in x - mu
                L2 = cov_factor(C, lam, a_scale)
                f = lambda Xt: maha(L2, (Xt - mu) * a_scale)
            elif d == "paper_cv":
                L3, _ = corr_factor(C, G["lam_cv"])
                f = lambda Xt: maha(L3, (Xt - mu) / sd)
            elif d == "start_phase":
                nb1, Sb1 = nb - nbe, Sb - Sbe
                W = Pm - Pe - (Sb1.T / np.maximum(nb1, 1.0)) @ Sb1  # scatter about each phase's own mean
                L4, sdw = corr_factor(W / (n1 - PHASES), lam)
                f = lambda Xt: maha(L4, (Xt - mu0) / sdw)
            elif d == "start_lw":
                fit = lw_fit(without(X0, e))
                extra.setdefault("lw_shrink", []).append(fit["shrink"])
                f = lambda Xt: lw_score(fit, Xt)
            elif d == "start_lw_render":
                if Rr is None:
                    continue
                fitr = lw_fit(without(Rr, e))
                extra.setdefault("lw_render_shrink", []).append(fitr["shrink"])
                f = lambda Xt: lw_score(fitr, Xt)
            else:
                raise KeyError(d)
            for name, idx, Xt in blocks:
                out[d][name][idx] = f(Xt)
        if e == G["grid_ep"] and G["sweep"]:
            # the grid scene's statistics: spectrum, the ridge sweep, and the moments for the pooled-token ablation
            w, V, sd2 = corr_eig(C)
            wraw = eigh(C * np.outer(a_scale, a_scale), eigvals_only=True, check_finite=False)
            extra["spectrum"] = dict(corr=np.quantile(w, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist(), corr_below_ridge=int((w < lam).sum()),
                                     corr_below_1e_4=int((w < 1e-4).sum()), corr_below_1e_2=int((w < 1e-2).sum()),
                                     corr_participation=float(w.sum() ** 2 / (w ** 2).sum()),
                                     raw=np.quantile(wraw, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist(),
                                     raw_below_ridge=int((wraw < lam).sum()), raw_channel_var_below_ridge=int((np.diag(C) * a_scale ** 2 < lam).sum()),
                                     h_channel_sd=np.quantile(sd, [0, 0.5, 1]).tolist())
            sweep = {}
            for name, idx in G["by_key"].get(key, []):
                if name not in G["sweep_sets"]:
                    continue
                Xt = tests[name][idx]
                y, y0 = ((Xt - mu) / sd2) @ V, ((Xt - mu0) / sd2) @ V
                sweep[name] = dict(idx=idx, paper=np.stack([(y ** 2 / (w + l)).sum(1) for l in LAM_GRID]),
                                   start_tied=np.stack([(y0 ** 2 / (w + l)).sum(1) for l in LAM_GRID]))
            extra["sweep"] = sweep
            extra["pooled"] = dict(n=n1, mean=mu + c, cov=C)
        if key < 0 and G["identity"] and s == 0:
            # in-sample: mean d2 must equal (n - 1) / n * tr((R + lam I)^-1 R) = (n - 1) / n * (D - lam |L^-1|_F^2)
            Li = solve_triangular(L1, np.eye(D), lower=True, check_finite=False)
            extra["identity"] = dict(mean_in_sample_d2=float(maha(L1, (X - mu) / sd).mean()),
                                     expected=float((n - 1) / n * (D - lam * (Li ** 2).sum())), D=D, n=n,
                                     cov_dtype=str(C.dtype), factor_dtype=str(L1.dtype), ridge=lam)
    extra["seconds"] = time.perf_counter() - t0
    return dict(s=s, out=out, extra=extra)


def cv_job(s: int) -> dict:
    """Held-out Gaussian negative log-likelihood of the training frames at token s for each ridge of CV_GRID:
    5 folds of episodes, statistics from the other folds."""
    X, _ = token_data(s)
    n = len(X)
    S, Pm = X.sum(0), X.T @ X
    fold = np.concatenate([np.full(b - a, e % 5) for e, (a, b) in enumerate(G["bounds"])])
    nll = np.zeros(len(CV_GRID))
    for k in range(5):
        Xk = X[fold == k]
        mu, C = cov_from_moments(n - len(Xk), S - Xk.sum(0), Pm - Xk.T @ Xk)
        w, V, sd = corr_eig(C)
        y2 = (((Xk - mu) / sd) @ V) ** 2
        for j, lam in enumerate(CV_GRID):
            nll[j] += 0.5 * (len(Xk) * np.log(w + lam).sum() + (y2 / (w + lam)).sum())
    return dict(s=s, nll=(nll / n).tolist())


# ------------------------------------------------------------------------------------------------ helpers
def load_rows(feats: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in open(feats / name / "obs.jsonl")]


def key_of(r: dict) -> int:
    v = r.get("left_out")
    return -1 if v is None else int(v)


def run_variant(feats: Path, variant: str, sets: list[str], rows: dict, dets: tuple, base: dict, workers: int,
                tokens: list[int], keys: list[int] | None = None, score_train: bool = True, sweep: bool = False,
                identity: bool = False) -> tuple[dict, dict]:
    """scores[detector][set] = (n, tokens) squared distances; extras per token."""
    by_key: dict = {}
    for name in sets:
        k = np.array([key_of(r) for r in rows[name]])
        for e in np.unique(k):
            by_key.setdefault(int(e), []).append((name, np.nonzero(k == e)[0]))
    all_keys = sorted(set(by_key) | ({-1} if identity else set()) | (set(range(len(base["bounds"]))) if score_train else set()))
    cfg = dict(base, feats=str(feats), variant=variant, sets=sets, dets=dets, by_key=by_key,
               keys=[k for k in all_keys if keys is None or k in keys], score_train=score_train, sweep=sweep, identity=identity)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(min(workers, len(tokens)), initializer=init, initargs=(cfg,)) as pool:
        res = pool.map(token_job, tokens, chunksize=1)
    scores = {d: {name: np.stack([r["out"][d][name] for r in res], axis=1) for name in res[0]["out"][d]
                  if score_train or name != "train"} for d in dets}
    extras = {r["s"]: r["extra"] for r in res}
    print(f"[fit] {variant}: {len(dets)} detectors x {len(tokens)} tokens x {len(cfg['keys'])} left-out sets in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return scores, extras


def reduce_max(per_token: np.ndarray) -> np.ndarray:
    return per_token.max(axis=1)


class Grid:
    """Experiment 3's labels of the 281 poses."""

    def __init__(self, rows: list[dict], band: float):
        self.rows = rows
        self.kind = np.array([r["kind"] for r in rows])
        self.pose_dist = np.array([r["pose_dist"] for r in rows], dtype=float)
        self.dyaw = np.array([r["dyaw"] for r in rows], dtype=float)
        self.fx = np.array([r["fx"] for r in rows], dtype=float)
        self.fy = np.array([r["fy"] for r in rows], dtype=float)
        in_range = np.array([bool(r["in_range"]) for r in rows])
        self.ev = self.kind != "anchor"
        self.anchor = ~self.ev
        self.inr = self.ev & in_range
        self.far = self.ev & (self.pose_dist > band)
        self.lattice = np.isin(self.kind, ("fine", "medium", "coarse"))
        self.inr_lat = self.inr & self.lattice
        self.yaw = self.kind == "yaw"
        self.turned = self.yaw & (np.abs(self.dyaw) >= RULE["turned_deg"] - 1e-9)
        self.plus5 = self.yaw & (np.abs(self.dyaw - 5.0) < 1e-6)
        self.bands = [("in the training range", self.inr), (f"up to {band * 100:.0f} cm outside", self.ev & ~in_range & ~self.far),
                      (f"{band * 100:.0f}-30 cm outside", self.far & (self.pose_dist <= 0.3)),
                      ("0.3-1 m outside", self.ev & (self.pose_dist > 0.3) & (self.pose_dist <= 1.0)),
                      ("more than 1 m outside", self.ev & (self.pose_dist > 1.0))]

    def counts(self) -> dict:
        return dict(poses=len(self.rows), anchors=int(self.anchor.sum()), in_range=int(self.inr.sum()),
                    in_range_lattice=int(self.inr_lat.sum()), far=int(self.far.sum()), headings=int(self.yaw.sum()),
                    turned=int(self.turned.sum()), bands={name: int(m.sum()) for name, m in self.bands})

    def evaluate(self, s: np.ndarray, tau: float) -> dict:
        acc = s <= tau
        pair = self.inr | self.far
        # threshold-free operating point: the smallest threshold that accepts all but one in-range pose
        srt = np.sort(s[self.inr])
        tau_o = float(srt[-2]) if len(srt) > 1 else float(srt[-1])
        ev = dict(
            tau=float(tau), auroc_far=auroc(s[pair], self.inr[pair]), auroc_all=auroc(s[self.ev], self.inr[self.ev]),
            rho=float(spearmanr(s[self.ev], self.pose_dist[self.ev]).statistic),
            in_range=[int(acc[self.inr].sum()), int(self.inr.sum())],
            in_range_lattice=[int(acc[self.inr_lat].sum()), int(self.inr_lat.sum())],
            far=[int(acc[self.far].sum()), int(self.far.sum())],
            by_band=[[int(acc[m].sum()), int(m.sum())] for _, m in self.bands],
            headings_accepted=sorted(float(v) for v in self.dyaw[self.yaw & acc]),
            turned_rejected=[int((~acc[self.turned]).sum()), int(self.turned.sum())],
            plus5_accepted=bool(acc[self.plus5].all()) if self.plus5.any() else None,
            anchors=[int(acc[self.anchor].sum()), int(self.anchor.sum())],
            # repeat renders of the training start pose: how much render noise alone moves the score
            anchor_scores=np.sort(s[self.anchor]).tolist(),
            far_at_oracle=[int((s[self.far] <= tau_o).sum()), int(self.far.sum())], tau_oracle=tau_o,
            median=dict(in_range=float(np.median(s[self.inr])), far=float(np.median(s[self.far]))))
        # relative to this scene's own training start (the median of its repeat renders): how much moving the robot
        # changes the score, with the scene held fixed
        if self.anchor.any():
            a0 = float(np.median(s[self.anchor]))
            ev["rel_start"] = dict(
                start=a0, repeat_renders=[float(s[self.anchor].min() / a0), float(s[self.anchor].max() / a0)],
                by_band=[float(np.median(s[m]) / a0) if m.any() else None for _, m in self.bands],
                band_range=[[float(s[m].min() / a0), float(s[m].max() / a0)] if m.any() else None for _, m in self.bands],
                headings={f"{v:g}": float(x / a0) for v, x in sorted(zip(self.dyaw[self.yaw], s[self.yaw]))})
        nf, nl = ev["far"][1], ev["in_range_lattice"][1]
        far_ok = ev["far"][0] <= math.floor(RULE["far_accept_max"] * nf + 1e-9)
        lat_ok = ev["in_range_lattice"][0] >= math.ceil(RULE["in_range_lattice_min"] * nl - 1e-9)
        if far_ok and lat_ok and ev["auroc_far"] >= RULE["auroc_min"]:
            fc = "pass"
        elif ev["far"][0] >= RULE["fail_far_accept"] * nf or ev["auroc_far"] < RULE["fail_auroc"]:
            fc = "fail"
        else:
            fc = "ranks"
        rot_ok = (ev["turned_rejected"][0] >= math.ceil(RULE["turned_rejected_min"] * ev["turned_rejected"][1] - 1e-9)
                  and bool(ev["plus5_accepted"]))
        ev["verdict"] = dict(far_vs_close=fc, rotation="pass" if rot_ok else "fail")
        return ev


def token_match(feats: Path, rows: dict, names: list[str]) -> dict:
    """Experiment 3's token-match score at layer 8 for the sets that kept their image tokens: the mean over the
    image tokens of the cosine distance to the nearest training-start token at the same position (channels with a
    massive mean dropped, tokens centred on the training starts' mean)."""
    sys.path.insert(0, str(HERE.parent / "vlm_camera_sweep"))
    from plot_indist_signal import token_novelty  # experiment 3's own function

    ref = np.load(feats / "trainstart" / "tok8.npy").astype(np.float32)
    ref_ep = np.array([r["episode"] for r in rows["trainstart"]])
    absmean = np.abs(ref).mean((0, 1))
    keep = ~(absmean > 8 * np.median(absmean))
    out = {}
    # the training starts against each other, then every other set against them
    T = ref[:, :, keep]
    idx = np.arange(len(ref))
    out["trainstart"] = token_novelty(T, idx, ref_ep[None, :] != ref_ep[:, None]).mean(1)
    for name in names:
        Q = np.load(feats / name / "tok8.npy").astype(np.float32)[:, :, keep]
        k = np.array([key_of(r) for r in rows[name]])
        allowed = np.concatenate([np.ones((len(ref), len(ref)), bool), ref_ep[None, :] != k[:, None]])
        out[name] = token_novelty(np.concatenate([T, Q]), idx, allowed).mean(1)[len(ref):]
    out["_massive_channels"] = int((~keep).sum())
    return out


def knn_ladder(feats: Path, variant: str, rows: dict, grid: Grid, bounds: list) -> dict:
    """The step between experiment 3 and LLMD: the same kNN score experiment 3 used, on the LLMD feature."""
    sys.path.insert(0, str(HERE.parent / "vlm_camera_sweep"))
    from plot_indist_signal import gram, knn_score

    tr = np.load(feats / "train" / f"h_{variant}.npy", mmap_mode="r")
    ref = np.asarray(tr[[b[0] for b in bounds]], dtype=np.float32)
    q = np.load(feats / "grid" / f"h_{variant}.npy").astype(np.float32)
    K = gram(np.concatenate([ref, q]))
    r = np.arange(len(ref))
    k_grid = np.array([key_of(x) for x in rows["grid"]])
    allowed = np.concatenate([~np.eye(len(ref), dtype=bool), r[None, :] != k_grid[:, None]])
    s = knn_score(K, r, allowed, 3)
    tau = float(np.percentile(s[:len(ref)], 95))
    return grid.evaluate(s[len(ref):], tau)


def ablation_from_moments(feats: Path, name: str, rows: dict, grid: Grid, grid_ep: int) -> dict | None:
    """Paper LLMD on an ablated feature: statistics from the accumulated moments of the training frames with the
    grid scene's episode removed; judged on the grid without a threshold."""
    f = feats / "train" / f"mom_{name}.npz"
    if not f.exists():
        return None
    m = np.load(f)
    ep = np.load(feats / "train" / f"f_{name}_ep.npy").astype(np.float64)
    q = np.load(feats / "grid" / f"f_{name}.npy").astype(np.float64)
    T = q.shape[1]
    d2 = np.zeros((len(q), T))
    for s in range(T):
        Xe = ep[:, s]
        mu, C = cov_from_moments(int(m["n"]) - len(Xe), m["s"][s] - Xe.sum(0), m["P"][s] - Xe.T @ Xe)
        L, sd = corr_factor(C, LAMBDA)
        d2[:, s] = maha(L, (q[:, s] - mu) / sd)
    assert all(key_of(r) == grid_ep for r in rows["grid"])
    return grid.evaluate(d2.max(1), float("nan"))


def pooled_ablation(extras: dict, feats: Path, variant: str, grid: Grid) -> dict:
    """One mean and covariance shared by all 30 tokens (the paper's 'global' statistics), still the max over tokens."""
    toks = sorted(extras)
    means = np.stack([extras[s]["pooled"]["mean"] for s in toks])
    m = means.mean(0)
    C = sum(extras[s]["pooled"]["cov"] * (extras[s]["pooled"]["n"] - 1) / extras[s]["pooled"]["n"] + np.outer(mu - m, mu - m)
            for s, mu in zip(toks, means)) / len(toks)
    L, sd = corr_factor(C, LAMBDA)
    q = np.load(feats / "grid" / f"h_{variant}.npy")
    d2 = np.stack([maha(L, (q[:, s].astype(np.float64) - m) / sd) for s in toks], axis=1)
    return grid.evaluate(d2.max(1), float("nan"))


def paired_auroc(score: np.ndarray, success: np.ndarray, mask: np.ndarray, group: np.ndarray) -> dict:
    """Probability that a success scores lower than a failure, over the pairs that share a group (a scene, a start
    condition): what is left of the link once the groups' own score levels and success rates are taken out."""
    wins, pairs = 0.0, 0
    for g in sorted(set(group[mask].tolist())):
        k = mask & (group == g)
        a, b = score[k & success], score[k & ~success]
        wins += float((a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum())
        pairs += len(a) * len(b)
    return dict(auroc=wins / pairs if pairs else float("nan"), pairs=int(pairs))


def thirds(score: np.ndarray, success: np.ndarray) -> list:
    """Success rate in the lowest, middle and highest third of the scores."""
    order = np.argsort(score, kind="stable")
    return [[int(success[p].sum()), int(len(p))] for p in np.array_split(order, 3)]


def main() -> None:
    args = parse_args()
    out, exp3 = Path(args.out), Path(args.exp3)
    feats, dest = out / f"feats{args.suffix}", out / f"scores{args.suffix}"
    dest.mkdir(parents=True, exist_ok=True)
    run = json.loads((feats / "run.json").read_text())
    grid_ep, band = int(run["grid_episode"]), RULE["band_m"]
    variants = [v for v in args.variants.split(",") if (feats / "train" / f"h_{v}.npy").exists()]
    primary = variants[0]
    started = time.perf_counter()

    # ---- the observation tables
    names = sorted(p.parent.name for p in feats.glob("*/obs.jsonl") if (p.parent / f"h_{primary}.npy").exists())
    # a set whose extraction did not finish (interrupted, or restarted for new rows) must not be scored
    for name in names:
        pa, pb = feats / name / "progress_a.json", feats / name / "progress_b.json"
        a = json.loads(pa.read_text()) if pa.exists() else {}
        b = json.loads(pb.read_text()) if pb.exists() else {}
        n_rows = sum(1 for _ in open(feats / name / "obs.jsonl"))
        if not (a.get("done") == a.get("n") == b.get("n") == n_rows and a.get("fingerprint") == b.get("fingerprint")):
            raise SystemExit(f"{name}: its features are not finished for its {n_rows} observations "
                             f"(phase A {a.get('done')}/{a.get('n')}, phase B {b.get('n')}); run the extraction again")
    rows = {name: load_rows(feats, name) for name in names}
    tr = rows["train"]
    ep_of = np.array([r["episode"] for r in tr])
    episodes = sorted(set(ep_of.tolist()))
    assert episodes == list(range(len(episodes))), "training episodes must be 0..n-1"
    bounds = []
    for e in episodes:
        idx = np.nonzero(ep_of == e)[0]
        assert idx[-1] - idx[0] + 1 == len(idx), f"episode {e}: its frames are not contiguous"
        bounds.append((int(idx[0]), int(idx[-1]) + 1))
    phase = np.minimum((np.array([r["frame"] for r in tr]) * PHASES) // np.array([r["length"] for r in tr]), PHASES - 1)
    test_sets = [n for n in names if n not in ("train", "trainstart")]
    start_order = None
    if "start" in rows:  # row of the re-rendered start of each episode
        pos = {r["episode"]: i for i, r in enumerate(rows["start"])}
        if len(pos) == len(episodes):
            start_order = [pos[e] for e in episodes]
    ad = np.load(feats / "adaln.npz")
    base = dict(bounds=bounds, phase=phase, scale=ad["scale_1000"].astype(np.float64), lam=LAMBDA, lam_cv=LAMBDA,
                grid_ep=grid_ep, start_order=start_order, sweep_sets=("grid", "extra"))
    T = np.load(feats / "train" / f"h_{primary}.npy", mmap_mode="r").shape[1]
    tokens = list(range(T)) if not args.pilot else [0, T - 1]
    grid = Grid(rows["grid"], band)
    print(f"[fit] {len(tr)} training frames of {len(episodes)} episodes; sets {', '.join(f'{n} {len(rows[n])}' for n in test_sets)}; "
          f"variants {variants}", flush=True)

    # ---- the ridge of paper_cv: chosen on training episodes only
    cfg = dict(base, feats=str(feats), variant=primary, sets=[])
    with mp.get_context("spawn").Pool(min(args.workers, len(CV_TOKENS)), initializer=init, initargs=(cfg,)) as pool:
        cv = pool.map(cv_job, [t for t in CV_TOKENS if t < T][:2 if args.pilot else None], chunksize=1)
    nll = np.sum([c["nll"] for c in cv], axis=0)
    lam_cv = float(CV_GRID[int(np.argmin(nll))])
    base["lam_cv"] = lam_cv
    print(f"[fit] ridge chosen by 5-fold held-out likelihood over tokens {[c['s'] for c in cv]}: {lam_cv:g} "
          f"(nll per frame {dict(zip(CV_GRID, np.round(nll, 1).tolist()))})", flush=True)

    # ---- the primary variant: every detector, every observation, leave-one-episode-out
    keys = None if not args.pilot else [-1, grid_ep, 1, 2]
    dets = tuple(d for d in FULL if start_order is not None or not d.endswith("_render"))
    per_tok, extras = run_variant(feats, primary, test_sets, rows, dets, base, args.workers, tokens, keys=keys,
                                  score_train=True, sweep=True, identity=True)
    score = {d: {name: reduce_max(a) for name, a in per_tok[d].items()} for d in dets}
    argmax = {d: {name: a.argmax(1).astype(np.int16) for name, a in per_tok[d].items()} for d in dets}
    mean_tok = {name: a.mean(1) for name, a in per_tok["paper"].items()}

    # ---- token match (experiment 3's signal) on the sets that kept their layer-8 tokens
    tm_sets = [n for n in test_sets if (feats / n / "tok8.npy").exists()]
    tm = token_match(feats, rows, tm_sets)
    massive = tm.pop("_massive_channels")
    score["token_match"] = tm
    dets_all = dets + ("token_match",)

    # ---- thresholds: the conformal 95 % quantile of the re-rendered training starts (same domain as the test poses)
    def tau_of(d: str, suffix: str = "") -> dict:
        s = score[d]
        t = dict()
        if f"start{suffix}" in s and not np.isnan(s[f"start{suffix}"]).any():
            t["render"] = conformal_tau(s[f"start{suffix}"], RULE["alpha"])
        video = s["trainstart"] if d == "token_match" else (s["train"][[b[0] for b in bounds]] if "train" in s else None)
        if video is not None and not np.isnan(video).any():
            t["video"] = conformal_tau(video, RULE["alpha"])
            t["video_p95"] = float(np.percentile(video, 95))  # experiment 3's rule
        return t

    summary: dict = dict(
        task=run.get("task", "G1WholebodyOpenFaucetTeleop-v0"), run_dir=run["run_dir"], ckpt_step=run["ckpt_step"],
        instruction=run["instruction"], grid_episode=grid_ep, primary_variant=primary, rule=RULE, labels=LABEL,
        counts=dict(train_frames=len(tr), train_episodes=len(episodes), grid=grid.counts(),
                    sets={n: len(rows[n]) for n in names}),
        feature=dict(tokens=T, dim=int(base["scale"].shape[0]), timestep=run["timestep"], seeds=run["seeds"]),
        pilot=bool(args.pilot), detectors={})

    def frac(mask_scores: np.ndarray, tau: float) -> list:
        ok = ~np.isnan(mask_scores)
        return [int((mask_scores[ok] <= tau).sum()), int(ok.sum())]

    for d in dets_all:
        s = score[d]
        taus = tau_of(d)
        tau = taus.get("render", taus.get("video", float("nan")))
        e: dict = dict(label=LABEL[d], tau=taus, tau_used="render" if "render" in taus else "video",
                       grid=grid.evaluate(s["grid"], tau))
        if "video" in taus:
            e["grid_video_tau"] = grid.evaluate(s["grid"], taus["video"])
        if "start" in s:
            q = np.nanpercentile(s["start"], [0, 5, 25, 50, 75, 95, 100])
            # the training starts in their own scenes: how much the score differs between scenes at the same kind of pose
            e["start_render"] = dict(median=float(q[3]), q95=float(q[5]), min=float(q[0]), q05=float(q[1]), q25=float(q[2]),
                                     q75=float(q[4]), max=float(q[6]), spread_q95_over_q05=float(q[5] / q[1]),
                                     scores=np.sort(s["start"]).tolist())
        if "unused" in s:
            k, n = frac(s["unused"], tau)
            inr = np.array([bool(r.get("in_range", True)) for r in rows["unused"]])
            e["unused"] = dict(accepted=[k, n], ci=wilson(k, n), median=float(np.nanmedian(s["unused"])),
                               in_range_accepted=frac(s["unused"][inr], tau))
        if "extra" in s:
            kinds = np.array([r["kind"] for r in rows["extra"]])
            e["extra"] = {k: dict(accepted=frac(s["extra"][kinds == k], tau),
                                  items=[dict(id=r["id"], fx=r.get("fx"), fy=r.get("fy"), dyaw=r.get("dyaw"), in_range=r.get("in_range"),
                                              pose_dist=r.get("pose_dist"), score=float(v), accepted=bool(v <= tau))
                                         for r, v in zip(rows["extra"], s["extra"]) if r["kind"] == k])
                          for k in sorted(set(kinds.tolist()))}
        # the H.264 round trip of the same renders: the writer's settings (_h264), the crf that quantises a lone
        # frame like the training frames (_h264s), lossless = the colour conversion alone (_h264c)
        for suffix in ("_h264", "_h264s", "_h264c"):
            if f"grid{suffix}" not in s:
                continue
            th = tau_of(d, suffix).get("render", float("nan"))
            raw, cod = s["grid"], s[f"grid{suffix}"]
            a_raw, a_cod = raw <= tau, cod <= th
            h = dict(tau=float(th), grid=grid.evaluate(cod, th), grid_at_raw_tau=grid.evaluate(cod, tau),
                     spearman=float(spearmanr(raw[grid.ev], cod[grid.ev]).statistic),
                     median_log_ratio=float(np.median(np.log(cod[grid.ev] / raw[grid.ev]))),
                     iqr_log_ratio=np.percentile(np.log(cod[grid.ev] / raw[grid.ev]), [25, 75]).tolist(),
                     decisions_changed=[int((a_raw != a_cod)[grid.ev].sum()), int(grid.ev.sum())],
                     decisions_changed_at_raw_tau=[int((a_raw != (cod <= tau))[grid.ev].sum()), int(grid.ev.sum())])
            if "start" in s and f"start{suffix}" in s and start_order is not None:
                video = s["trainstart"] if d == "token_match" else s["train"][[b[0] for b in bounds]]
                rr, rc = s["start"][start_order], s[f"start{suffix}"][start_order]
                h["starts"] = dict(median_video=float(np.median(video)), median_render=float(np.median(rr)),
                                   median_render_codec=float(np.median(rc)),
                                   median_log_render_over_video=float(np.median(np.log(rr / video))),
                                   median_log_codec_over_video=float(np.median(np.log(rc / video))),
                                   render_above_video=[int((rr > video).sum()), len(video)],
                                   codec_above_video=[int((rc > video).sum()), len(video)])
            e[suffix[1:]] = h
        # the same renders with another VLM batch size (bf16 rounding differs), against the batch-1 statistics
        if "grid_b16" in s:
            b16 = s["grid_b16"]
            e["batch16"] = dict(grid=grid.evaluate(b16, tau),
                                median_log_ratio=float(np.median(np.log(b16[grid.ev] / s["grid"][grid.ev]))),
                                spearman=float(spearmanr(b16[grid.ev], s["grid"][grid.ev]).statistic),
                                decisions_changed=[int(((b16 <= tau) != (s["grid"] <= tau))[grid.ev].sum()), int(grid.ev.sum())])
        # controls: only detectors with a header feature
        if d != "token_match":
            if "other" in s:
                tv = taus.get("video", tau)  # other tasks' frames are video frames
                e["other_task"] = dict(accepted_video_tau=frac(s["other"], tv), accepted_render_tau=frac(s["other"], tau),
                                       min_over_tau_video=float(np.nanmin(s["other"]) / tv), median=float(np.nanmedian(s["other"])),
                                       by_pack={p: frac(s["other"][np.array([r["pack"] == p for r in rows["other"]])], tv)
                                                for p in sorted({r["pack"] for r in rows["other"]})})
            if "swap" in s and "train" in s:
                sw = rows["swap"]
                val, base_s, base_l = {}, [], []
                for kind in ("start_image_late_state", "late_image_start_state"):
                    m = np.array([r["kind"] == kind for r in sw])
                    val[kind] = s["swap"][m]
                for r in sw:
                    if r["kind"] == "start_image_late_state":
                        # (nan where the features do not cover the episode: a pilot on a slice of the training set)
                        a0, b0 = bounds[r["episode"]] if r["episode"] < len(bounds) else (0, 0)
                        base_s.append(s["train"][a0 + r["image_frame"]] if a0 + r["image_frame"] < b0 else np.nan)
                        base_l.append(s["train"][a0 + r["state_frame"]] if a0 + r["state_frame"] < b0 else np.nan)
                base_s, base_l = np.array(base_s), np.array(base_l)
                ok = ~np.isnan(base_s) & ~np.isnan(base_l) & ~np.isnan(val["start_image_late_state"])
                if ok.any():
                    e["swap"] = dict(
                        n=int(ok.sum()), median_start=float(np.median(base_s[ok])), median_late=float(np.median(base_l[ok])),
                        median_start_image_late_state=float(np.median(val["start_image_late_state"][ok])),
                        median_late_image_start_state=float(np.median(val["late_image_start_state"][ok])),
                        # how much the score of a start frame rises when only one input is replaced by the late one
                        log_rise_state_swapped=float(np.median(np.log(val["start_image_late_state"][ok] / base_s[ok]))),
                        log_rise_image_swapped=float(np.median(np.log(val["late_image_start_state"][ok] / base_s[ok]))),
                        log_rise_both=float(np.median(np.log(base_l[ok] / base_s[ok]))))
            # earlier closed-loop trials: does a low score go with success? (ranking inside each set only)
            tri = {}
            for name in ("traj", "traj_ms", "target", "target_tf", "target_ms"):
                if name not in s:
                    continue
                succ = np.array([bool(r["success"]) for r in rows[name]])
                stand = np.array([bool(r.get("standing_start", True)) for r in rows[name]])
                scene = np.array([str(r.get("scene")) for r in rows[name]])
                cond = np.array([str(r.get("condition")) for r in rows[name]])
                ok = ~np.isnan(s[name])
                part = {}
                for label, m in (("all", ok), ("standing", ok & stand), ("takeover", ok & ~stand)):
                    if m.sum() >= 6 and succ[m].any() and (~succ[m]).any():
                        part[label] = dict(n=int(m.sum()), successes=int(succ[m].sum()), auroc=auroc(s[name][m], succ[m]),
                                           thirds=thirds(s[name][m], succ[m]),
                                           # success vs failure compared only inside one scene / one start condition
                                           auroc_within_scene=paired_auroc(s[name], succ, m, scene),
                                           auroc_within_condition=paired_auroc(s[name], succ, m, cond),
                                           by_condition={c: dict(n=int((m & (cond == c)).sum()), successes=int(succ[m & (cond == c)].sum()),
                                                                 median=float(np.median(s[name][m & (cond == c)])))
                                                         for c in sorted(set(cond[m].tolist()))},
                                           spearman=float(spearmanr(s[name][m], succ[m].astype(float)).statistic))
                tri[name] = part
            if tri:
                e["trials"] = tri
            if "train" in s and not np.isnan(s["train"]).any():
                tv = taus.get("video", float("nan"))
                e["train"] = dict(
                    accepted_by_phase=[frac(s["train"][phase == b], tv) for b in range(PHASES)],
                    median_by_phase=[float(np.median(s["train"][phase == b])) for b in range(PHASES)],
                    # the paper's time-constant band: the largest score of a demo, queried every 24 steps
                    demo_tau=conformal_tau(np.array([np.max(s["train"][a:b:24]) for a, b in bounds]), RULE["alpha"]))
                e["grid_demo_tau"] = grid.evaluate(s["grid"], e["train"]["demo_tau"])
        summary["detectors"][d] = e

    # ---- aggregation over tokens, spectrum, ridge
    spec = [extras[s]["spectrum"] for s in sorted(extras) if "spectrum" in extras[s]]
    ident = next((extras[s]["identity"] for s in sorted(extras) if "identity" in extras[s]), None)
    heldout = float(np.nanmean(per_tok["paper"]["train"])) if "train" in per_tok["paper"] else float("nan")
    # like for like with the in-sample identity, which is checked on the first token only
    heldout0 = float(np.nanmean(per_tok["paper"]["train"][:, 0])) if "train" in per_tok["paper"] else float("nan")
    summary["ridge"] = dict(
        value=LAMBDA, dtype="float64", factorisation="Cholesky of the 1536 x 1536 matrix + ridge * I (LAPACK dpotrf)",
        identity=ident,
        heldout_mean_d2_per_token=heldout, heldout_mean_d2_first_token=heldout0,
        heldout_over_in_sample=heldout0 / ident["mean_in_sample_d2"] if ident else None,
        corr_eig_min=float(min(s["corr"][0] for s in spec)) if spec else None,
        corr_eig_median=float(np.median([s["corr"][3] for s in spec])) if spec else None,
        corr_eigs_below_ridge=[int(s["corr_below_ridge"]) for s in spec],
        corr_eigs_below_1e_4=[int(s["corr_below_1e_4"]) for s in spec],
        raw_eig_min=float(min(s["raw"][0] for s in spec)) if spec else None,
        raw_eig_median=float(np.median([s["raw"][3] for s in spec])) if spec else None,
        raw_eigs_below_ridge=[int(s["raw_below_ridge"]) for s in spec],
        raw_channels_var_below_ridge=[int(s["raw_channel_var_below_ridge"]) for s in spec],
        cv=dict(grid=CV_GRID, nll_per_frame=nll.tolist(), chosen=lam_cv, tokens=[c["s"] for c in cv]),
        lw_shrinkage=dict(video=float(np.mean([np.mean(extras[s]["lw_shrink"]) for s in extras if "lw_shrink" in extras[s]])),
                          render=float(np.mean([np.mean(extras[s]["lw_render_shrink"]) for s in extras
                                                if "lw_render_shrink" in extras[s]])) if any("lw_render_shrink" in extras[s] for s in extras) else None))
    summary["token_match_massive_channels"] = massive

    # ridge sweep on the grid (threshold-free)
    sweep = {}
    for det in ("paper", "start_tied"):
        per_l = []
        for j, lam in enumerate(LAM_GRID):
            d2 = np.full((len(rows["grid"]), len(tokens)), np.nan)
            for ti, s in enumerate(sorted(extras)):
                sw = extras[s].get("sweep", {}).get("grid")
                if sw is not None:
                    d2[sw["idx"], ti] = sw[det][j]
            ev = grid.evaluate(d2.max(1), float("nan"))
            per_l.append(dict(ridge=lam, auroc_far=ev["auroc_far"], auroc_all=ev["auroc_all"], far_at_oracle=ev["far_at_oracle"], rho=ev["rho"]))
        sweep[det] = per_l
    summary["ridge_sweep"] = sweep
    summary["aggregation"] = dict(max=summary["detectors"]["paper"]["grid"],
                                  mean=grid.evaluate(mean_tok["grid"], conformal_tau(mean_tok["start"], RULE["alpha"])
                                                     if "start" in mean_tok and not np.isnan(mean_tok["start"]).any() else float("nan")),
                                  argmax_token_hist=np.bincount(argmax["paper"]["grid"], minlength=len(tokens)).tolist())

    # ---- the other noise seeds and the served precision: paper + the two start detectors
    reps = {}
    for v in variants[1:]:
        if args.pilot:
            break
        pt, _ = run_variant(feats, v, [n for n in ("grid", "start", "unused", "extra") if n in rows], rows, LIGHT, base,
                            args.workers, tokens, score_train=False)
        reps[v] = {}
        for d in LIGHT:
            sc = {name: reduce_max(a) for name, a in pt[d].items()}
            tau = conformal_tau(sc["start"], RULE["alpha"]) if "start" in sc else float("nan")
            reps[v][d] = dict(grid=grid.evaluate(sc["grid"], tau),
                              spearman_with_primary=float(spearmanr(sc["grid"], score[d]["grid"]).statistic))
            score[f"{d}@{v}"] = sc
    summary["replicates"] = reps

    # ---- ablations of the feature (threshold-free, on the grid)
    abl = {}
    if not args.skip_ablations and not args.pilot:
        for name, label in (("fresh", "new noise for every observation (timestep 1000)"),
                            ("t500", "fixed noise at timestep 500 (half-way)"),
                            ("rand", "random timestep and noise (paper App. C.1)")):
            ev = ablation_from_moments(feats, name, rows, grid, grid_ep)
            if ev is not None:
                abl[name] = dict(label=label, grid=ev)
        if all("pooled" in extras[s] for s in extras):
            abl["pooled_tokens"] = dict(label="one mean and covariance for all 30 tokens", grid=pooled_ablation(extras, feats, primary, grid))
        abl["knn_on_feature"] = dict(label="experiment 3's kNN score on the LLMD feature (99 starts as reference)",
                                     grid=knn_ladder(feats, primary, rows, grid, bounds))
    summary["ablations"] = abl
    s3 = json.loads((exp3 / "exp3_indist_signal" / "figures" / "summary.json").read_text())
    summary["exp3"] = dict(token_best=s3["signals"]["token"]["best"],
                           action_latent=[c for c in s3.get("action_expert", []) if c.get("latent") == "act"])
    hook = out / "checks" / "hook_check.json"
    if hook.exists():
        summary["hook_check"] = json.loads(hook.read_text())
    summary["seconds"] = time.perf_counter() - started

    np.savez_compressed(dest / "scores.npz", **{f"{d}/{name}": a for d, per in score.items() for name, a in per.items()},
                        **{f"argmax/{d}/{name}": a for d, per in argmax.items() for name, a in per.items()},
                        **{f"mean_tokens/{name}": a for name, a in mean_tok.items()})
    (dest / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with open(dest / "table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "far_vs_close", "rotation", "auroc_far", "auroc_all", "in_range", "in_range_lattice", "far",
                    "turned_rejected", "headings_accepted", "tau", "unused_accepted"])
        for d, e in summary["detectors"].items():
            g = e["grid"]
            w.writerow([d, g["verdict"]["far_vs_close"], g["verdict"]["rotation"], round(g["auroc_far"], 4), round(g["auroc_all"], 4),
                        "{}/{}".format(*g["in_range"]), "{}/{}".format(*g["in_range_lattice"]), "{}/{}".format(*g["far"]),
                        "{}/{}".format(*g["turned_rejected"]), " ".join(f"{v:+g}" for v in g["headings_accepted"]),
                        f"{g['tau']:.4g}", "{}/{}".format(*e["unused"]["accepted"]) if "unused" in e else ""])
    for d, e in summary["detectors"].items():
        g = e["grid"]
        print(f"[fit] {d:20s} far vs close {g['verdict']['far_vs_close']:5s} rotation {g['verdict']['rotation']:4s}  "
              f"AUROC {g['auroc_far']:.3f}  in range {g['in_range'][0]}/{g['in_range'][1]}  far {g['far'][0]}/{g['far'][1]}  "
              f"turned rejected {g['turned_rejected'][0]}/{g['turned_rejected'][1]}", flush=True)
    print(f"wrote {dest / 'summary.json'} in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
