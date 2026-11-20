"""Is there an in-distribution signal in the VLA's latent? Scores, clusters and figures (stage 3). Psi0 venv, CPU.

Every observation (a training start frame, or a render at a base pose of the plan) is one row of the latents
extract_grid_latents.py saved. The reference set R is the training start frames. For a latent, a score says how far
an observation is from R; it is computed for a grid render against R without the grid scene's own episode (the
scene is then as new to R as a test scene would be; "scene kept" numbers keep it) and for a training start against
the other episodes' starts. The 0-1 signal is score <= tau, with tau the --quantile-th percentile of the training
starts' own scores: the threshold comes from the training set alone.

Latents (per VLM layer):
  mean      the image tokens averaged                                              (step 3)
  attn      the image tokens averaged with the action tokens' attention on them    (steps 1 + 3)
  spatial   every image token kept, in place                                       (step 4)
  also attn_all / txt / last (other poolings), ctx / act (the action expert's context stream and action tokens)
Scores:
  knn       mean cosine distance to the --k nearest training starts, vectors centred on the training starts' mean
  pca       distance to the PCA subspace of the training starts (--pca components): what of the latent the
            training starts' own variation cannot explain
  token     spatial only: every image token is matched to the nearest training-start token at the same position
            (cosine, tokens centred per position); the score is the mean over the image, its map says which part
            of the image is unfamiliar. tokenw weights that mean with the action attention.
Massive-activation channels (extract stage) are left out of every VLM latent.

Ground truth of a pose (prepare_base_grid.py): pose_dist, how far the faucet seen from the pelvis is outside the
training range (0 inside). "Far" poses are more than --band outside; the poses in between are not held against a
signal either way. Per (latent, layer, score): AUROC of in-range against the rest, Spearman of the score against
pose_dist, the share accepted in range and of the far poses, and their mean (balanced accuracy).

Unsupervised (the "does it form two clusters" question): k-means, a Gaussian mixture, Ward linkage and spectral
clustering into 2 clusters, on the first --cluster-pcs principal components of the latent over all observations;
the cluster that holds most training starts is the "familiar" one.

Can a navigation policy follow the score? Greedy descent over the lattice of base poses at the training heading:
from a pose, step to the pose within 1.5 lattice steps that scores lowest, while that lowers the score. A pose "gets
back" when the descent ends in the training range.

Written:
  --report (default <out>/index.html)  the result page (indist_report.py): one conclusion per section over one simple
      chart, a pose's camera view on hover. Self-contained. --fragment also writes it without the document head.
  <out>/summary.json, candidates.csv   the numbers; every (latent, layer, score)
  <out>/1_setup.png ... 8_token_novelty.png   detail figures, every signal and layer (the page links them): 1 setup,
      2 the action expert's attention, 3 score maps around the training start, 4 score maps of the room and of the
      poses facing the faucet, 5 by layer, 6 score against pose distance and heading, 7 clusters, 8 which image
      tokens are unfamiliar
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import rankdata, spearmanr

import indist_report
from indist_common import (AXIS, C1, C2, C3, DIV, FLOOR_FILL, INK, INK2, MUTED, SEQ, SURFACE, draw_cells, draw_scene,
                           heading_xy, scene_handles, scene_labels, style, train_box)

C4, C5 = "#eda100", "#e87ba4"  # categorical slots 4 and 5
SIGNALS = (  # the signals the figures follow: key, latent, score, colour, label, short label
    ("mean", "mean", "knn", C1, "mean of the image tokens, kNN", "mean, kNN"),
    ("attn", "attn", "knn", C2, "action-attention-weighted mean, kNN", "attention, kNN"),
    ("meanpca", "mean", "pca", C3, "mean of the image tokens, PCA subspace", "mean, PCA"),
    ("pca", "spatial", "pca", C4, "all image tokens, PCA subspace", "tokens, PCA"),
    ("token", "spatial", "token", C5, "all image tokens, token match", "tokens, match"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renders", required=True, help="directory of render_base_grid.py and extract_grid_latents.py")
    p.add_argument("--prep", required=True, help="directory of prepare_base_grid.py")
    p.add_argument("--out", default=None, help="default: <renders>/figures")
    p.add_argument("--k", type=int, default=3, help="neighbours of the knn score")
    p.add_argument("--pca", type=int, default=32, help="components of the pca score")
    p.add_argument("--quantile", type=float, default=95.0, help="percentile of the training starts' scores = threshold")
    p.add_argument("--band", type=float, default=0.10, help="m; poses farther outside the training range are 'far'")
    p.add_argument("--cluster-pcs", type=int, default=10)
    p.add_argument("--report", default=None, help="the result page; default: <out>/index.html")
    p.add_argument("--fragment", default=None, help="also write the page without its document head to this file")
    p.add_argument("--thumb", type=int, default=320, help="px; width of the camera views embedded in the page")
    return p.parse_args()


# ----------------------------------------------------------------------------- scores
def gram(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    return X @ X.T


def knn_score(K: np.ndarray, ref: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance to the k nearest allowed references, in the space centred on the references' mean."""
    a = K[:, ref].mean(1)
    Kc = K - a[:, None] - a[None, :] + K[np.ix_(ref, ref)].mean()
    norm = np.sqrt(np.maximum(np.diag(Kc), 1e-12))
    dist = 1 - Kc[:, ref] / norm[:, None] / norm[ref][None, :]
    dist[~allowed] = np.inf
    return np.sort(dist, axis=1)[:, :k].mean(1)


def pca_score(K: np.ndarray, ref: np.ndarray, allowed: np.ndarray, m: int) -> np.ndarray:
    """Distance to the m-component PCA subspace of the allowed references (fitted without the left-out ones)."""
    out = np.empty(len(K))
    patterns, which = np.unique(allowed, axis=0, return_inverse=True)
    for j, pat in enumerate(patterns):
        rows, S = np.nonzero(which.ravel() == j)[0], ref[pat]
        a, c = K[:, S].mean(1), K[np.ix_(S, S)].mean()
        Kss = K[np.ix_(S, S)] - a[S][:, None] - a[S][None, :] + c
        lam, V = np.linalg.eigh(Kss)
        lam, V = lam[::-1], V[:, ::-1]
        keep = min(m, int((lam > lam[0] * 1e-10).sum()))
        Kqs = K[np.ix_(rows, S)] - a[rows][:, None] - a[S][None, :] + c
        proj = Kqs @ V[:, :keep] / np.sqrt(lam[:keep])
        out[rows] = np.sqrt(np.maximum(K[rows, rows] - 2 * a[rows] + c - (proj ** 2).sum(1), 0.0))
    return out


def token_novelty(T: np.ndarray, ref: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """(n, tokens): cosine distance of every image token to the nearest allowed reference token at its position."""
    X = T.astype(np.float32)
    X = X - X[ref].mean(0, keepdims=True)
    X /= np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9
    nov = np.empty(X.shape[:2], np.float32)
    for t in range(X.shape[1]):
        dist = 1 - X[:, t] @ X[ref, t].T
        dist[~allowed] = np.inf
        nov[:, t] = dist.min(1)
    return nov


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a positive scores lower than a negative."""
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float(1 - (rankdata(score)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------------- clustering (numpy / scipy only)
def principal(K: np.ndarray, m: int) -> np.ndarray:
    a = K.mean(1)
    lam, V = np.linalg.eigh(K - a[:, None] - a[None, :] + K.mean())
    lam, V = lam[::-1][:m], V[:, ::-1][:, :m]
    return V * np.sqrt(np.maximum(lam, 0))


def kmeans(X: np.ndarray, k: int, seed: int = 0, restarts: int = 10) -> np.ndarray:
    rng, best = np.random.default_rng(seed), None
    for _ in range(restarts):
        C = [X[rng.integers(len(X))]]
        for _ in range(k - 1):  # k-means++
            d = np.min([((X - c) ** 2).sum(1) for c in C], axis=0)
            C.append(X[rng.choice(len(X), p=d / d.sum())])
        C = np.array(C)
        for _ in range(100):
            lab = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
            new = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(new, C):
                break
            C = new
        cost = ((X - C[lab]) ** 2).sum()
        if best is None or cost < best[0]:
            best = (cost, lab)
    return best[1]


def gmm(X: np.ndarray, k: int, init: np.ndarray, iters: int = 200) -> np.ndarray:
    """Full-covariance Gaussian mixture by EM, started from `init` labels."""
    n, d = X.shape
    reg = 1e-3 * X.var(0).mean() * np.eye(d)
    resp = np.eye(k)[init]
    for _ in range(iters):
        nk = resp.sum(0) + 1e-9
        mu = resp.T @ X / nk[:, None]
        logp = np.empty((n, k))
        for j in range(k):
            diff = X - mu[j]
            cov = (resp[:, j, None] * diff).T @ diff / nk[j] + reg
            _, logdet = np.linalg.slogdet(cov)
            logp[:, j] = np.log(nk[j] / n) - 0.5 * (logdet + np.einsum("ij,ij->i", diff @ np.linalg.inv(cov), diff))
        new = np.exp(logp - logp.max(1, keepdims=True))
        new /= new.sum(1, keepdims=True)
        if np.abs(new - resp).max() < 1e-6:
            break
        resp = new
    return resp.argmax(1)


def spectral(X: np.ndarray, k: int, nn: int = 10) -> np.ndarray:
    d2 = ((X[:, None] - X[None]) ** 2).sum(2)
    near = np.argsort(d2, axis=1)[:, 1:nn + 1]
    sigma2 = np.median(np.take_along_axis(d2, near, 1))
    W = np.zeros_like(d2)
    rows = np.repeat(np.arange(len(X)), nn)
    W[rows, near.ravel()] = np.exp(-d2[rows, near.ravel()] / (2 * sigma2))
    W = np.maximum(W, W.T)
    dinv = 1 / np.sqrt(W.sum(1) + 1e-12)
    _, vec = np.linalg.eigh(np.eye(len(X)) - dinv[:, None] * W * dinv[None, :])
    U = vec[:, :k]
    return kmeans(U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12), k)


# ----------------------------------------------------------------------------- following the score
def descend(score: np.ndarray, xy: np.ndarray, step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy descent of `score` over poses at `xy`: from a pose, the pose within 1.5 of its lattice `step` that
    scores lowest, taken while it scores lower. Returns each pose's next pose (-1 where none is lower) and the pose
    its descent ends at."""
    dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    nxt = np.full(len(score), -1)
    for i in range(len(score)):
        near = np.nonzero((dist[i] <= 1.5 * step[i] + 1e-6) & (np.arange(len(score)) != i))[0]
        if len(near) and score[near].min() < score[i]:
            nxt[i] = near[np.argmin(score[near])]
    end = np.arange(len(score))
    for i in range(len(score)):
        while nxt[end[i]] >= 0:  # the score falls at every step, so this ends
            end[i] = nxt[end[i]]
    return nxt, end


# ----------------------------------------------------------------------------- pieces of the result page
def thumbnail(path: Path, width: int) -> str:
    im = Image.open(path).convert("RGB")
    im = im.resize((width, round(width * im.height / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=72, method=4)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


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


def listing(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


CLUSTERERS = {
    "k-means": lambda X: kmeans(X, 2),
    "Gaussian mixture": lambda X: gmm(X, 2, kmeans(X, 2)),
    "Ward linkage": lambda X: fcluster(linkage(X, "ward"), 2, "maxclust") - 1,
    "spectral": lambda X: spectral(X, 2),
}


def main() -> None:
    args = parse_args()
    renders, prep = Path(args.renders), Path(args.prep)
    out = Path(args.out) if args.out else renders / "figures"
    out.mkdir(parents=True, exist_ok=True)
    style()
    plan = json.loads((prep / "plan.json").read_text())
    index = json.loads((prep / "train_index.json").read_text())
    room = np.load(prep / "room_map.npz") if (prep / "room_map.npz").exists() else None
    lat = json.loads((renders / "latents.json").read_text())
    meta = json.loads((renders / "meta.json").read_text())
    L = renders / "latents"
    obs, n = lat["obs"], lat["n"]
    by_id = {q["id"]: q for q in plan["poses"]}
    pose = [by_id[o["id"]] if o["group"] == "grid" else None for o in obs]
    kind = np.array([o.get("kind", "train") for o in obs])
    episode = np.array([o.get("episode", -1) for o in obs])
    is_ref = kind == "train"
    ref = np.nonzero(is_ref)[0]
    get = lambda key, fill=np.nan: np.array([p[key] if p else fill for p in pose], dtype=float)
    fx, fy, dyaw, pose_dist, cell = get("fx"), get("fy"), get("dyaw"), get("pose_dist"), get("cell")
    in_range = np.array([bool(p["in_range"]) if p else False for p in pose])
    anchor = kind == "anchor"
    ev = ~is_ref & ~anchor  # the poses a signal is judged on
    lattice = np.isin(kind, ("fine", "medium", "coarse"))
    far = ev & (pose_dist > args.band)
    bands = [("in the training range", ev & in_range), (f"up to {args.band * 100:.0f} cm outside", ev & ~in_range & ~far),
             (f"{args.band * 100:.0f}-30 cm outside", far & (pose_dist <= 0.3)),
             ("0.3-1 m outside", ev & (pose_dist > 0.3) & (pose_dist <= 1.0)), ("more than 1 m outside", ev & (pose_dist > 1.0))]
    # references a row may be compared with: another episode's (training starts), not the grid scene's (renders)
    ref_ep = episode[ref]
    allowed_out = np.where(is_ref[:, None], ref_ep[None, :] != episode[:, None], ref_ep[None, :] != plan["episode"])
    allowed_in = np.where(is_ref[:, None], ref_ep[None, :] != episode[:, None], True)
    massive = np.load(L / "massive.npy")
    is_img = np.array(lat["is_img"])
    gh, gw = lat["img_grid"]
    Tp, N = lat["action_tokens"], lat["n_tokens"]
    attn = np.load(L / "attn.npy")  # (n, steps, blocks, Tp + N + 1)
    w_img = attn[:, :, :, Tp:Tp + N][..., is_img].mean((1, 2))
    w_img /= w_img.sum(1, keepdims=True)

    def evaluate(s: np.ndarray) -> dict:
        tau = float(np.percentile(s[is_ref], args.quantile))
        med = float(np.median(s[is_ref]))
        acc = s <= tau
        near = ev & (pose_dist <= 0.6)
        tpr, far_acc = float(acc[ev & in_range].mean()), float(acc[far].mean())
        flat = float(np.ptp(s[ev])) < 1e-12  # the same latent for every observation (a text token at layer 0)
        beyond = ev & (pose_dist > 1.0)
        turned = np.nonzero(kind == "yaw")[0]
        return dict(auroc=0.5 if flat else auroc(s[ev], in_range[ev]),
                    rho=float("nan") if flat else float(spearmanr(s[ev], pose_dist[ev]).statistic),
                    rho_near=float("nan") if flat else float(spearmanr(s[near], pose_dist[near]).statistic),
                    rho_beyond_1m=float("nan") if flat or beyond.sum() < 3 else
                    float(spearmanr(s[beyond], pose_dist[beyond]).statistic),
                    headings_accepted_deg=sorted(float(dyaw[i]) for i in turned if acc[i]),
                    tau=tau, ref_median=med,
                    accept_in_range=tpr, accept_far=far_acc, balanced_acc=(tpr + 1 - far_acc) / 2,
                    accept_by_band=[float(acc[m].mean()) if m.any() else None for _, m in bands],
                    anchors_accepted=int(acc[anchor].sum()),
                    units=((s - med) / max(tau - med, 1e-12)))  # 0 = the median training start, 1 = the threshold

    # ---- every (latent, layer, score)
    cands: list[dict] = []

    def add(latent: str, layer, score: str, s_out: np.ndarray, s_in: np.ndarray, extra: dict | None = None) -> None:
        e_out, e_in = evaluate(s_out), evaluate(s_in)
        cands.append(dict(latent=latent, layer=layer, score=score, s=s_out, **e_out,
                          scene_kept=dict(auroc=e_in["auroc"], balanced_acc=e_in["balanced_acc"],
                                          accept_in_range=e_in["accept_in_range"], accept_far=e_in["accept_far"]),
                          **(extra or {})))

    pooled = {"mean": "pool_mean_img", "attn": "pool_attn_img", "attn_all": "pool_attn_all", "txt": "pool_mean_txt",
              "last": "pool_last"}
    grams: dict = {}
    for latent, fname in pooled.items():
        P = np.load(L / f"{fname}.npy")
        for l in range(P.shape[1]):
            K = gram(P[:, l].astype(np.float32)[:, ~massive[l]])
            grams[(latent, l)] = K
            add(latent, l, "knn", knn_score(K, ref, allowed_out, args.k), knn_score(K, ref, allowed_in, args.k))
            if latent in ("mean", "attn"):
                add(latent, l, "pca", pca_score(K, ref, allowed_out, args.pca), pca_score(K, ref, allowed_in, args.pca))
    novelty: dict = {}
    for l in lat["spatial_layers"]:
        T = np.load(L / f"tokens_l{l}.npy")[:, :, ~massive[l]]
        K = gram(T.astype(np.float32))
        grams[("spatial", l)] = K
        add("spatial", l, "knn", knn_score(K, ref, allowed_out, args.k), knn_score(K, ref, allowed_in, args.k))
        add("spatial", l, "pca", pca_score(K, ref, allowed_out, args.pca), pca_score(K, ref, allowed_in, args.pca))
        nov_out, nov_in = token_novelty(T, ref, allowed_out), token_novelty(T, ref, allowed_in)
        novelty[l] = nov_out
        add("spatial", l, "token", nov_out.mean(1), nov_in.mean(1))
        add("spatial", l, "tokenw", (nov_out * w_img).sum(1), (nov_in * w_img).sum(1))
        del T
    for latent, fname in (("ctx", "ctx_mean"), ("act", "act")):  # the action expert's own latents
        if not (L / f"{fname}.npy").exists():
            continue
        P = np.load(L / f"{fname}.npy").astype(np.float32)
        for b in (range(P.shape[1]) if latent == "ctx" else ("flat", "mean")):
            K = gram(P[:, b] if latent == "ctx" else (P if b == "flat" else P.mean(1)))
            layer = f"block {b}" if latent == "ctx" else b
            add(latent, layer, "knn", knn_score(K, ref, allowed_out, args.k), knn_score(K, ref, allowed_in, args.k))
            add(latent, layer, "pca", pca_score(K, ref, allowed_out, args.pca), pca_score(K, ref, allowed_in, args.pca))

    def of(latent: str, score: str, layer=None) -> list[dict]:
        return [c for c in cands if c["latent"] == latent and c["score"] == score and (layer is None or c["layer"] == layer)]

    rank = lambda c: (round(c["balanced_acc"], 4), c["auroc"])
    best = {key: max(of(latent, score), key=rank) for key, latent, score, *_ in SIGNALS}
    last_layer = lat["layers"] - 1
    name = {key: label for key, _, _, _, label, _ in SIGNALS}
    short = {key: lab for key, *_, lab in SIGNALS}
    for key, latent, score, *_ in SIGNALS:
        b = best[key]
        print(f"[signal] {name[key]:38s} best layer {b['layer']:>2}: AUROC {b['auroc']:.3f}, accepted in range "
              f"{b['accept_in_range']:.2f}, far {b['accept_far']:.3f}, anchors {b['anchors_accepted']}/{int(anchor.sum())}; "
              f"last layer: AUROC {of(latent, score, last_layer)[0]['auroc']:.3f}, far accepted "
              f"{of(latent, score, last_layer)[0]['accept_far']:.3f}", flush=True)

    # ---- helpers for the figures
    norm_units = TwoSlopeNorm(vmin=-1.0, vcenter=1.0, vmax=3.0)
    lat_idx = np.nonzero(lattice)[0]
    face_idx = np.nonzero(kind == "facing")[0]
    yaw_idx = np.nonzero(kind == "yaw")[0]
    yaw_idx = yaw_idx[np.argsort(dyaw[yaw_idx])]

    def img_of(i: int, w: int = 0) -> np.ndarray:
        o = obs[i]
        if o["group"] == "train":
            im = Image.open(renders / "img" / "00000_recorded.png") if o["episode"] == plan["episode"] and o["frame"] == 0 \
                else None
        else:
            im = Image.open(renders / o["file"])
        im = im.convert("RGB")
        return np.asarray(im.resize((w, int(w * im.height / im.width))) if w else im)

    def score_map(ax, c: dict, zoom: bool, sel: np.ndarray = lat_idx, labels: bool = False):
        draw_scene(ax, plan, index, room, zoom=zoom, labels=labels, starts=False)
        return draw_cells(ax, [pose[i] for i in sel], c["units"][sel], DIV, norm=norm_units)

    def stat_text(c: dict) -> str:
        return (f"AUROC {c['auroc']:.2f}   in range accepted {c['accept_in_range'] * 100:.0f}%\n"
                f"far accepted {c['accept_far'] * 100:.0f}%")

    def units_bar(fig, cax, mappable) -> None:
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend="both")
        cb.set_ticks([-1, 0, 1, 2, 3])
        cb.set_label("score in threshold units: 0 = the median training start, 1 = the threshold "
                     f"({args.quantile:.0f}% of the training starts score below it); blue = accepted, red = rejected",
                     fontsize=8, color=INK2)
        cb.outline.set_visible(False)

    def find(k: str, **want) -> int | None:
        """The observation of kind k whose plan pose is nearest the wanted values."""
        idx = np.nonzero(kind == k)[0]
        if not len(idx):
            return None
        cost = sum(np.abs({"fx": fx, "fy": fy, "dyaw": dyaw, "pose_dist": pose_dist}[key][idx] - v) / (30 if key == "dyaw" else 1)
                   for key, v in want.items())
        return int(idx[np.argmin(cost)])

    train_anchor = int(ref[ref_ep == plan["episode"]][0])
    inr = np.nonzero(ev & in_range & lattice)[0]
    examples = [("training image (episode %d, frame 0)" % plan["episode"], train_anchor),
                ("re-render at the training start", int(np.nonzero(anchor)[0][0])),
                ("in the training range", int(inr[np.argmax(np.hypot(fx[inr], fy[inr]))]) if len(inr) else None),
                ("15 cm to the left", find("fine", fx=0, fy=0.15)), ("30 cm to the left", find("medium", fx=0, fy=0.3)),
                ("30 cm back", find("medium", fx=-0.3, fy=0)), ("turned 20 deg", find("yaw", dyaw=20)),
                ("beside the table", find("coarse", fx=1.2, fy=1.2)),
                ("facing the faucet, across the table", find("facing", fx=3.0, fy=0.0)),
                ("facing the faucet, from the side", find("facing", fx=0.5, fy=1.5))]
    examples = [(t, i) for t, i in examples if i is not None]

    def label_of(i: int) -> str:
        p = pose[i]
        if p is None:
            return ""
        turn = f", turned {p['dyaw']:+.0f} deg" if abs(p["dyaw"]) > 0.5 else ""
        where = "in the training range" if p["in_range"] else f"{p['pose_dist'] * 100:.0f} cm outside the training range"
        return f"{p['fx']:+.2f} m ahead, {p['fy']:+.2f} m left{turn}\n{where}"

    # ======================================================================== 1. setup
    fig = plt.figure(figsize=(17, 9.6))
    gs = fig.add_gridspec(2, 2, left=0.045, right=0.47, top=0.9, bottom=0.235, wspace=0.22, hspace=0.3,
                          height_ratios=[1.35, 1])
    pd_norm = plt.Normalize(0, 1.0)
    for ax, zoom, sel, title in ((fig.add_subplot(gs[0, 0]), False, lat_idx, "training heading: the room"),
                                 (fig.add_subplot(gs[0, 1]), True, lat_idx, "training heading: around the start"),
                                 (fig.add_subplot(gs[1, 0]), False, face_idx, "facing the faucet")):
        if not len(sel):
            ax.set_axis_off()
            continue
        draw_scene(ax, plan, index, room, zoom=zoom, labels=not zoom, starts=zoom)
        pc = draw_cells(ax, [pose[i] for i in sel], np.minimum(pose_dist[sel], 1.0), SEQ, norm=pd_norm)
        ax.set_title(title, loc="left")
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(dyaw[yaw_idx], pose_dist[yaw_idx], "-o", color=C1, ms=3.5, lw=1.6)
    ax.set_xlabel("heading at the start position (deg, + = left)")
    ax.set_ylabel("pose distance (m)")
    ax.set_title("turned at the start", loc="left")
    cb = fig.colorbar(pc, cax=fig.add_axes((0.045, 0.165, 0.2, 0.012)), orientation="horizontal", extend="max")
    cb.set_label("pose distance: how far the faucet, seen from the pelvis, is outside the training range (m)",
                 fontsize=8, color=INK2, loc="left")
    cb.outline.set_visible(False)
    fig.legend(scene_handles(), scene_labels(index), loc="lower left", bbox_to_anchor=(0.038, 0.012), ncol=1, fontsize=8)
    eg = fig.add_gridspec(4, 3, left=0.5, right=0.99, top=0.9, bottom=0.045, wspace=0.04, hspace=0.5)
    for k, (title, i) in enumerate(examples[:12]):
        ax = fig.add_subplot(eg[k // 3, k % 3])
        ax.imshow(img_of(i, 420))
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=8.5, loc="left", pad=3)
        ax.set_xlabel(label_of(i), fontsize=7.5, color=INK2, labelpad=2, loc="left", linespacing=1.3)
    counts = ", ".join(f"{v} {k}" for k, v in plan["counts"].items() if v)
    fig.suptitle(f"The experiment: {plan['n']} base poses in the scene of training episode {plan['episode']} ({counts}), "
                 f"against the {len(ref)} training start frames", x=0.045, y=0.975, ha="left", fontsize=12)
    fig.text(0.045, 0.945, "The robot keeps the posture of the training start and is moved rigidly over the floor. "
             f"{int((ev & in_range).sum())} poses lie in the training range, {int(far.sum())} more than "
             f"{args.band * 100:.0f} cm outside it.", fontsize=8.5, color=INK2, va="top")
    fig.savefig(out / "1_setup.png", dpi=130)
    plt.close(fig)

    # ======================================================================== 2. the action expert's attention
    A_ref = attn[is_ref].mean((0, 1))  # (blocks, Tp + N + 1)
    n_blocks = A_ref.shape[0]
    parts = {"other action tokens": A_ref[:, :Tp].sum(1), "image tokens": A_ref[:, Tp:Tp + N][:, is_img].sum(1),
             "text tokens": A_ref[:, Tp:Tp + N][:, ~is_img].sum(1), "proprio token": A_ref[:, Tp + N]}
    eff = lambda w: float(np.exp(-(w * np.log(w + 1e-12)).sum(-1)).mean())
    eff_mean = eff(w_img[is_ref])
    heads = np.load(L / "attn_heads.npy") if (L / "attn_heads.npy").exists() else None
    sharp = None
    if heads is not None:
        hw = heads[:, :, :, Tp:Tp + N][..., is_img]  # (n, blocks, heads, image tokens)
        hw = hw / hw.sum(-1, keepdims=True)
        eff_heads = np.exp(-(hw[is_ref] * np.log(hw[is_ref] + 1e-12)).sum(-1)).mean(0)  # (blocks, heads)
        sharp = np.unravel_index(int(eff_heads.argmin()), eff_heads.shape)
    rows_ = [e for e in examples if e[0] in ("training image (episode %d, frame 0)" % plan["episode"],
                                             "re-render at the training start", "30 cm to the left", "turned 20 deg",
                                             "beside the table", "facing the faucet, across the table")]
    cols = ["image", "all blocks"] + [f"block {b}" for b in range(n_blocks)] + ([f"block {sharp[0]}, head {sharp[1]}"] if sharp else [])
    H = 1.55 * len(rows_) + 4.1  # figure height (in)
    fig = plt.figure(figsize=(17, H))
    gs = fig.add_gridspec(len(rows_), len(cols), left=0.115, right=0.98, top=1 - 1.15 / H, bottom=2.85 / H, wspace=0.05,
                          hspace=0.14, width_ratios=[1.75] + [1] * (len(cols) - 1))
    for r, (title, i) in enumerate(rows_):
        w_blocks = attn[i].mean(0)[:, Tp:Tp + N][:, is_img]  # (blocks, image tokens)
        maps = [w_blocks.mean(0)] + list(w_blocks) + ([heads[i][sharp][Tp:Tp + N][is_img]] if sharp else [])
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(img_of(i, 320))
        ax.set_ylabel("\n".join(_wrap(title, 20)), fontsize=7.5, rotation=0, ha="right", va="center", linespacing=1.3)
        ax.set_xticks([]), ax.set_yticks([])
        if r == 0:
            ax.set_title(cols[0], fontsize=8.5)
        for c, m in enumerate(maps):
            ax = fig.add_subplot(gs[r, c + 1])
            m = m / m.sum()
            one_head = bool(sharp) and c == len(maps) - 1
            im = ax.imshow(m.reshape(gh, gw) * len(m), cmap=SEQ, vmin=0, vmax=12 if one_head else 3,
                           interpolation="nearest", aspect="auto")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(cols[c + 1], fontsize=8.5)
            ax.set_xlabel(f"{np.exp(-(m * np.log(m + 1e-12)).sum()):.0f} of {len(m)} tokens", fontsize=7, color=INK2,
                          labelpad=1.5)
            if not one_head:
                shared = im
    cb = fig.colorbar(shared, cax=fig.add_axes((0.665, 2.05 / H, 0.2, 0.011)), orientation="horizontal", extend="max")
    cb.set_label("attention on a token, relative to uniform\n(the single head on the right: 0 to 12)", fontsize=8,
                 color=INK2, loc="left")
    cb.outline.set_visible(False)
    ax = fig.add_axes((0.115, 0.4 / H, 0.5, 1.4 / H))
    left = np.zeros(n_blocks)
    for (pname, v), col in zip(parts.items(), (MUTED, C1, C2, C3)):
        ax.barh(np.arange(n_blocks), v, left=left, color=col, height=0.62, label=pname, edgecolor=SURFACE, lw=1.5)
        for b in range(n_blocks):
            if v[b] > 0.06:
                ax.text(left[b] + v[b] / 2, b, f"{v[b] * 100:.0f}%", ha="center", va="center", fontsize=7.5,
                        color=SURFACE if col != C3 else INK)
        left += v
    ax.set_yticks(np.arange(n_blocks), [f"block {b}" for b in range(n_blocks)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(False)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=8)
    ax.set_title("where the action tokens' attention goes (mean over the training starts)", loc="left", fontsize=9, pad=24)
    note = (f"Averaged over heads, blocks and steps the attention is nearly flat over the image: it is spread like a "
            f"uniform weighting over {eff_mean:.0f} of the {int(is_img.sum())} image tokens, so the attention-weighted "
            "mean of the image tokens is almost their plain mean.")
    if sharp:
        note += (f" Per head: half of the {eff_heads.size} heads spread over more than {np.median(eff_heads):.0f} tokens; "
                 f"the sharpest (block {sharp[0]}, head {sharp[1]}) over {eff_heads.min():.0f}, on the same corner "
                 "tokens whatever the image shows.")
    fig.text(0.665, 1.35 / H, "\n".join(_wrap(note, 74)), fontsize=8.5, color=INK2, va="top", linespacing=1.45)
    fig.suptitle("Where the action expert looks: attention of the 30 action tokens on the image tokens\n"
                 f"Each cell is one image token ({gh} x {gw}), averaged over the 10 denoising steps. Under each map: "
                 "the number of tokens the attention is effectively spread over.", x=0.02, y=1 - 0.22 / H, ha="left",
                 va="top", fontsize=10.5, linespacing=1.45)
    fig.savefig(out / "2_action_attention.png", dpi=130)
    plt.close(fig)

    # ======================================================================== 3. score maps around the start
    layer_cols = lambda key, latent, score: [of(latent, score, lat["spatial_layers"][0] if latent == "spatial" else 0)[0],
                                             best[key], of(latent, score, last_layer)[0]]
    fig, axs = plt.subplots(len(SIGNALS), 3, figsize=(13.2, 3.25 * len(SIGNALS) + 1.5), squeeze=False)
    for r, (key, latent, score, _, label, _) in enumerate(SIGNALS):
        for c, cand in enumerate(layer_cols(key, latent, score)):
            ax = axs[r, c]
            pc = score_map(ax, cand, zoom=True)
            head = ("first layer %d" % cand["layer"], "best layer %d" % cand["layer"], "last layer %d" % cand["layer"])[c]
            ax.set_title(f"{label}\n{head}" if c == 0 else head, loc="left", fontsize=9)
            ax.text(0.02, 0.03, stat_text(cand), transform=ax.transAxes, fontsize=7.5, color=INK, va="bottom",
                    linespacing=1.35, bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="none", alpha=0.9), zorder=9)
            if r < len(SIGNALS) - 1:
                ax.set_xlabel("")
            if c:
                ax.set_ylabel("")
    fig.subplots_adjust(left=0.06, right=0.985, top=1 - 0.95 / (3.25 * len(SIGNALS) + 1.5),
                        bottom=1.05 / (3.25 * len(SIGNALS) + 1.5), wspace=0.16, hspace=0.26)
    units_bar(fig, fig.add_axes((0.2, 0.42 / (3.25 * len(SIGNALS) + 1.5), 0.6, 0.011)), pc)
    fig.suptitle("Top-down, around the training start, the robot at the training heading: is a base pose scored as "
                 "in-distribution?\nBlack box = the training range. Far = more than "
                 f"{args.band * 100:.0f} cm outside it. Layer 0 = the vision tower's output, layer {last_layer} = what the "
                 "action expert reads.", x=0.06, y=0.995, ha="left", fontsize=10.5, linespacing=1.4)
    fig.savefig(out / "3_score_maps.png", dpi=130)
    plt.close(fig)

    # ======================================================================== 4. the room, and facing the faucet
    fig, axs = plt.subplots(2, len(SIGNALS), figsize=(4.2 * len(SIGNALS), 9.6), squeeze=False)
    for c, (key, _, _, _, label, _) in enumerate(SIGNALS):
        cand = best[key]
        pc = score_map(axs[0, c], cand, zoom=False, labels=True)
        axs[0, c].set_title(f"{label}\nlayer {cand['layer']}: training heading", loc="left", fontsize=9)
        ax = axs[1, c]
        if len(face_idx):
            score_map(ax, cand, zoom=False, sel=face_idx)
            a = np.radians(dyaw[face_idx])
            ax.quiver(fy[face_idx], fx[face_idx], np.sin(a), np.cos(a), color=INK, angles="xy", scale_units="xy",
                      scale=5.5, width=0.005, headwidth=4, zorder=5, pivot="tail")
            acc = cand["s"][face_idx] <= cand["tau"]
            ax.set_title(f"facing the faucet: {int(acc.sum())} of {len(face_idx)} accepted", loc="left", fontsize=9)
        else:
            ax.set_axis_off()
        for ax in axs[:, c]:
            if c:
                ax.set_ylabel("")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.11, wspace=0.15, hspace=0.22)
    units_bar(fig, fig.add_axes((0.2, 0.045, 0.6, 0.011)), pc)
    fig.suptitle("The same scores over the whole floor around the table (top), and with the robot turned to face the "
                 "faucet (bottom; arrows = heading)", x=0.05, y=0.975, ha="left", fontsize=11)
    fig.savefig(out / "4_room_maps.png", dpi=130)
    plt.close(fig)

    # ======================================================================== 5. by layer
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
    metrics = (("auroc", "AUROC: in the training range against every other pose", (0.5, 1.005)),
               ("accept_far", f"share of the far poses (> {args.band * 100:.0f} cm outside) accepted", (-0.02, 1.02)),
               ("accept_in_range", "share of the poses in the training range accepted", (-0.02, 1.02)))
    for ax, (mkey, title, ylim) in zip(axs, metrics):
        ends = []
        for key, latent, score, col, label, _ in SIGNALS:
            cs = sorted(of(latent, score), key=lambda c: c["layer"])
            xs, ys = [c["layer"] for c in cs], [c[mkey] for c in cs]
            ax.plot(xs, ys, "-o", color=col, lw=1.6 if key == "attn" else 2.0, ms=5 if latent == "spatial" else 3,
                    label=label, zorder=4 if key == "attn" else 3)
            ends.append((ys[-1], key, col))
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.8, last_layer + 0.8)
        ax.set_xlabel("VLM layer (0 = vision tower output, %d = read by the action expert)" % last_layer)
        ax.set_title(title, loc="left")
        ys_ = _spread([e[0] for e in ends], 0.052 * (ylim[1] - ylim[0]), *ylim)
        for (y, key, col), yl in zip(ends, ys_):
            ax.plot([last_layer + 0.25, last_layer + 1.0], [y, yl], color=AXIS, lw=0.6, clip_on=False)
            ax.plot([last_layer + 1.0, last_layer + 1.7], [yl, yl], color=col, lw=2.5, solid_capstyle="round", clip_on=False)
            ax.text(last_layer + 2.0, yl, short[key], fontsize=7.5, color=INK2, va="center", clip_on=False)
    handles, labels_ = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower left", bbox_to_anchor=(0.035, 0.0), ncol=5, fontsize=8.5)
    fig.suptitle("Which layer carries the signal? The mean and the attention-weighted mean nearly coincide (the "
                 "attention is nearly flat); the latents that keep every token are computed at layers "
                 + ", ".join(str(v) for v in lat["spatial_layers"]), x=0.035, y=0.985, ha="left", fontsize=11)
    fig.subplots_adjust(left=0.035, right=0.935, top=0.86, bottom=0.2, wspace=0.36)
    fig.savefig(out / "5_layers.png", dpi=130)
    plt.close(fig)

    # ======================================================================== 6. score against pose distance, heading
    fig, axs = plt.subplots(2, len(SIGNALS), figsize=(4.2 * len(SIGNALS), 7.6), squeeze=False)
    groups = (("training heading", lattice, C1, "o"), ("turned at the start", kind == "yaw", C3, "D"),
              ("facing the faucet", kind == "facing", C2, "s"))
    rng = np.random.default_rng(0)
    for c, (key, _, _, _, label, _) in enumerate(SIGNALS):
        cand, ax = best[key], axs[0, c]
        u = cand["units"]
        top = float(np.clip(np.percentile(u[ev], 99) * 1.1, 4, 60))
        ax.axhspan(-3, 1, color=FLOOR_FILL, lw=0, zorder=0.5)
        ax.axhline(1, color=INK2, lw=0.8, zorder=1)
        ax.scatter(rng.uniform(-0.085, -0.025, len(ref)), u[ref], s=7, color=INK, lw=0, alpha=0.55, zorder=2,
                   label="training starts (each against the other episodes)")
        for gname, m, col, mk in groups:
            if m.any():
                ax.scatter(pose_dist[m], np.minimum(u[m], top), s=13, color=col, marker=mk, lw=0.4, ec=SURFACE,
                           zorder=3, label=gname)
        ax.set_xscale("symlog", linthresh=0.1, linscale=0.7)
        ax.set_xlim(-0.1, max(4.0, float(np.nanmax(pose_dist)) * 1.1))
        ax.set_xticks([0, 0.1, 0.3, 1, 3], ["0", "0.1", "0.3", "1", "3"])
        ax.set_ylim(min(-1.5, float(u[ev | is_ref].min()) - 0.2), top * 1.03)
        ax.set_title(f"{label}\nlayer {cand['layer']}", loc="left", fontsize=9)
        ax.set_xlabel("pose distance: m outside the training range")
        ax.text(0.985, 1.0, "threshold", transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, va="bottom", ha="right")
        ax = axs[1, c]
        ax.axhspan(-3, 1, color=FLOOR_FILL, lw=0, zorder=0.5)
        ax.axhline(1, color=INK2, lw=0.8, zorder=1)
        if len(yaw_idx):
            a0 = float(np.median(u[anchor]))
            xs = np.r_[dyaw[yaw_idx], 0.0]
            o = np.argsort(xs)
            ax.plot(xs[o], np.r_[u[yaw_idx], a0][o], "-o", color=C3, ms=3.5, lw=1.6)
            ax.set_xlim(-185, 185)
            ax.set_xticks([-180, -90, -45, 0, 45, 90, 180])
            lim = float(np.r_[u[yaw_idx], a0].max())
            ax.set_ylim(min(-1.5, float(np.r_[u[yaw_idx], a0].min()) - 0.2), lim * 1.05)
            tr = np.degrees(plan["train_range"]["dyaw"]) - np.degrees(plan["anchor"]["faucet_rel"][2])
            ax.axvspan(-tr[1], -tr[0], color=INK, alpha=0.12, lw=0, zorder=0.6)
        ax.set_xlabel("heading at the start position (deg, + = left)")
        ax.set_title("turned at the start", loc="left", fontsize=9)
        for ax in axs[:, c]:
            if c == 0:
                ax.set_ylabel("score in threshold units\n(0 = median training start, 1 = threshold)")
    handles, labels_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower left", bbox_to_anchor=(0.045, 0.0), ncol=4, fontsize=8.5)
    fig.suptitle("Is the score a basin around the training range? Top: each dot is one base pose; in the shaded band "
                 "a pose is accepted as in-distribution.\nBottom: the robot turned on the spot at the training start "
                 "(dark band: the headings of the training range).", x=0.045, y=0.99, ha="left", va="top", fontsize=11,
                 linespacing=1.4)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.86, bottom=0.115, wspace=0.2, hspace=0.42)
    fig.savefig(out / "6_basin.png", dpi=130)
    plt.close(fig)

    # ======================================================================== 7. two clusters
    cluster_rows = [("mean of the image tokens", ("mean", best["mean"]["layer"])),
                    ("all image tokens", ("spatial", best["pca"]["layer"]))]
    clusters: dict = {}
    familiar: dict = {}  # per clustering, which observations fall into the familiar cluster
    fig, axs = plt.subplots(len(cluster_rows), len(CLUSTERERS), figsize=(4.2 * len(CLUSTERERS), 3.5 * len(cluster_rows) + 1.3),
                            squeeze=False)
    two = matplotlib.colors.ListedColormap([C2, C1])
    for r, (rname, gkey) in enumerate(cluster_rows):
        Z = principal(grams[gkey], args.cluster_pcs)
        for c, (cname, fn) in enumerate(CLUSTERERS.items()):
            lab = np.asarray(fn(Z))
            share = [float((lab[ref] == j).mean()) for j in (0, 1)]
            fam = int(np.argmax(share))
            acc = lab == fam
            res = dict(training_starts_in_familiar=share[fam], familiar_size=int(acc[ev].sum()),
                       accept_in_range=float(acc[ev & in_range].mean()), accept_far=float(acc[far].mean()),
                       anchors_in_familiar=int(acc[anchor].sum()),
                       max_pose_dist_in_familiar=float(pose_dist[ev & acc].max()) if (ev & acc).any() else None)
            res["balanced_acc"] = (res["accept_in_range"] + 1 - res["accept_far"]) / 2
            clusters[f"{gkey[0]}@{gkey[1]}:{cname}"] = res
            familiar[f"{gkey[0]}@{gkey[1]}:{cname}"] = acc
            ax = axs[r, c]
            draw_scene(ax, plan, index, room, zoom=False, labels=False, starts=False)
            draw_cells(ax, [pose[i] for i in lat_idx], acc[lat_idx].astype(float), two, vmin=0, vmax=1)
            ax.set_title((f"{rname}, layer {gkey[1]}: " if c == 0 else "") + cname, loc="left", fontsize=9, pad=30)
            ax.text(0.0, 1.02, f"familiar cluster: {share[fam] * 100:.0f}% of the training starts,\n"
                    f"{res['accept_in_range'] * 100:.0f}% of the in-range poses, {res['accept_far'] * 100:.0f}% of the far ones",
                    transform=ax.transAxes, fontsize=7.5, va="bottom", linespacing=1.3, color=INK2)
            if c:
                ax.set_ylabel("")
            if r < len(cluster_rows) - 1:
                ax.set_xlabel("")
    fig.legend([Rectangle((0, 0), 1, 1, color=C1), Rectangle((0, 0), 1, 1, color=C2)],
               ["familiar: the cluster that holds most training starts", "the other cluster"], loc="lower left",
               bbox_to_anchor=(0.05, 0.0), ncol=2, fontsize=8.5)
    fig.suptitle(f"Do the latents fall into two clusters, familiar and not? Clustered without labels into 2 (first "
                 f"{args.cluster_pcs} principal components, all {n} observations); the training heading poses shown",
                 x=0.05, y=0.985, ha="left", fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 1.1 / (3.5 * len(cluster_rows) + 1.3),
                        bottom=0.85 / (3.5 * len(cluster_rows) + 1.3), wspace=0.15, hspace=0.36)
    fig.savefig(out / "7_clusters.png", dpi=130)
    plt.close(fig)

    # ======================================================================== 8. which image tokens are unfamiliar
    tl = best["token"]["layer"]
    nov = novelty[tl]
    tok_med, tok_tau = np.median(nov[ref], axis=0), np.percentile(nov[ref], args.quantile, axis=0)
    nov_units = (nov - tok_med) / np.maximum(tok_tau - tok_med, 1e-9)  # per position, like the score's units
    fig, axs = plt.subplots(2, len(examples), figsize=(1.95 * len(examples), 3.75), squeeze=False,
                            gridspec_kw=dict(height_ratios=[1, 0.78]))
    for c, (title, i) in enumerate(examples):
        axs[0, c].imshow(img_of(i, 320))
        axs[0, c].set_title("\n".join(_wrap(title, 24)), fontsize=7.5, loc="left", pad=3)
        hm = axs[1, c].imshow(nov_units[i].reshape(gh, gw), cmap=DIV, norm=norm_units, interpolation="nearest",
                              aspect="auto")
        u = best["token"]["units"][i]
        axs[1, c].set_xlabel(f"score {u:+.1f}: {'accepted' if u <= 1 else 'rejected'}\n"
                             f"{int((nov_units[i] > 1).sum())} of {nov.shape[1]} tokens unfamiliar", fontsize=7.5,
                             color=INK2, labelpad=2, linespacing=1.3)
        for ax in axs[:, c]:
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
    cb = fig.colorbar(hm, cax=fig.add_axes((0.3, 0.135, 0.4, 0.02)), orientation="horizontal", extend="both")
    cb.set_ticks([-1, 0, 1, 2, 3])
    cb.set_label(f"token novelty at layer {tl}: distance to the nearest training-start token at the same position, in "
                 f"that position's threshold units\n(0 = the median training start, 1 = the {args.quantile:.0f}th "
                 "percentile of the training starts; red = unfamiliar)", fontsize=7.5, color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Which part of the image is unfamiliar? One cell per image token; the score is the mean distance over "
                 "the image, in threshold units", x=0.01, y=0.99, ha="left", fontsize=10.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.83, bottom=0.31, wspace=0.05, hspace=0.05)
    fig.savefig(out / "8_token_novelty.png", dpi=130)
    plt.close(fig)

    # ======================================================================== numbers
    strip = lambda c: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ("s", "units")}
    order = sorted(cands, key=lambda c: (-round(c["balanced_acc"], 4), -c["auroc"]))
    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["latent", "layer", "score", "auroc", "balanced_acc", "accept_in_range", "accept_far", "spearman",
                    "spearman_within_0.6m", "anchors_accepted", *[f"accepted: {b}" for b, _ in bands],
                    "auroc_scene_kept", "balanced_acc_scene_kept"])
        for c in order:
            w.writerow([c["latent"], c["layer"], c["score"], *(round(c[k], 4) for k in ("auroc", "balanced_acc",
                        "accept_in_range", "accept_far", "rho", "rho_near")), c["anchors_accepted"],
                        *("" if v is None else round(v, 3) for v in c["accept_by_band"]),
                        round(c["scene_kept"]["auroc"], 4), round(c["scene_kept"]["balanced_acc"], 4)])
    gaps = meta["anchor"].get("recorded_gap_repeats") or []
    summary = dict(
        task=index["task"], run_dir=lat["run_dir"], ckpt_step=lat["ckpt_step"], instruction=lat["instruction"],
        grid_episode=plan["episode"], n_training_starts=int(len(ref)), n_poses=plan["n"], counts=plan["counts"],
        n_in_range=int((ev & in_range).sum()), n_far=int(far.sum()), band_m=args.band,
        bands={b: int(m.sum()) for b, m in bands}, train_range=plan["train_range"],
        render_gap_255=dict(first=meta["anchor"].get("recorded_gap_first"), repeats=gaps),
        tokens=dict(vlm=N, image=int(is_img.sum()), grid=[gh, gw], layers=lat["layers"], massive_channels_last_layer=
                    lat["massive_channels"][str(last_layer)]),
        settings=dict(k=args.k, pca=args.pca, quantile=args.quantile, cluster_pcs=args.cluster_pcs),
        attention=dict(share={k: np.round(v, 3).tolist() for k, v in parts.items()}, image_tokens=int(is_img.sum()),
                       effective_tokens_mean_over_heads=round(eff_mean, 1),
                       effective_tokens_per_head=None if sharp is None else dict(
                           median=round(float(np.median(eff_heads)), 1), min=round(float(eff_heads.min()), 1),
                           sharpest=[int(v) for v in sharp])),
        signals={key: dict(label=name[key], best=strip(best[key]),
                           last_layer=strip(of(latent, score, last_layer)[0]),
                           by_layer={str(c["layer"]): dict(auroc=round(c["auroc"], 4), accept_far=round(c["accept_far"], 4),
                                                           accept_in_range=round(c["accept_in_range"], 4))
                                     for c in sorted(of(latent, score), key=lambda c: c["layer"])})
                 for key, latent, score, *_ in SIGNALS},
        top=[strip(c) for c in order[:15]],
        action_expert=[strip(c) for c in cands if c["latent"] in ("ctx", "act")],
        clusters={k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                  for k, v in clusters.items()},
        facing={key: dict(accepted=int((best[key]["s"][face_idx] <= best[key]["tau"]).sum()), n=int(len(face_idx)))
                for key in best},
        latent_by_score={f"{la}:{sc}": strip(max(of(la, sc), key=rank))
                         for la in ("mean", "attn", "spatial", "ctx", "act") for sc in ("knn", "pca", "token") if of(la, sc)})

    # ======================================================================== can the score be followed?
    hero, pooled = best["token"], best["mean"]
    accepted = lambda c: c["s"] <= c["tau"]
    counts_of = lambda a: (int(a[ev & in_range].sum()), int(a[far].sum()))  # accepted in range, accepted far
    n_in, n_far, n_img = int((ev & in_range).sum()), int(far.sum()), int(is_img.sum())
    a0 = int(np.nonzero(anchor)[0][0])  # the first render at the training start
    nodes = lat_idx if (np.hypot(fx[lat_idx], fy[lat_idx]) < 1e-6).any() else np.r_[a0, lat_idx]
    node_step = np.where(cell[nodes] > 0, cell[nodes], cell[lat_idx].min())
    nxt, end = descend(hero["s"][nodes], np.stack([fx[nodes], fy[nodes]], axis=1), node_step)
    back = in_range[nodes][end]
    reach = []
    for lo, hi, label in ((0.0, 0.1, "≤0.1"), (0.1, 0.3, "0.1–0.3"), (0.3, 0.6, "0.3–0.6"), (0.6, 1.0, "0.6–1"),
                          (1.0, 2.0, "1–2"), (2.0, np.inf, ">2")):  # m outside the training range
        m = (pose_dist[nodes] > lo) & (pose_dist[nodes] <= hi)
        reach.append(dict(label=label, lo=lo, hi=None if np.isinf(hi) else hi, n=int(m.sum()), k=int(back[m].sum())))
    radius, everywhere = 0.0, True  # how far outside the training range (nearly) every start pose still gets back
    for r in reach:
        if r["n"] and r["k"] < 0.95 * r["n"]:
            everywhere = False
            break
        if r["n"]:
            radius = r["hi"] if r["hi"] is not None else float(pose_dist[nodes].max())
    summary["reach"] = dict(signal=f"spatial, layer {hero['layer']}, token", radius_m=radius, by_start_distance=reach)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # ======================================================================== the result page
    (h_in, h_far), (p_in, p_far), (a_in, a_far) = (counts_of(accepted(best[k])) for k in ("token", "mean", "attn"))
    band_cm = f"{args.band * 100:.0f} cm"
    grid_rows = np.nonzero(~is_ref)[0]
    k_of = {int(i): k for k, i in enumerate(grid_rows)}  # observation row -> index among the page's poses
    node_of = {int(i): k for k, i in enumerate(nodes)}
    mkey = f"mean@{best['mean']['layer']}"
    km, ward = familiar[f"{mkey}:k-means"], familiar[f"{mkey}:Ward linkage"]
    cap = lambda t: t[0].upper() + t[1:]

    def where(p: dict) -> str:
        parts = [f"{abs(p[k]):.2f} m {pos if p[k] > 0 else neg}"
                 for k, pos, neg in (("fx", "ahead", "back"), ("fy", "left", "right")) if abs(p[k]) > 0.004]
        if abs(p["dyaw"]) > 0.5:
            parts.append(f"turned {abs(p['dyaw']):.0f}° {'left' if p['dyaw'] > 0 else 'right'}")
        return cap(", ".join(parts)) if parts else "At the training start"

    def outside(p: dict) -> str:
        d = p["pose_dist"]
        return ("in the training range" if p["in_range"] else
                f"{d * 100:.0f} cm outside the training range" if d < 1 else f"{d:.1f} m outside the training range")

    page_poses = []
    for i in grid_rows:
        q, n = pose[i], node_of.get(int(i))
        page_poses.append(dict(
            x=round(q["fy"], 3), y=round(q["fx"], 3), yaw=round(q["dyaw"], 1),
            c=float(q["cell"] or (node_step[n] if n is not None else 0.0)),
            far=bool(far[i]), where=where(q), dist=outside(q), unf=int((nov_units[i] > 1).sum()),
            a=dict(hero=bool(accepted(hero)[i]), pooled=bool(accepted(pooled)[i])),
            cl=dict(km=bool(km[i]), ward=bool(ward[i])),
            reach=None if n is None else bool(back[n]),
            nx=-1 if n is None or nxt[n] < 0 else k_of[int(nodes[nxt[n]])],
            end=None if n is None else k_of[int(nodes[end[n]])]))

    fx_lo, fx_hi, fy_lo, fy_hi = train_box(plan)
    tb = plan["anchor"]["table"]
    half, ct, st = np.asarray(tb["size"][:2]) / 2, math.cos(tb["yaw"]), math.sin(tb["yaw"])
    table = heading_xy(plan, [[tb["position"][0] + ct * a * half[0] - st * b * half[1],
                               tb["position"][1] + st * a * half[0] + ct * b * half[1]]
                              for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    pts = np.r_[np.stack([fy[grid_rows], fx[grid_rows]], axis=1), table]
    views = dict(zoom=dict(l=0.95, r=-0.95, b=-0.75, t=0.85),
                 room=dict(l=round(float(pts[:, 0].max()) + 0.35, 3), r=round(float(pts[:, 0].min()) - 0.35, 3),
                           b=round(float(pts[:, 1].min()) - 0.35, 3), t=round(float(pts[:, 1].max()) + 0.35, 3)))

    shot = lambda title, i: dict(title=title, i=k_of[int(i)], a=bool(accepted(hero)[i]))
    shots = [dict(title="Training frame", i=-1, a=bool(accepted(hero)[train_anchor])), shot("Same pose, rendered again", a0)]
    shots += [shot(where(pose[i]), i) for i in ([int(inr[np.argmax(np.hypot(fx[inr], fy[inr]))])] if len(inr) else [])
              + [find("fine", fx=0, fy=0.15), find("medium", fx=-0.3, fy=0), find("yaw", dyaw=20)] if i is not None]
    looks = [dict(title="At the training start", i=k_of[a0], w=np.round(w_img[a0] * n_img, 2).tolist())]
    looks += [dict(title=where(pose[i]), i=k_of[i], w=np.round(w_img[i] * n_img, 2).tolist())  # attention / an even share
              for i in (find("medium", fx=0, fy=0.3), find("coarse", fx=1.2, fy=1.2)) if i is not None]

    plain = {"mean": "Patches averaged", "attn": "Averaged with action attention", "meanpca": "Averaged, PCA subspace",
             "pca": "Kept in place, PCA subspace", "token": "Kept in place, patch by patch"}
    methods = []
    for key, *_ in SIGNALS:
        k_in, k_far = counts_of(accepted(best[key]))
        methods.append(dict(label=plain[key], layer=best[key]["layer"], inr=[k_in, n_in], far=[k_far, n_far],
                            hero=key == "token", sub=f"VLM layer {best[key]['layer']}", verb="accepted"))
    crows = [dict(label=cname, far=[int(familiar[f"{mkey}:{cname}"][far].sum()), n_far],
                  inr=[int(familiar[f"{mkey}:{cname}"][ev & in_range].sum()), n_in], hero=False,
                  sub=f"averaged patches, VLM layer {best['mean']['layer']}", verb="in the familiar cluster")
             for cname in CLUSTERERS]
    by_layer = lambda la, sc: [dict(l=int(c["layer"]), far=round(c["accept_far"], 4), inr=round(c["accept_in_range"], 4))
                               for c in sorted(of(la, sc), key=lambda c: c["layer"])]
    hero_layers, runs, run = by_layer("spatial", "token"), [], []
    for r in hero_layers:  # the stretch of layers with no far pose accepted and none in range rejected
        if r["far"] == 0 and r["inr"] == 1:
            run.append(r["l"])
        elif run:
            runs, run = runs + [run], []
    runs += [run] if run else []
    works = max(runs, key=lambda q: (hero["layer"] in q, len(q))) if runs else None
    works = [works[0], works[-1]] if works else None
    head = [dict(deg=round(float(dyaw[i]), 1), u=round(float(hero["units"][i]), 3), a=bool(accepted(hero)[i]), i=k_of[int(i)])
            for i in yaw_idx]
    if not any(r["deg"] == 0 for r in head):
        head.append(dict(deg=0.0, u=round(float(hero["units"][a0]), 3), a=bool(accepted(hero)[a0]), i=k_of[a0]))
    head.sort(key=lambda r: r["deg"])

    data = dict(
        meta=dict(n_patches=n_img, far_text=f"more than {band_cm} outside"),
        map=dict(views=views, walls=wall_rects(room, plan, views["room"]) if room is not None else [],
                 table=np.round(table, 3).tolist(), faucet=np.round(heading_xy(plan, plan["anchor"]["faucet"][:2]), 3).tolist(),
                 box=dict(l=round(fy_hi, 3), r=round(fy_lo, 3), b=round(fx_lo, 3), t=round(fx_hi, 3))),
        poses=page_poses,
        sets=dict(lattice=[k_of[int(i)] for i in lat_idx], facing=[k_of[int(i)] for i in face_idx],
                  nodes=[k_of[int(i)] for i in nodes]),
        thumbs=[thumbnail(renders / obs[i]["file"], args.thumb) for i in grid_rows],
        train_thumb=thumbnail(renders / "img" / "00000_recorded.png", args.thumb),
        examples=shots, attention=dict(examples=looks), methods=methods, clusters=crows,
        layers=dict(last=last_layer, hero=hero_layers, pooled=by_layer("mean", "knn"), works=works, works_label="no mistakes",
                    labels=dict(hero="patches kept in place", pooled="patches averaged")),
        reach=reach, heading=head)

    # the sentences: chosen from the numbers, so that another scene or checkpoint gets its own
    clean = h_in == n_in and h_far == 0
    worse_last = of("spatial", "token", last_layer)[0]["accept_far"] - hero["accept_far"] > 0.2
    flat = eff_mean >= 0.9 * n_img
    split = [r["far"][0] <= 0.05 * n_far and r["inr"][0] == n_in for r in crows]
    ae = [max(of(la, sc), key=rank)["accept_far"] for la in ("ctx", "act") for sc in ("knn", "pca") if of(la, sc)]
    deg = lambda d: f"{d:+.0f}°" if d else "0°"
    ok = [deg(r["deg"]) for r in head if r["a"]]
    tr = plan["train_range"]
    yaw_lo, yaw_hi = np.degrees(tr["dyaw"])
    kinds = dict(anchor="re-renders of the start", fine="on a {:.0f} cm grid", medium="on a {:.0f} cm grid",
                 coarse="on a {:.0f} cm grid", yaw="headings at the start", facing="turned to face the faucet")
    steps = {k: 100 * float(np.median(cell[kind == k])) for k in ("fine", "medium", "coarse") if (kind == k).any()}
    report = Path(args.report) if args.report else out / "index.html"
    rel = lambda q: os.path.relpath(q, report.parent)
    text = dict(
        eyebrow="Open faucet · VLA in-distribution signal",
        lede=f"{plan['n']} base poses around the table in one training scene. Each camera view is compared with the "
             f"first frame of the {len(ref)} training episodes.",
        answer=("Yes, close up. Comparing the image patch by patch with the training starts accepts every pose in the "
                "training range and none far from it." if clean else
                f"Partly. Comparing the image patch by patch with the training starts accepts {h_in} of {n_in} poses in "
                f"the training range and {h_far} of {n_far} far ones."),
        tile_in=f"{h_in} / {n_in}", tile_far=f"{h_far} / {n_far}",
        tile_far_label=f"poses more than {band_cm} outside accepted",
        tile_reach=f"{radius:g} m" if radius else "none",
        tile_reach_label=("reach: following the score leads back from this far" if radius else
                          "reach: following the score does not lead back"),
        recipe_1=f"Take the {n_img} image patches at VLM layer {hero['layer']}.",
        h_signal=("The signal is on only where the robot was trained" if clean else
                  f"The signal accepts {h_in} of {n_in} poses in the training range and {h_far} of {n_far} far ones"),
        t_facing=f"Turned to face the faucet: {int(accepted(hero)[face_idx].sum())} of {len(face_idx)} accepted",
        h_pool=("Averaging the patches lets far poses in" if p_far > max(2 * h_far, h_far + 0.02 * n_far) else
                "Averaging the patches does about as well"),
        t_pool_map=f"Patches averaged: {p_far} far poses accepted",
        t_bars=f"Far poses accepted, of {n_far}",
        n_bars=("All five accept every pose in the training range." if all(m_["inr"][0] == n_in for m_ in methods) else
                "Where a signal rejects poses in the training range, its row says so."),
        h_attn=(("Action attention does not help: it is spread over the whole image" if a_far >= 0.7 * p_far else
                 "Action attention is spread over the whole image, yet weighting by it helps") if flat else
                f"Action attention concentrates on about {eff_mean:.0f} of the {n_img} patches"),
        attn_eff=f"{eff_mean:.0f}", attn_n=n_img,
        n_attn=f"Weighting the average by it: {a_far} far poses accepted, {p_far} without.",
        h_layers=((f"Layers {works[0]} to {works[1]} work." if works[0] != works[1] else f"Layer {works[0]} works.")
                  + " The last layer, the one the action expert reads, does not" if works and worse_last else
                  "The last layer, the one the action expert reads, is the worst" if worse_last else
                  "The signal holds up to the last layer"),
        t_layers=f"Far poses accepted, by VLM layer (share of {n_far})",
        h_reach=("Following the score does not lead back, even from close by" if not radius else
                 f"Following the score leads back from everywhere tested, up to {radius:.1f} m" if everywhere else
                 f"It guides only the last {radius:g} m"),
        t_heading=("Turned on the spot: never accepted" if not ok else
                   "Turned on the spot: accepted only at " + listing(ok) if len(ok) <= 4 else
                   f"Turned on the spot: accepted from {ok[0]} to {ok[-1]}"),
        h_clusters=("Two clusters separate the training range on their own" if all(split) else
                    "Clustering finds the two groups only with some algorithms" if any(split) else
                    "Clustering alone does not find the two groups"),
        t_km=f"k-means: {crows[0]['far'][0]} far poses in the familiar cluster",
        t_ward=f"Ward linkage: {int(ward[far].sum())} far poses in the familiar cluster",
        t_cbars=f"Far poses in the familiar cluster, of {n_far}",
        n_clusters="Two clusters of the averaged patches, no labels. Familiar: the cluster that holds the training starts.",
        also=([f"The action expert's own latents: {min(ae) * 100:.0f} to {max(ae) * 100:.0f}% of far poses accepted"
               + (", no signal." if min(ae) > 0.2 else ".")] if ae else [])
             + [f"The same pose rendered again: accepted {hero['anchors_accepted']} of {int(anchor.sum())} times."],
        limits=[f"One scene, the one of training episode {plan['episode']}.", f"{n_in} poses in the training range.",
                "The robot keeps its start posture and is moved rigidly.",
                "Layer and threshold are picked on these same poses.", "Not yet compared with task success."],
        defs=[f"Training range: base poses that see the faucet where a training start had it, {tr['dx'][0] * 100:.0f} to "
              f"{tr['dx'][1] * 100:.0f} cm ahead of the pelvis, {abs(tr['dy'][0]) * 100:.0f} cm right to "
              f"{tr['dy'][1] * 100:.0f} cm left, turned {yaw_lo:+.1f}° to {yaw_hi:+.1f}°.",
              f"Far: more than {band_cm} outside the training range ({n_far} poses). The "
              f"{int((ev & ~in_range & ~far).sum())} poses nearer than that count for neither side.",
              f"Accepted: score at most the {args.quantile:.0f}th percentile of the training starts, each scored "
              "against the other episodes. The grid scene's own episode is left out of the references.",
              "Score, where a chart shows it: 0 is the median training start, 1 the acceptance limit.",
              "Patches that share the attention: as many as an even weighting of the same entropy covers; attention "
              "of the action tokens on the image, averaged over heads, blocks and denoising steps.",
              f"{plan['n']} poses: " + ", ".join(
                  f"{v} " + kinds[k].format(steps.get(k, 0)) for k, v in plan["counts"].items() if v and k in kinds) + ".",
              f"Policy {Path(lat['run_dir']).name}, step {lat['ckpt_step']}, instruction \"{lat['instruction']}\"."]
             + ([f"A render differs from the training image by {min(gaps):.1f} to {max(gaps):.1f} of 255."] if gaps else []),
        links=[("detail figures", rel(out) + "/"), ("summary.json", rel(out / "summary.json")),
               ("candidates.csv", rel(out / "candidates.csv"))])
    indist_report.write(report, text, data)
    if args.fragment:
        indist_report.write(Path(args.fragment), text, data, fragment=True)
    stale = out / "index.html"  # the page of an earlier version of this script
    if stale.exists() and stale.resolve() != report.resolve() and "<title>In-distribution signal</title>" in stale.read_text():
        stale.unlink()
    print(f"result page: {report} ({report.stat().st_size / 1e6:.1f} MB)")
    print(f"detail figures ({len(list(out.glob('*.png')))}), summary.json and candidates.csv: {out}")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width)


def _spread(ys: list[float], gap: float, lo: float = -np.inf, hi: float = np.inf) -> np.ndarray:
    """Label positions at least `gap` apart, order kept, as near their values as that allows, inside [lo, hi]."""
    order = np.argsort(ys, kind="stable")
    pos = np.array(ys, dtype=float)[order]
    for _ in range(200):
        moved = False
        for k in range(len(pos) - 1):
            lack = gap - (pos[k + 1] - pos[k])
            if lack > 1e-9:
                pos[k] -= lack / 2
                pos[k + 1] += lack / 2
                moved = True
        if not moved:
            break
    pos += min(0.0, hi - pos.max()) if len(pos) else 0.0
    pos += max(0.0, lo - pos.min()) if len(pos) else 0.0
    outp = np.empty(len(pos))
    outp[order] = pos
    return outp


if __name__ == "__main__":
    main()
