"""Numerical core of the LLMD fit: float64 statistics and distances (numpy / scipy only, no I/O).

Everything the detectors need is here so it can be tested on synthetic data (``python llmd_stats.py`` runs the
self-test):
  cov_from_moments   mean and sample covariance from (n, sum, sum of outer products); a left-out episode is removed
                     by subtracting its own moments (exact, no refit)
  corr_factor        Cholesky factor of the correlation matrix + lam * I: the Mahalanobis distance in coordinates
                     where every channel has unit variance, so lam means the same for every channel
  cov_factor         Cholesky factor of the covariance + lam * I in the features' own units (the paper's literal
                     recipe)
  maha               squared Mahalanobis distances from a factor
  lw_fit / lw_score  Ledoit-Wolf (2004) shrinkage covariance of a few rows (fewer rows than channels), and the
                     distance to it, computed in the rows' span so no D x D inverse is formed
  conformal_tau      the split-conformal threshold: the ceil((n + 1)(1 - alpha))-th smallest calibration score
  auroc              probability that a positive scores lower than a negative
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.stats import rankdata

LAMBDA = 1e-6  # the paper's ridge: "only needed to ensure invertibility"


def cov_from_moments(n: int, s: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean (D,) and sample covariance (D, D) of n rows with sum s and sum of outer products P."""
    assert s.dtype == np.float64 and P.dtype == np.float64
    mu = s / n
    C = (P - n * np.outer(mu, mu)) / (n - 1)
    return mu, C


def corr_factor(C: np.ndarray, lam: float = LAMBDA) -> tuple[np.ndarray, np.ndarray]:
    """(L, sd): L L' = corr(C) + lam * I, sd the channel standard deviations. d2(x) = |L^-1 ((x - mu) / sd)|^2."""
    assert C.dtype == np.float64
    sd = np.sqrt(np.maximum(np.diag(C), 1e-300))
    R = C / sd[:, None] / sd[None, :]
    R[np.diag_indices_from(R)] += lam
    return cholesky(R, lower=True, overwrite_a=True, check_finite=False), sd


def cov_factor(C: np.ndarray, lam: float = LAMBDA, scale: np.ndarray | None = None) -> np.ndarray:
    """L with L L' = diag(scale) C diag(scale) + lam * I (scale = None: C itself)."""
    assert C.dtype == np.float64
    A = C * np.outer(scale, scale) if scale is not None else C.copy()
    A[np.diag_indices_from(A)] += lam
    return cholesky(A, lower=True, overwrite_a=True, check_finite=False)


def maha(L: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Squared distances |L^-1 z|^2 of the rows of Z (m, D)."""
    if len(Z) == 0:
        return np.zeros(0)
    Y = solve_triangular(L, np.asarray(Z, dtype=np.float64).T, lower=True, check_finite=False)
    return np.einsum("ij,ij->j", Y, Y)


def corr_eig(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w, V, sd): eigenvalues (ascending, clipped at 0) and eigenvectors of corr(C). With y = V' ((x - mu) / sd),
    d2(x; lam) = sum_i y_i^2 / (w_i + lam) for any lam."""
    sd = np.sqrt(np.maximum(np.diag(C), 1e-300))
    w, V = eigh(C / sd[:, None] / sd[None, :], check_finite=False)
    return np.maximum(w, 0.0), V, sd


def lw_fit(X0: np.ndarray) -> dict:
    """Ledoit-Wolf shrinkage covariance of the rows of X0 (m, D), m < D allowed.

    The rows are centred and every channel divided by its sample std, S = Z'Z / m is the (rank-deficient) sample
    covariance of that, and the estimate is (1 - a) S + a mu I with mu = tr(S) / D and the shrinkage a of Ledoit &
    Wolf 2004 (the formulas of scikit-learn's ``ledoit_wolf``). Returned: what lw_score needs."""
    X0 = np.asarray(X0, dtype=np.float64)
    m, p = X0.shape
    mean = X0.mean(0)
    sd = np.maximum(X0.std(0, ddof=1), 1e-300)
    Z = (X0 - mean) / sd
    G = Z @ Z.T  # (m, m): everything is computed in the span of the rows
    tr = float(np.trace(G))
    mu = tr / (m * p)
    s2 = float((G ** 2).sum()) / m ** 2  # |S|_F^2
    beta = (float((np.diag(G) ** 2).sum()) / m - s2) / (p * m)
    delta = (s2 - 2.0 * mu * tr / m + p * mu ** 2) / p
    beta = min(beta, delta)
    shrink = 0.0 if beta == 0 else beta / delta
    w, U = np.linalg.eigh(G)
    keep = w > w[-1] * 1e-10  # centring leaves at most m - 1 directions
    V = (Z.T @ U[:, keep]) / np.sqrt(w[keep])  # (D, r) eigenvectors of S
    return dict(mean=mean, sd=sd, V=V, eig=w[keep] / m, shrink=shrink, mu=mu, rank=int(keep.sum()))


def lw_score(fit: dict, X: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of the rows of X to the Ledoit-Wolf estimate of lw_fit."""
    if len(X) == 0:
        return np.zeros(0)
    Z = (np.asarray(X, dtype=np.float64) - fit["mean"]) / fit["sd"]
    floor = fit["shrink"] * fit["mu"]  # the estimate's eigenvalue outside the rows' span
    if floor <= 0:
        raise ValueError("Ledoit-Wolf shrinkage is 0: the estimate is singular")
    proj = Z @ fit["V"]
    inside = 1.0 / ((1.0 - fit["shrink"]) * fit["eig"] + floor)
    return (Z ** 2).sum(1) / floor - (proj ** 2 * (1.0 / floor - inside)).sum(1)


def conformal_tau(cal: np.ndarray, alpha: float = 0.05) -> float:
    """Split-conformal threshold: a new exchangeable score is at most this with probability >= 1 - alpha."""
    cal = np.sort(np.asarray(cal, dtype=np.float64))
    k = math.ceil((len(cal) + 1) * (1 - alpha))
    return float(cal[k - 1]) if k <= len(cal) else float("inf")


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a positive scores lower than a negative (ties count half)."""
    positive = np.asarray(positive, dtype=bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float(1 - (rankdata(score)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95 % Wilson interval of a proportion k / n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


# ------------------------------------------------------------------------------------------------ self-test
def _selftest() -> None:
    rng = np.random.default_rng(0)
    D, lens = 40, [30, 55, 41, 62, 37, 50, 44, 58]
    A = rng.normal(size=(D, D)) * rng.uniform(0.01, 2.0, size=D)  # channels of very different scale
    X = np.concatenate([rng.normal(size=(n, D)) @ A.T + rng.normal(size=D) * 3 + 100.0 for n in lens])
    bounds = np.cumsum([0] + lens)
    n, s, P = len(X), X.sum(0), X.T @ X

    # leave-one-episode-out by subtracting moments = refitting without the episode
    worst = 0.0
    for e in range(len(lens)):
        Xe = X[bounds[e]:bounds[e + 1]]
        mu, C = cov_from_moments(n - len(Xe), s - Xe.sum(0), P - Xe.T @ Xe)
        rest = np.delete(X, np.s_[bounds[e]:bounds[e + 1]], axis=0)
        mu_d, C_d = rest.mean(0), np.cov(rest.T)
        L, sd = corr_factor(C, 1e-6)
        d2 = maha(L, (Xe - mu) / sd)
        sd_d = np.sqrt(np.diag(C_d))
        ref = np.einsum("ij,jk,ik->i", (Xe - mu_d) / sd_d, np.linalg.inv(C_d / np.outer(sd_d, sd_d) + 1e-6 * np.eye(D)),
                        (Xe - mu_d) / sd_d)
        worst = max(worst, float(np.abs(d2 / ref - 1).max()))
    assert worst < 1e-6, worst
    print(f"leave-one-episode-out by moment subtraction vs refit: relative error {worst:.1e}")

    # in-sample identity: mean d2 = (n - 1) / n * tr((R + lam I)^-1 R)
    mu, C = cov_from_moments(n, s, P)
    L, sd = corr_factor(C, 1e-6)
    R = C / np.outer(sd, sd)
    got = float(maha(L, (X - mu) / sd).mean())
    want = (n - 1) / n * float(np.trace(np.linalg.solve(R + 1e-6 * np.eye(D), R)))
    assert abs(got / want - 1) < 1e-9, (got, want)
    print(f"in-sample mean d2 {got:.6f} = (n-1)/n tr((R+lam)^-1 R) {want:.6f}  (D = {D})")

    # the raw-units factor with a channel scale = scaling the data first
    a = rng.uniform(1e-4, 1.0, size=D)
    Lf = cov_factor(C, 1e-6, a)
    ref = np.einsum("ij,jk,ik->i", (X[:20] - mu) * a, np.linalg.inv(np.cov((X * a).T) + 1e-6 * np.eye(D)), (X[:20] - mu) * a)
    assert np.abs(maha(Lf, (X[:20] - mu) * a) / ref - 1).max() < 1e-6
    print("covariance in raw units with a channel scale: ok")

    # eigen form of the distance for any ridge
    w, V, sd2 = corr_eig(C)
    y = ((X[:20] - mu) / sd2) @ V
    for lam in (1e-6, 1e-2):
        L, _ = corr_factor(C, lam)
        assert np.abs((y ** 2 / (w + lam)).sum(1) / maha(L, (X[:20] - mu) / sd) - 1).max() < 1e-8
    print("eigen form of the distance for any ridge: ok")

    # Ledoit-Wolf in the rows' span vs the textbook formulas in D x D, m < D
    m = 15
    X0, Q = X[:m], X[100:130]
    fit = lw_fit(X0)
    Z = (X0 - X0.mean(0)) / X0.std(0, ddof=1)
    S = Z.T @ Z / m
    mu_ = np.trace(S) / D
    delta_ = ((S - mu_ * np.eye(D)) ** 2).sum() / D
    beta_ = sum((((np.outer(z, z) - S) ** 2).sum()) for z in Z) / (D * m * m)  # LW 2004, eq. for b-bar^2
    shrink = min(beta_, delta_) / delta_
    assert abs(fit["shrink"] - shrink) < 1e-10, (fit["shrink"], shrink)
    Sig = (1 - shrink) * S + shrink * mu_ * np.eye(D)
    Zq = (Q - X0.mean(0)) / X0.std(0, ddof=1)
    ref = np.einsum("ij,jk,ik->i", Zq, np.linalg.inv(Sig), Zq)
    assert np.abs(lw_score(fit, Q) / ref - 1).max() < 1e-8
    print(f"Ledoit-Wolf: shrinkage {fit['shrink']:.4f} and distances match the D x D formulas (rank {fit['rank']} of {D})")

    assert conformal_tau(np.arange(1, 100)) == 95.0 and conformal_tau(np.arange(1, 20)) == 19.0
    assert conformal_tau(np.arange(1, 19)) == float("inf")
    assert abs(auroc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([1, 1, 0, 0])) - 1.0) < 1e-12
    print("conformal threshold and AUROC: ok")


if __name__ == "__main__":
    _selftest()
