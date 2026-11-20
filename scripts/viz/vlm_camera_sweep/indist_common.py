"""Shared drawing helpers of the in-distribution experiment (prepare_base_grid.py, plot_indist_signal.py).

Top-down pictures are drawn in the heading frame of the training start pose, the robot looking up the page:
forward x up, left y to the left, the start pose at (0, 0).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D

# the reference palette of plot_sweep.py
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FLOOR_FILL, WALL = "#ecebe6", "#d3d1c7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, fixed order
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#9b2322"])


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "medium",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 4,
        "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
    })


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


def scene_handles() -> list:
    """Legend handles of what draw_scene() draws without a label: the training range and the training starts."""
    from matplotlib.lines import Line2D

    return [Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=1.3),
            Line2D([], [], marker="o", ms=3, color=INK, ls="none", alpha=0.7),
            Line2D([], [], marker="P", ms=7, mfc=SURFACE, mec=INK, mew=1.2, ls="none")]


def scene_labels(index: dict | None = None) -> list[str]:
    n = len(index["episodes"]) if index else "the"
    return ["training range: base poses that see the faucet where a training start had it",
            f"the {n} training starts, relative to the faucet", "faucet"]


def draw_scene(ax, plan: dict, index: dict | None, room, zoom: bool = False, starts: bool = True,
               labels: bool = True) -> None:
    """The floor plan under a top-down map: walls and furniture, the table, the faucet, the training range (see
    scene_handles() for their legend entries)."""
    x0, y0, yaw0 = plan["anchor"]["pelvis"]
    c, s = math.cos(yaw0), math.sin(yaw0)
    ax.set_aspect("equal")
    ax.grid(False)
    if room is not None:
        lo, res, occ = room["lo"], float(room["res"]), room["robot"]
        world_to_plot = Affine2D(np.array([[-s, c, s * x0 - c * y0], [c, s, -(c * x0 + s * y0)], [0, 0, 1.0]]))
        rgba = np.zeros((occ.shape[1], occ.shape[0], 4))
        rgba[occ.T] = (*[int(WALL[i:i + 2], 16) / 255 for i in (1, 3, 5)], 1.0)
        ax.imshow(rgba, origin="lower", interpolation="nearest", zorder=0.5,
                  extent=(lo[0], lo[0] + occ.shape[0] * res, lo[1], lo[1] + occ.shape[1] * res),
                  transform=world_to_plot + ax.transData)
    t = plan["anchor"]["table"]
    half = np.asarray(t["size"][:2]) / 2
    ct, st = math.cos(t["yaw"]), math.sin(t["yaw"])
    corners = np.array([[t["position"][0] + ct * a * half[0] - st * b * half[1],
                         t["position"][1] + st * a * half[0] + ct * b * half[1]]
                        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    ax.add_patch(Polygon(heading_xy(plan, corners), closed=True, fc=FLOOR_FILL, ec=INK2, lw=1.0, zorder=1))
    fx_lo, fx_hi, fy_lo, fy_hi = train_box(plan)
    if starts and index is not None:  # where each training start stood relative to the faucet, put into this scene
        f0 = plan["anchor"]["faucet_rel"]
        rel = np.array([e["faucet_rel"] for e in index["episodes"]])
        ax.scatter(f0[1] - rel[:, 1], f0[0] - rel[:, 0], s=5 if zoom else 2, color=INK, lw=0, zorder=6, alpha=0.7)
    ax.add_patch(Rectangle((fy_lo, fx_lo), fy_hi - fy_lo, fx_hi - fx_lo, fill=False, ec=INK, lw=1.3, zorder=7))
    fa = heading_xy(plan, plan["anchor"]["faucet"][:2])
    ax.plot([fa[0]], [fa[1]], marker="P", ms=8 if zoom else 6, mfc=SURFACE, mec=INK, mew=1.2, zorder=8, ls="none")
    if labels and not zoom:
        tc = heading_xy(plan, t["position"][:2])
        ax.text(tc[0], tc[1], "table", ha="center", va="center", fontsize=8, color=INK2, zorder=2)
    if zoom:
        ax.set_xlim(0.95, -0.95)
        ax.set_ylim(-0.75, 0.85)
    else:
        pts = np.array([[q["fy"], q["fx"]] for q in plan["poses"]])
        table_p = heading_xy(plan, corners)
        lo_, hi_ = np.minimum(pts.min(0), table_p.min(0)) - 0.35, np.maximum(pts.max(0), table_p.max(0)) + 0.35
        ax.set_xlim(hi_[0], lo_[0])
        ax.set_ylim(lo_[1], hi_[1])
    ax.set_xlabel("left of the training start (m)")
    ax.set_ylabel("ahead of the training start (m)")


def draw_cells(ax, poses: list[dict], values, cmap, norm=None, vmin=None, vmax=None, gap: float = 0.0):
    """One square per lattice pose, its side the lattice step (the big cells first, so the fine ones lie on top)."""
    order = sorted(range(len(poses)), key=lambda i: -poses[i]["cell"])
    patches = [Rectangle((poses[i]["fy"] - poses[i]["cell"] / 2 + gap, poses[i]["fx"] - poses[i]["cell"] / 2 + gap),
                         poses[i]["cell"] - 2 * gap, poses[i]["cell"] - 2 * gap) for i in order]
    pc = PatchCollection(patches, cmap=cmap, norm=norm, lw=0, zorder=3)
    pc.set_array(np.asarray(values, dtype=float)[order])
    if norm is None:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    return pc
