"""Top-down map of what stands in an HSSD room (furniture, walls, door and window frames), for the scene sweep.

Runs in SIMPLE's venv with Isaac Sim's bundled USD libraries on PYTHONPATH and LD_LIBRARY_PATH (the runner sets them;
no Isaac process). The room is placed the way SIMPLE's Isaac scene does it (IsaacSimSimulator.__update_scene):
rotateXYZ(90, 0, yaw) about the USD origin, then translated so the centre of the room's own table (the "surface" prim,
which SIMPLE hides and replaces with the task's table) lands on the scene's center_offset, on the floor. The map is
in that frame, i.e. the world frame of every scene the room is recorded with.

Writes an .npz: two occupancy grids, RES m per cell, cell [i, j] at (lo[0] + (i + 0.5) * RES, lo[1] + (j + 0.5) * RES):
  table   anything between TABLE_BAND (m above the floor): where a table top, the faucet and the clutter would be
  robot   anything between ROBOT_BAND: where the robot stands and walks
and, with --png, a picture of it. With --all-rooms DIR it also maps every other configured room whose USD is on disk,
each placed at the middle of its configured placement range (center_offset_limit_*, center_orientation_limit_*), as
DIR/<uid>.npz and DIR/<uid>.png, listed in DIR/index.json ({uid: {npz, offset, yaw}}); rooms whose USD shares --room's
file are left out, and a room DIR already holds a map of (at the same place) is not mapped again.

    python room_map.py --room scene2 --yaw 90 --offset 0.15 0 --out room_map.npz [--png room_map.png] [--all-rooms DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom

RES = 0.02
REACH = 6.0  # m around the table spot mapped of the other rooms (--all-rooms)
TABLE_BAND = (0.45, 1.25)
ROBOT_BAND = (0.05, 1.8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simple-dir", required=True)
    p.add_argument("--room", required=True, help="SIMPLE's HSSD room uid, e.g. scene2")
    p.add_argument("--yaw", type=float, required=True, help="the scene's center_orientation[2] (deg)")
    p.add_argument("--offset", type=float, nargs=2, required=True, help="the scene's center_offset x y (m)")
    p.add_argument("--out", required=True)
    p.add_argument("--png", default="")
    p.add_argument("--all-rooms", default="", help="also map every other room on disk into this folder")
    return p.parse_args()


def clip(poly: list, z: float, keep_above: bool) -> list:
    """The part of a polygon (3D points) above (or below) the plane at height z."""
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        in_a, in_b = (a[2] >= z) == keep_above, (b[2] >= z) == keep_above
        if in_a:
            out.append(a)
        if in_a != in_b:
            out.append(a + (b - a) * (z - a[2]) / (b[2] - a[2]))
    return out


def main() -> None:
    args = parse_args()
    simple = Path(args.simple_dir)
    rooms = {r["uid"]: r for r in yaml.safe_load(open(simple / "src/simple/resources/hssd-scenes/config.yaml"))}
    map_room(simple, rooms[args.room], args.yaw, args.offset, args.out, args.png)
    if not args.all_rooms:
        return
    out = Path(args.all_rooms)
    out.mkdir(parents=True, exist_ok=True)
    usd = lambda r: simple / "data" / r["data_dir"] / r["name"] / f"{r['name']}.usd"
    index = {}
    for uid, r in rooms.items():
        if uid == args.room or r["name"] == rooms[args.room]["name"] or not usd(r).exists():
            continue
        offset = [(u + d) / 2 for u, d in zip(r["center_offset_limit_up"][:2], r["center_offset_limit_down"][:2])]
        yaw = (r["center_orientation_limit_up"][2] + r["center_orientation_limit_down"][2]) / 2
        npz = out / f"{uid}.npz"
        try:
            if npz.exists():  # mapped before, at the same place: kept
                z = np.load(npz)
                done = np.isclose(float(z["yaw"]), yaw) and np.allclose(z["offset"], offset)
            else:
                done = False
            if not done:
                map_room(simple, r, yaw, offset, str(npz), str(out / f"{uid}.png"), reach=REACH)
        except Exception as e:  # a room that cannot be read is left out
            print(f"[room_map] {uid}: skipped ({e})", flush=True)
            continue
        index[uid] = dict(npz=f"{uid}.npz", offset=offset, yaw=yaw, name=r["name"])
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"[room_map] {len(index)} other rooms -> {out / 'index.json'}")


def map_room(simple: Path, room: dict, yaw: float, offset, out: str, png: str, reach: float | None = None) -> None:
    """room_map.py for one room placed at (yaw, offset): the .npz (and the .png) as the module docstring says; with
    `reach`, only what lies within that far (m, along x and y) of center_offset."""
    args = argparse.Namespace(room=room["uid"], yaw=yaw, offset=list(offset), out=out, png=png)
    stage = Usd.Stage.Open(str(simple / "data" / room["data_dir"] / room["name"] / f"{room['name']}.usd"))
    surface = room["surface"]["prim_path"]
    centroid = np.array(UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
                        .ComputeWorldBound(stage.GetPrimAtPath(surface)).ComputeCentroid())

    # USD rotateXYZ(90, 0, yaw) = Rz(yaw) Rx(90) on column vectors, after the room's scale (SIMPLE appends its
    # rotateXYZ op after the file's identity translate/orient and its scale op); the translation puts the rotated
    # surface centroid at center_offset, with z = 0
    rx = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    c, s = np.cos(np.radians(args.yaw)), np.sin(np.radians(args.yaw))
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rx * float(room.get("scale", 1))
    shift = np.array([*args.offset, 0.0]) - rot @ centroid
    shift[2] = 0.0

    triangles = []
    xforms = UsdGeom.XformCache()
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World"), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if (not prim.IsA(UsdGeom.Mesh) or path.startswith(surface)
                or not path.startswith(("/World/furniture", "/World/walls", "/World/openings"))
                or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible):
            continue
        mesh = UsdGeom.Mesh(prim)
        points, counts = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
        if not points or not counts:
            continue
        local_to_world = np.array(xforms.GetLocalToWorldTransform(prim)).T  # column-vector convention
        p = np.asarray(points, dtype=np.float64)
        p = (np.c_[p, np.ones(len(p))] @ local_to_world.T)[:, :3] @ rot.T + shift
        index, start, faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), 0, []
        for n in counts:  # fan-triangulate every face
            f = index[start:start + n]
            faces += [(f[0], f[k], f[k + 1]) for k in range(1, n - 1)]
            start += n
        tri = p[np.asarray(faces)]
        if reach is not None:
            tri = tri[(np.abs(tri[:, :, :2] - np.asarray(args.offset)) <= reach).all(2).any(1)]
        if len(tri):
            triangles.append(tri)
    triangles = np.concatenate(triangles)

    lo = triangles[:, :, :2].reshape(-1, 2).min(0) - 0.5
    shape = np.ceil((triangles[:, :, :2].reshape(-1, 2).max(0) + 0.5 - lo) / RES).astype(int)
    grids = {}
    for band, (z0, z1) in (("table", TABLE_BAND), ("robot", ROBOT_BAND)):
        img = Image.new("1", tuple(shape), 0)
        draw = ImageDraw.Draw(img)
        zmin, zmax = triangles[:, :, 2].min(1), triangles[:, :, 2].max(1)
        for tri in triangles[(zmax >= z0) & (zmin <= z1)]:
            poly = clip(list(tri), z0, True)
            poly = clip(poly, z1, False) if len(poly) >= 3 else []
            if len(poly) >= 3:
                draw.polygon([((q[0] - lo[0]) / RES, (q[1] - lo[1]) / RES) for q in poly], fill=1, outline=1)
        grids[band] = np.array(img, dtype=bool).T  # [ix, iy]
    np.savez_compressed(args.out, lo=lo, res=RES, room=args.room, yaw=args.yaw, offset=np.asarray(args.offset),
                        table_band=TABLE_BAND, robot_band=ROBOT_BAND, **grids)
    print(f"[room_map] {args.room}: {len(triangles)} triangles, {shape[0]} x {shape[1]} cells of {RES} m -> {args.out}")

    if args.png:
        rgb = np.full((shape[1], shape[0], 3), 255, np.uint8)
        rgb[grids["robot"].T] = (170, 170, 170)
        rgb[grids["table"].T] = (40, 40, 40)
        Image.fromarray(rgb[::-1]).save(args.png)


if __name__ == "__main__":
    main()
