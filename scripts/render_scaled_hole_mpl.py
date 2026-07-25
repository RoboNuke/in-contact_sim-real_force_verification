"""CPU render of the scaled-hole peg-insert variants from the real factory meshes.

The Isaac Sim RTX render path hangs on this GPU/driver (the same reason the
TiledCamera recorder is disabled in the repo), so instead of rendering in-sim we
load the *actual* factory peg/hole STL meshes, apply the exact same in-plane
scale the sim uses to the hole, place the peg in the geometric success pose
(peg root coincident with hole root -> concentric, seated), and rasterize with
matplotlib (no GPU).

Meshes: factory_peg_8mm.stl (D=7.986 mm, H=50 mm, base at z=0) and
factory_hole_8mm.stl (40x40 mm block, bore opening at z=25 mm, root at z=0).
Units are millimetres. Clearance is diametral: bore = 8.1*s, so s = (7.986+c)/8.1.

Run:
    conda run -n general python scripts/render_scaled_hole_mpl.py
"""

import argparse
import os

import numpy as np
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PEG_STL = "/home/hunter/Continuous_Force_RL/real_robot_exps/factory_peg_8mm.stl"
HOLE_STL = "/home/hunter/Continuous_Force_RL/real_robot_exps/factory_hole_8mm.stl"

PEG_D_MM = 7.986
BORE_D_NATIVE_MM = 8.1

# (label, tag, diametral clearance in mm)
VARIANTS = [
    ("baseline\n0.114 mm", "baseline", 0.114),
    ("0.5 mm", "clear0p5", 0.5),
    ("1.0 mm", "clear1p0", 1.0),
    ("2.0 mm", "clear2p0", 2.0),
    ("5.0 mm", "clear5p0", 5.0),
]

PEG_COLOR = "#c76b3a"   # copper peg
HOLE_COLOR = "#9aa4ad"  # steel socket


def scale_for_clearance(c_mm):
    return (PEG_D_MM + c_mm) / BORE_D_NATIVE_MM


def shaded_faces(mesh, base_color, ls, azdeg=315, altdeg=55):
    """Return per-face RGBA shaded by face-normal vs a light source."""
    normals = mesh.face_normals
    rgb = np.array(matplotlib.colors.to_rgb(base_color))
    # simple lambert term from the light direction
    light = np.array([
        np.cos(np.radians(altdeg)) * np.cos(np.radians(azdeg)),
        np.cos(np.radians(altdeg)) * np.sin(np.radians(azdeg)),
        np.sin(np.radians(altdeg)),
    ])
    intensity = 0.45 + 0.55 * np.clip(normals @ light, 0, 1)
    colors = np.clip(rgb[None, :] * intensity[:, None], 0, 1)
    return np.concatenate([colors, np.ones((len(colors), 1))], axis=1)


def add_scene(ax, parts, ls):
    """Draw several meshes as ONE Poly3DCollection so all faces depth-sort
    together. matplotlib 3D does not z-order *between* separate collections, so
    keeping peg + hole in one collection is what makes the peg correctly seat
    inside (and be occluded by) the socket instead of floating in front of it.
    ``parts`` is a list of (mesh, base_color)."""
    tris_list, col_list = [], []
    for mesh, base_color in parts:
        tris_list.append(mesh.vertices[mesh.faces])          # (F,3,3)
        col_list.append(shaded_faces(mesh, base_color, ls))  # (F,4)
    tris = np.concatenate(tris_list, axis=0)
    cols = np.concatenate(col_list, axis=0)
    coll = Poly3DCollection(tris, facecolors=cols, edgecolors="none", linewidths=0)
    coll.set_zsort("average")
    ax.add_collection3d(coll)


def build_meshes(clearance_mm):
    peg = trimesh.load(PEG_STL)
    hole = trimesh.load(HOLE_STL)
    s = scale_for_clearance(clearance_mm)
    # scale the hole in-plane only (X, Y); leave Z (height) unchanged
    S = np.diag([s, s, 1.0, 1.0])
    hole.apply_transform(S)
    return peg, hole, s


def iso_view(ax, peg, hole, ls, elev=32, azim=-55):
    # single merged collection => correct depth sorting between peg and socket
    add_scene(ax, [(hole, HOLE_COLOR), (peg, PEG_COLOR)], ls)
    allpts = np.vstack([peg.vertices, hole.vertices])
    c = allpts.mean(0)
    r = 24
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(-4, 52)
    ax.set_box_aspect((2 * r, 2 * r, 56))
    ax.view_init(elev=elev, azim=azim)  # "looking down" isometric
    ax.set_axis_off()


def topdown_zoom(ax, clearance_mm, s):
    """Crisp top-down schematic of the annular clearance (to scale)."""
    bore_r = BORE_D_NATIVE_MM * s / 2.0
    peg_r = PEG_D_MM / 2.0
    ax.set_aspect("equal")
    # socket outer (scaled) as a rounded square-ish patch -> just show a disk region
    ax.add_patch(plt.Circle((0, 0), bore_r + 1.2, color=HOLE_COLOR, zorder=0))
    ax.add_patch(plt.Circle((0, 0), bore_r, color="white", zorder=1))            # bore
    ax.add_patch(plt.Circle((0, 0), bore_r, fill=False, color="#3a3f44", lw=1.2, zorder=3))
    ax.add_patch(plt.Circle((0, 0), peg_r, color=PEG_COLOR, zorder=2))           # peg
    ax.add_patch(plt.Circle((0, 0), peg_r, fill=False, color="#5a2c15", lw=1.2, zorder=3))
    lim = bore_r + 1.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    gap = bore_r - peg_r
    ax.set_title(f"gap {gap:.3f} mm radial\nbore ⌀{2*bore_r:.3f}", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_analysis/scaled_hole_figs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ls = LightSource(azdeg=315, altdeg=55)

    # ---- per-variant isometric close-ups ---------------------------------
    for label, tag, c in VARIANTS:
        peg, hole, s = build_meshes(c)
        fig = plt.figure(figsize=(5, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        iso_view(ax, peg, hole, ls)
        ax.set_title(f"Peg in hole — clearance {label.replace(chr(10),' ')}\n"
                     f"bore ⌀{BORE_D_NATIVE_MM*s:.3f} mm,  peg ⌀{PEG_D_MM} mm  (scale {s:.4f})",
                     fontsize=10)
        p = os.path.join(args.out_dir, f"iso_{tag}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {p}  (scale={s:.4f}, bore={BORE_D_NATIVE_MM*s:.3f} mm)", flush=True)

    # ---- comparison contact sheet: iso (top row) + top-down (bottom) -----
    fig = plt.figure(figsize=(4 * len(VARIANTS), 8.5))
    for i, (label, tag, c) in enumerate(VARIANTS):
        peg, hole, s = build_meshes(c)
        ax3d = fig.add_subplot(2, len(VARIANTS), i + 1, projection="3d")
        iso_view(ax3d, peg, hole, ls)
        ax3d.set_title(label, fontsize=12, pad=0)

        ax2d = fig.add_subplot(2, len(VARIANTS), len(VARIANTS) + i + 1)
        topdown_zoom(ax2d, c, s)
    fig.suptitle("Scaled-hole FORGE peg-insert — peg in success pose (concentric, seated)\n"
                 "top: isometric looking-down view   bottom: top-down clearance (to scale)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(args.out_dir, "peg_in_hole_comparison.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
