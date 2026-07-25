"""Render the scaled-hole peg-insert variants for visual inspection.

Builds a minimal scene (just the hole + peg -- no robot/table) with one
(hole, peg) pair per clearance laid out in a row. The peg is placed in the
task's geometric success pose (peg root coincident with hole root -- concentric,
fully inserted). A camera captures an isometric, looking-down view of each pair
(close-up) plus one wide comparison shot.

Run:
    conda run -n isaaclab python scripts/render_scaled_hole.py --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render scaled-hole peg-insert variants.")
parser.add_argument("--out_dir", type=str, default="data_analysis/scaled_hole_figs")
parser.add_argument("--width", type=int, default=1200)
parser.add_argument("--height", type=int, default=1200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Cameras must be enabled for the Camera sensor to render.
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from isaaclab_tasks.direct.factory.factory_tasks_cfg import Hole8mm, Peg8mm  # noqa: E402
from envs.forge_scaled_hole_cfg import (  # noqa: E402
    PEG_DIAMETER_NATIVE,
    NATIVE_CLEARANCE,
    scale_for_clearance,
)

# (label, filename-tag, diametral clearance in metres)
VARIANTS = [
    ("baseline (0.114 mm)", "baseline", NATIVE_CLEARANCE),
    ("0.5 mm", "clear0p5", 0.0005),
    ("1.0 mm", "clear1p0", 0.001),
    ("2.0 mm", "clear2p0", 0.002),
    ("5.0 mm", "clear5p0", 0.005),
]

SPACING = 0.08  # metres between hole centres along +X


def _save_png(rgb_np, path):
    try:
        from PIL import Image
        Image.fromarray(rgb_np[..., :3]).save(path)
    except Exception:  # noqa: BLE001
        import imageio.v2 as imageio
        imageio.imwrite(path, rgb_np[..., :3])


def _world_bbox(prim_path):
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    return np.array([mn[0], mn[1], mn[2]]), np.array([mx[0], mx[1], mx[2]])


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device))

    # Lighting only -- keep the scene clean for inspection.
    dome_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    dome_cfg.func("/World/DomeLight", dome_cfg)
    key_cfg = sim_utils.SphereLightCfg(intensity=30000.0, radius=0.1)
    key_cfg.func("/World/KeyLight", key_cfg, translation=(0.1, -0.1, 0.4))

    z0 = 0.1  # arbitrary height; hole & peg roots coincide here (success pose)
    positions = []
    for i, (label, tag, clearance) in enumerate(VARIANTS):
        x = i * SPACING
        positions.append(x)
        s = scale_for_clearance(clearance)

        # Hole: scaled in-plane only.
        hole_cfg = sim_utils.UsdFileCfg(usd_path=Hole8mm.usd_path, scale=(s, s, 1.0))
        hole_cfg.func(f"/World/Hole_{tag}", hole_cfg, translation=(x, 0.0, z0))

        # Peg: native size, root coincident with hole root (success pose).
        peg_cfg = sim_utils.UsdFileCfg(usd_path=Peg8mm.usd_path)
        peg_cfg.func(f"/World/Peg_{tag}", peg_cfg, translation=(x, 0.0, z0))

    # Camera sensor (spawned as its own prim; posed explicitly below).
    cam = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            update_period=0.0,
            width=args.width,
            height=args.height,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=28.0, focus_distance=0.4, clipping_range=(0.01, 10.0)
            ),
        )
    )

    sim.reset()
    # Let a few frames render so replicator populates the buffers.
    for _ in range(8):
        sim.step(render=True)
    cam.update(dt=sim.get_physics_dt())

    # Report actual geometry from USD world bounds (sanity check).
    print("\n=== geometry from USD world bounds ===", flush=True)
    for (label, tag, clearance), x in zip(VARIANTS, positions):
        hmn, hmx = _world_bbox(f"/World/Hole_{tag}")
        pmn, pmx = _world_bbox(f"/World/Peg_{tag}")
        hole_xy = (hmx[:2] - hmn[:2]) * 1000.0
        peg_xy = (pmx[:2] - pmn[:2]) * 1000.0
        print(f"  {label:>18}: scale={scale_for_clearance(clearance):.4f}  "
              f"hole outer XY={hole_xy[0]:.2f}x{hole_xy[1]:.2f} mm  "
              f"peg XY={peg_xy[0]:.2f}x{peg_xy[1]:.2f} mm  "
              f"hole top z={hmx[2]:.4f}  peg top z={pmx[2]:.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    def capture(eye, target, path):
        cam.set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=args.device),
            torch.tensor([target], dtype=torch.float32, device=args.device),
        )
        for k in range(6):
            sim.step(render=True)
            cam.update(dt=sim.get_physics_dt())
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
        _save_png(rgb, path)
        print(f"  saved {path}  shape={rgb.shape}", flush=True)

    # Per-clearance isometric close-ups (looking down from +X/-Y/+Z).
    print("\n=== rendering ===", flush=True)
    for (label, tag, clearance), x in zip(VARIANTS, positions):
        hmn, hmx = _world_bbox(f"/World/Hole_{tag}")
        top = float(hmx[2])
        tgt = [x, 0.0, top - 0.004]
        d = 0.030
        eye = [x + d, -d, top + d * 1.3]
        capture(eye, tgt, os.path.join(args.out_dir, f"peg_in_hole_{tag}.png"))

    # Wide comparison shot of the whole row.
    x_center = float(np.mean(positions))
    span = positions[-1] - positions[0]
    top_all = float(_world_bbox(f"/World/Hole_{VARIANTS[-1][1]}")[1][2])
    d = span * 1.1 + 0.15
    capture([x_center + d * 0.45, -d, top_all + d * 0.9],
            [x_center, 0.0, top_all - 0.01],
            os.path.join(args.out_dir, "peg_in_hole_comparison.png"))

    print("\nDone.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
