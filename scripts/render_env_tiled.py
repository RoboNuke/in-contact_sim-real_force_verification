"""Render the real ForgeEnv at the peg-insert success pose using a TiledCamera.

Mimics the working recorder in ../generalized_hybrid_vic_action_space
(learning/env_setup.py + wrappers/recording.py): the ONLY reliable way to read
pixels from Isaac Sim on this box is a TiledCamera that is spawned INSIDE
``_setup_scene`` *before* ``clone_environments`` (so it is cloned per-env and
initialized together with the env), with ``scene.clone_in_fabric=False`` (a
rendering sensor is not Fabric-replicated). Then ``camera.data.output["rgb"]``
just works. Post-hoc Camera creation / env.render / viewport capture all fail
here -- this pre-clone injection is the difference.

One variant per process. Run:
    conda run -n isaaclab python scripts/render_env_tiled.py --headless --enable_cameras --idx 0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0, help="0=baseline..4=5mm")
parser.add_argument("--out_dir", type=str, default="data_analysis/scaled_hole_figs")
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--hide_robot", action="store_true", default=True)
parser.add_argument("--show_robot", dest="hide_robot", action="store_false")
parser.add_argument("--render_frames", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402
from isaaclab.sim.spawners.sensors import PinholeCameraCfg  # noqa: E402
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv  # noqa: E402  (Forge inherits _setup_scene)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import envs  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

VARIANTS = [
    ("Isaac-Forge-PegInsert-Direct-v0", "baseline", 0.114),
    ("Isaac-Forge-PegInsert-Clear0p5-Direct-v0", "0.5mm", 0.5),
    ("Isaac-Forge-PegInsert-Clear1p0-Direct-v0", "1.0mm", 1.0),
    ("Isaac-Forge-PegInsert-Clear2p0-Direct-v0", "2.0mm", 2.0),
    ("Isaac-Forge-PegInsert-Clear5p0-Direct-v0", "5.0mm", 5.0),
]

CAM_KEY = "viz_camera"
_holder = {}


def _install_camera(env_cls, cam_cfg):
    """Spawn the TiledCamera pre-clone inside _setup_scene (the working pattern)."""
    orig_setup = env_cls._setup_scene

    def patched(self):
        orig_clone = self.scene.clone_environments

        def shim_clone(*a, **k):
            self.scene.clone_environments = orig_clone  # fire once
            print(f"[tiled] spawning TiledCamera at {cam_cfg.prim_path} before clone...", flush=True)
            cam = TiledCamera(cam_cfg)
            ret = orig_clone(*a, **k)
            self.scene._sensors[CAM_KEY] = cam
            _holder["cam"] = cam
            print("[tiled] clone done; camera registered.", flush=True)
            return ret

        self.scene.clone_environments = shim_clone
        return orig_setup(self)

    env_cls._setup_scene = patched


def _save_png(rgb_np, path):
    from PIL import Image
    Image.fromarray(rgb_np[..., :3]).save(path)


def main():
    task_id, tag, clearance = VARIANTS[args.idx]
    print(f"[tiled] variant={tag} task={task_id}", flush=True)

    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
    # rendering sensor needs real per-env prims, not Fabric clones
    env_cfg.scene.clone_in_fabric = False

    cam_cfg = TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/{CAM_KEY}",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.35), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=PinholeCameraCfg(focal_length=28.0, focus_distance=0.12,
                               horizontal_aperture=20.955, clipping_range=(0.02, 20.0)),
        width=args.width,
        height=args.height,
        update_period=0.0,
    )
    _install_camera(FactoryEnv, cam_cfg)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    print("[tiled] env built. resetting...", flush=True)
    env.reset()
    cam = _holder["cam"]

    # hide robot for an unobstructed peg/hole view
    if args.hide_robot:
        from pxr import UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
        print("[tiled] robot hidden.", flush=True)

    # teleport peg to success pose (peg root == hole root)
    u._compute_intermediate_values(dt=u.physics_dt)
    fixed_pos_w = u._fixed_asset.data.root_pos_w.clone()
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=u.device)
    u._held_asset.write_root_pose_to_sim(torch.cat([fixed_pos_w, quat], dim=-1))
    u._held_asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=u.device))
    for _ in range(3):
        u.sim.step(render=False)
    u._compute_intermediate_values(dt=u.physics_dt)
    succ = u._get_curr_successes(success_threshold=u.cfg_task.success_threshold, check_rot=False)
    print(f"[tiled] success={succ.tolist()} held_pos={u.held_pos[0].tolist()} "
          f"fixed_pos={u.fixed_pos[0].tolist()}", flush=True)

    # aim the (now-initialized) camera at the hole, isometric looking-down
    tgt = fixed_pos_w[0].detach().cpu().numpy().copy()
    tgt[2] += 0.012
    d = 0.11
    eye = tgt + np.array([d * 0.75, -d, d * 0.9])
    cam.set_world_poses_from_view(
        torch.tensor(np.array([eye]), dtype=torch.float32, device=u.device),
        torch.tensor(np.array([tgt]), dtype=torch.float32, device=u.device),
    )
    print(f"[tiled] camera aimed eye={eye.round(3).tolist()} tgt={tgt.round(3).tolist()}", flush=True)

    # render + update, then read rgb
    rgb = None
    for i in range(args.render_frames):
        u.sim.step(render=True)
        cam.update(dt=u.physics_dt)
        out = cam.data.output.get("rgb", None)
        ready = out is not None and out.shape[0] > 0 and int(out.abs().sum().item()) > 0
        if i == 0 or (i + 1) % 6 == 0:
            print(f"[tiled]   frame {i+1}/{args.render_frames} rgb_ready={ready} "
                  f"shape={None if out is None else tuple(out.shape)}", flush=True)
        if ready:
            rgb = out[0].detach().cpu().numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    outp = os.path.join(args.out_dir, f"env_success_{tag}.png")
    if rgb is not None:
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        _save_png(rgb, outp)
        print(f"[tiled] SAVED {outp} shape={rgb.shape}", flush=True)
    else:
        print("[tiled] ERROR: no rgb frames captured", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
