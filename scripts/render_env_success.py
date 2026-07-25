"""Render the REAL ForgeEnv with the peg teleported to the success pose.

Unlike a hand-built mesh scene, this uses the actual gym env (same reset/placement
the task uses), moves the held peg to the geometric success pose (peg root
coincident with the fixed-asset/hole root), and captures an isometric
"looking-down" camera image so you can visually confirm the peg is positioned
correctly at success *in the env*.

One variant per process (Isaac Lab deadlocks on a 2nd env in one process).

Run:
    conda run -n isaaclab python scripts/render_env_success.py --headless --enable_cameras --idx 0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0, help="0=baseline..4=5mm")
parser.add_argument("--out_dir", type=str, default="data_analysis/scaled_hole_figs")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=1280)
parser.add_argument("--hide_robot", action="store_true", default=True)
parser.add_argument("--show_robot", dest="hide_robot", action="store_false")
parser.add_argument("--render_frames", type=int, default=48, help="render steps before capture (RTX converge).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

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


def _save_png(rgb_np, path):
    from PIL import Image
    Image.fromarray(rgb_np[..., :3]).save(path)


def main():
    task_id, tag, clearance = VARIANTS[args.idx]
    print(f"[render] variant={tag} task={task_id}", flush=True)

    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
    env_cfg.viewer.resolution = (args.width, args.height)
    # render_mode=None: we capture the viewport framebuffer directly via
    # omni.kit.viewport.utility (a different path than the replicator rgb
    # annotator, which errors on this GPU: "Invalid object in Py_Graph").
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    print("[render] env built. resetting...", flush=True)
    env.reset()

    # --- hide the robot so the peg/hole are unobstructed --------------------
    if args.hide_robot:
        from pxr import UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
        print("[render] robot hidden.", flush=True)

    # --- teleport peg to the success pose -----------------------------------
    u._compute_intermediate_values(dt=u.physics_dt)
    fixed_pos_w = u._fixed_asset.data.root_pos_w.clone()
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=u.device)
    u._held_asset.write_root_pose_to_sim(torch.cat([fixed_pos_w, quat], dim=-1))
    u._held_asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=u.device))
    for _ in range(3):
        u.sim.step(render=False)
    u._compute_intermediate_values(dt=u.physics_dt)
    succ = u._get_curr_successes(success_threshold=u.cfg_task.success_threshold, check_rot=False)
    print(f"[render] teleported to success pose. success={succ.tolist()}  "
          f"held_pos={u.held_pos[0].tolist()}  fixed_pos={u.fixed_pos[0].tolist()}", flush=True)

    # --- aim the viewport (persp) camera: isometric looking-down ------------
    tgt = fixed_pos_w[0].detach().cpu().numpy().copy()
    tgt[2] += 0.012  # aim slightly above the hole opening
    d = 0.11
    eye = tgt + np.array([d * 0.75, -d, d * 0.9])  # +X / -Y / above -> iso looking down
    u.sim.set_camera_view(tuple(eye.tolist()), tuple(tgt.tolist()))
    print(f"[render] camera set. eye={eye.round(3).tolist()} tgt={tgt.round(3).tolist()}", flush=True)

    # --- render frames so the viewport has content, then capture to file ----
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(args.out_dir, f"env_success_{tag}.png"))
    if os.path.exists(out):
        os.remove(out)

    print(f"[render] rendering {args.render_frames} warm-up frames...", flush=True)
    for i in range(args.render_frames):
        u.sim.step(render=True)
        if i == 0 or (i + 1) % 10 == 0:
            print(f"[render]   frame {i+1}/{args.render_frames}", flush=True)

    import ctypes
    import omni.kit.viewport.utility as vp_utils
    vp = vp_utils.get_active_viewport()
    print(f"[render] viewport={vp}. scheduling buffer capture ...", flush=True)

    captured = {}

    def on_capture(buffer, buffer_size, width, height, fmt):
        try:
            data = (ctypes.c_ubyte * buffer_size).from_address(int(buffer))
            arr = np.frombuffer(data, dtype=np.uint8).copy()
            ch = max(1, buffer_size // (width * height))
            captured["rgb"] = arr.reshape(height, width, ch)
            captured["shape"] = (height, width, ch)
        except Exception as e:  # noqa: BLE001
            captured["err"] = repr(e)

    vp_utils.capture_viewport_to_buffer(vp, on_capture)

    # pump app updates until the capture callback fires (schedule is async)
    for i in range(300):
        simulation_app.update()
        if "rgb" in captured or "err" in captured:
            print(f"[render]   capture callback fired after {i+1} updates", flush=True)
            break

    if "rgb" in captured:
        rgb = captured["rgb"]
        print(f"[render] captured buffer shape={captured['shape']}", flush=True)
        _save_png(rgb[..., :3], out)
        print(f"[render] SAVED {out}  ({os.path.getsize(out)} bytes)", flush=True)
    else:
        print(f"[render] ERROR: capture failed err={captured.get('err')}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
