"""Windowed (non-headless) capture of the ForgeEnv at the peg-insert success pose.

Every HEADLESS pixel-readback path fails on this box (RTX 5090 / driver 580 /
Isaac Sim 5.0): TiledCamera init hangs, Camera sensor hangs, env.render()'s
replicator annotator errors, headless viewport capture has no window to grab.
But the other repo's visualize_env.py proves the WINDOWED renderer works here
(DISPLAY is set). So we boot a real window, drive the env to the success pose,
and grab the viewport framebuffer -- which exists only when windowed.

Must run with a display (DISPLAY set). Do NOT pass --headless.
    conda run -n isaaclab python scripts/render_env_windowed.py --idx 0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0, help="0=baseline..4=5mm")
parser.add_argument("--out_dir", type=str, default="data_analysis/scaled_hole_figs")
parser.add_argument("--hide_robot", action="store_true", default=True)
parser.add_argument("--show_robot", dest="hide_robot", action="store_false")
parser.add_argument("--warmup", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# windowed: do not allow headless (no window -> no viewport framebuffer)
args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402

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


def main():
    task_id, tag, clearance = VARIANTS[args.idx]
    print(f"[win] variant={tag} task={task_id}", flush=True)

    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    print("[win] env built. resetting...", flush=True)
    env.reset()

    if args.hide_robot:
        from pxr import UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
        print("[win] robot hidden.", flush=True)

    # teleport peg to the success pose
    u._compute_intermediate_values(dt=u.physics_dt)
    fixed_pos_w = u._fixed_asset.data.root_pos_w.clone()
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=u.device)
    u._held_asset.write_root_pose_to_sim(torch.cat([fixed_pos_w, quat], dim=-1))
    u._held_asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=u.device))
    for _ in range(3):
        u.sim.step(render=False)
    u._compute_intermediate_values(dt=u.physics_dt)
    succ = u._get_curr_successes(success_threshold=u.cfg_task.success_threshold, check_rot=False)
    print(f"[win] success={succ.tolist()} held_pos={u.held_pos[0].tolist()} "
          f"fixed_pos={u.fixed_pos[0].tolist()}", flush=True)

    # aim the persp viewport camera, isometric looking-down at the hole
    tgt = fixed_pos_w[0].detach().cpu().numpy().copy()
    tgt[2] += 0.012
    d = 0.11
    eye = tgt + np.array([d * 0.75, -d, d * 0.9])
    u.sim.set_camera_view(tuple(eye.tolist()), tuple(tgt.tolist()))
    print(f"[win] camera set eye={eye.round(3).tolist()} tgt={tgt.round(3).tolist()}", flush=True)

    # render frames so the viewport has content
    print(f"[win] rendering {args.warmup} warm-up frames...", flush=True)
    for i in range(args.warmup):
        u.sim.render()
        if i == 0 or (i + 1) % 15 == 0:
            print(f"[win]   frame {i+1}/{args.warmup}", flush=True)

    # capture the viewport framebuffer to a file
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(args.out_dir, f"env_success_{tag}.png"))
    if os.path.exists(out):
        os.remove(out)
    import omni.kit.viewport.utility as vp_utils
    vp = vp_utils.get_active_viewport()
    print(f"[win] viewport={vp}; capturing to {out}", flush=True)
    vp_utils.capture_viewport_to_file(vp, out)
    for i in range(200):
        simulation_app.update()
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"[win]   file written after {i+1} updates", flush=True)
            break
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[win] SAVED {out} ({os.path.getsize(out)} bytes)", flush=True)
    else:
        print("[win] ERROR: capture did not write a file", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
