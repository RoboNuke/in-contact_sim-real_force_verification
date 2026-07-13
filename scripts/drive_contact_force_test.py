"""Non-learning driver for ``Isaac-ContactForceTest-Direct-v0``.

Boots Isaac Lab, builds the contact-force test env, and applies a fixed action
that pushes the grasped cylinder straight down into the box. Each step it prints
the two force readings (contact sensor vs. joint-torque estimate) and the
cylinder-tip-to-box-top gap so you can confirm the env generates contact force.

The "fixed policies" mentioned in the task will replace the canned action here;
for now this just exercises the env end-to-end.

Example:
    python scripts/drive_contact_force_test.py --num_envs 4 --headless \
        --push_z -1.0 --kp_z 300 --steps 200
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive the contact-force test env with a fixed push action.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs.")
parser.add_argument("--steps", type=int, default=200, help="Number of policy steps to run.")
parser.add_argument(
    "--push_z",
    type=float,
    default=-1.0,
    help="Per-step z-target delta action (unit action; negative = push down). "
    "Scaled by ctrl.z_action_scale and clipped to ±ctrl.z_action_bound in the env.",
)
parser.add_argument("--kp_z", type=float, default=300.0, help="z proportional gain action (N/m).")
parser.add_argument("--log_every", type=int, default=10, help="Print telemetry every N steps.")
parser.add_argument(
    "--log_physics", action="store_true", help="Record per-physics-step data into the env logger."
)
parser.add_argument(
    "--log_out",
    type=str,
    default="physics_log.npz",
    help="Where to save the physics log (.npz) when --log_physics is set.",
)
parser.add_argument(
    "--delta_z",
    action="store_true",
    help="Use delta-from-current z control (constant-force, FORGE/Factory convention) instead of "
    "the default surface-referenced absolute setpoint (dz relative to the box top; settles to equilibrium).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("[drive] booting AppLauncher...", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[drive] AppLauncher booted.", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402  registers Isaac-* ids

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import envs  # noqa: F401,E402  registers Isaac-ContactForceTest-Direct-v0
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

print("[drive] imports done, envs registered.", flush=True)


def main() -> None:
    print("[drive] parsing env cfg...", flush=True)
    env_cfg = parse_env_cfg("Isaac-ContactForceTest-Direct-v0", device=args.device, num_envs=args.num_envs)
    env_cfg.ctrl.z_target_absolute = not args.delta_z
    env_cfg.log_physics = args.log_physics
    print("[drive] building env (gym.make -> scene setup)...", flush=True)
    env = gym.make("Isaac-ContactForceTest-Direct-v0", cfg=env_cfg, render_mode=None)
    print("[drive] env built.", flush=True)
    u = env.unwrapped

    box_size_z = u.cfg_task.fixed_asset_cfg.box_size[2]
    cyl_h = u.cfg_task.held_asset_cfg.height

    print("[drive] resetting env (grasp + place)...", flush=True)
    obs, _ = env.reset()
    # Confirm the initial (pre-push) placement: cylinder tip should sit tip_gap
    # above the box top.
    init_gap_mm = ((u.held_pos[:, 2] - cyl_h / 2.0) - (u.fixed_pos[:, 2] + box_size_z / 2.0)) * 1000.0
    print(
        f"[drive] reset done. initial tip-to-box gap = {init_gap_mm.mean().item():.3f} mm "
        f"(target {u.cfg_task.tip_gap * 1000:.1f} mm). starting push loop.",
        flush=True,
    )

    action = torch.tensor([[args.push_z, args.kp_z]], device=u.device).repeat(args.num_envs, 1)

    print(
        f"[drive] num_envs={args.num_envs} push_z={args.push_z} kp_z={args.kp_z} "
        f"(z_scale={u.cfg.ctrl.z_action_scale}, z_bound={u.cfg.ctrl.z_action_bound}, "
        f"torque_source={u.cfg.ctrl.joint_torque_source})",
        flush=True,
    )

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(action)

        if step % args.log_every == 0 or step == args.steps - 1:
            # Cylinder tip (bottom, center-origin) and box top (world z).
            tip_z = u.held_pos[:, 2] - cyl_h / 2.0
            box_top_z = u.fixed_pos[:, 2] + box_size_z / 2.0
            gap = (tip_z - box_top_z) * 1000.0  # mm

            contact_fz = u.contact_force_ee[:, 2].mean().item()  # EE frame
            meas_fz = u.est_force_ee_meas[:, 2].mean().item()  # measured torque, pinv(J^T)
            dyn_fz = u.est_force_ee_dyn[:, 2].mean().item()  # measured torque, dyn-consistent inv

            print(
                f"[step {step:4d}] gap(mm)={gap.mean().item():+6.2f}  "
                f"contact_Fz={contact_fz:+8.3f}  "
                f"est_Fz_pinv={meas_fz:+8.3f}  est_Fz_dynconsistent={dyn_fz:+8.3f}",
                flush=True,
            )

    if args.log_physics and u.physics_logger is not None:
        u.physics_logger.save(args.log_out)
        print(
            f"[drive] saved {len(u.physics_logger)} physics-step records to {args.log_out} "
            f"(fields: {sorted(u.physics_logger.as_dict().keys())})",
            flush=True,
        )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
