"""Run ONE parameter-sweep condition of the contact-force position push test.

Boots Isaac, builds ``Isaac-ContactForceTest-Direct-v0`` with a single
(solver_position_iteration_count, max_depenetration_velocity, sim.dt,
contact_offset) combination, then for each force target in turn:

  1. resets the env (fresh grasp + tip placed ``tip_gap`` = 0.2 mm above the box),
  2. commands the position controller to a surface-referenced setpoint
     ``dz = -force / gain`` (so the settled contact force ~ gain * dz = force,
     using the DEFAULT z task gain), and
  3. holds it for ``--seconds`` (default 2 s) of sim time while the env's
     physics-rate logger records every substep.

All logged fields for all envs and all force levels are stored together in one
compressed ``.npz`` named
``{iters}_{max_depen}_{dt_denom}_{contact_offset}.npz`` (decimal points -> "-",
sim.dt written as its denominator). The per-condition wall time is included.

This is the worker invoked once per grid point by ``scripts/param_sweep.py``;
it can also be run standalone, e.g.::

    python scripts/param_sweep_run.py --headless \
        --solver_iters 192 --max_depen 5.0 --dt_denom 120 --contact_offset 0.005
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run one contact-force param-sweep condition.")
# --- the four swept parameters (this process = one grid point) ---
parser.add_argument("--solver_iters", type=int, required=True, help="solver_position_iteration_count.")
parser.add_argument("--max_depen", type=float, required=True, help="max_depenetration_velocity (m/s).")
parser.add_argument("--dt_denom", type=int, required=True, help="sim.dt denominator (e.g. 120 => dt=1/120).")
parser.add_argument("--contact_offset", type=float, required=True, help="collision contact_offset (m).")
# --- fixed protocol ---
parser.add_argument("--num_envs", type=int, default=5, help="Parallel envs (replicates).")
parser.add_argument("--seconds", type=float, default=2.0, help="Sim seconds held at each force goal.")
parser.add_argument("--forces", type=str, default="1,2,5,10,15", help="Comma-separated force goals (N).")
parser.add_argument(
    "--gain",
    type=float,
    default=None,
    help="z task gain used to convert force->setpoint (dz=-force/gain) and commanded to the "
    "controller. Default: the cfg default z proportional gain.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "param_sweep"),
    help="Directory for the output .npz.",
)
parser.add_argument("--log_every", type=int, default=0, help="If >0, print force telemetry every N policy steps.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(
    f"[sweep-run] condition: iters={args.solver_iters} max_depen={args.max_depen} "
    f"dt=1/{args.dt_denom} contact_offset={args.contact_offset}",
    flush=True,
)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402  registers Isaac-* ids

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import envs  # noqa: F401,E402  registers Isaac-ContactForceTest-Direct-v0
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-ContactForceTest-Direct-v0"


def _fmt(x) -> str:
    """Format a number for the filename: drop decimals by replacing '.' with '-'."""
    return str(x).replace(".", "-")


def main() -> None:
    dt = 1.0 / args.dt_denom
    forces = [float(f) for f in args.forces.split(",") if f.strip() != ""]

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)

    # ---- Apply the four swept parameters ----
    # 1) sim.dt (config field).
    env_cfg.sim.dt = dt
    # 2) solver_position_iteration_count. Pin the GLOBAL PhysX scene range to
    #    [N, N] so every body in the coupled island (robot + cylinder + box) is
    #    forced to exactly N position iterations -- controlling it per-body would
    #    be overridden by the robot articulation, which shares the contact island.
    env_cfg.sim.physx.min_position_iteration_count = args.solver_iters
    env_cfg.sim.physx.max_position_iteration_count = args.solver_iters
    # 3) max_depenetration_velocity and 4) contact_offset, on both contacting
    #    bodies (held cylinder + fixed box), via the cfg fields the env reads.
    env_cfg.task.held_asset_cfg.max_depenetration_velocity = args.max_depen
    env_cfg.task.fixed_asset_cfg.max_depenetration_velocity = args.max_depen
    env_cfg.task.held_asset_cfg.contact_offset = args.contact_offset
    env_cfg.task.fixed_asset_cfg.contact_offset = args.contact_offset

    # ---- Position controller, surface-referenced absolute setpoint ----
    env_cfg.ctrl.control_mode = "position"
    env_cfg.ctrl.z_target_absolute = True
    env_cfg.log_physics = True

    gain = float(args.gain) if args.gain is not None else float(env_cfg.ctrl.default_task_prop_gains[2])

    print("[sweep-run] building env...", flush=True)
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    decimation = int(env_cfg.decimation)
    n_policy_steps = int(round(args.seconds / (dt * decimation)))
    print(
        f"[sweep-run] built. gain={gain:.1f} decimation={decimation} "
        f"policy_steps/goal={n_policy_steps} (~{n_policy_steps * decimation} substeps).",
        flush=True,
    )

    # Warn if any setpoint would be clipped by the z action bound.
    max_depth = max(forces) / gain
    if max_depth > env_cfg.ctrl.z_action_bound + 1e-9:
        print(
            f"[sweep-run] WARNING: setpoint depth {max_depth * 1000:.1f} mm exceeds "
            f"z_action_bound {env_cfg.ctrl.z_action_bound * 1000:.1f} mm; force targets will be clipped.",
            flush=True,
        )

    per_force = []  # list of {field: (T, N, ...)} dicts, one per force level
    wall_per_force = []

    for f_target in forces:
        # (2) surface-referenced setpoint dz = -force/gain (negative = press down);
        # z_action_scale is 1.0 so the action IS dz. Command the same gain so the
        # settled force ~ gain * depth = f_target.
        dz = -f_target / gain
        action = torch.tensor([[dz, gain]], device=u.device).repeat(args.num_envs, 1)

        # (1) fresh reset (grasp + place) before EACH force level.
        env.reset()
        u.physics_logger.clear()  # discard anything logged during reset settling

        t0 = time.perf_counter()
        for step in range(n_policy_steps):
            env.step(action)
            if args.log_every and (step % args.log_every == 0 or step == n_policy_steps - 1):
                print(
                    f"[sweep-run]   f*={f_target:5.1f}N step {step:4d} "
                    f"contact_Fz={u.contact_force_ee[:, 2].mean().item():+8.3f}",
                    flush=True,
                )
        wall = time.perf_counter() - t0
        wall_per_force.append(wall)

        d = {k: v.numpy() for k, v in u.physics_logger.as_dict().items()}
        per_force.append(d)
        print(
            f"[sweep-run] f*={f_target:5.1f}N done: {d['t'].shape[0]} substeps, "
            f"contact_Fz_final={u.contact_force_ee[:, 2].mean().item():+7.3f} N, wall={wall:.2f}s",
            flush=True,
        )

    env.close()

    # ---- Stack across force levels: (n_forces, T, N, ...) ----
    field_names = [k for k in per_force[0].keys()]
    stacked = {}
    ok = True
    for name in field_names:
        try:
            stacked[name] = np.stack([d[name] for d in per_force], axis=0)
        except ValueError:
            # Timelines differ in length across force levels (should not happen at
            # fixed dt/seconds); fall back to object array so nothing is lost.
            ok = False
            stacked[name] = np.array([d[name] for d in per_force], dtype=object)
    if not ok:
        print("[sweep-run] WARNING: force-level timelines had unequal length; saved as object arrays.", flush=True)

    out = dict(stacked)
    out["force_targets"] = np.asarray(forces, dtype=np.float64)
    out["wall_time_per_force"] = np.asarray(wall_per_force, dtype=np.float64)
    out["wall_time_total"] = np.asarray(float(sum(wall_per_force)), dtype=np.float64)
    # Condition metadata (also encoded in the filename, kept here for convenience).
    out["meta_solver_iters"] = np.asarray(args.solver_iters)
    out["meta_max_depenetration_velocity"] = np.asarray(args.max_depen)
    out["meta_sim_dt"] = np.asarray(dt)
    out["meta_dt_denom"] = np.asarray(args.dt_denom)
    out["meta_contact_offset"] = np.asarray(args.contact_offset)
    out["meta_gain"] = np.asarray(gain)
    out["meta_num_envs"] = np.asarray(args.num_envs)
    out["meta_decimation"] = np.asarray(decimation)
    out["meta_seconds"] = np.asarray(args.seconds)
    out["meta_n_policy_steps"] = np.asarray(n_policy_steps)

    os.makedirs(args.out_dir, exist_ok=True)
    fname = f"{args.solver_iters}_{_fmt(args.max_depen)}_{args.dt_denom}_{_fmt(args.contact_offset)}.npz"
    fpath = os.path.join(args.out_dir, fname)
    np.savez_compressed(fpath, **out)
    print(
        f"[sweep-run] saved {fpath}  (total wall {out['wall_time_total']:.2f}s, "
        f"{len(forces)} force levels x {args.num_envs} envs)",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
