"""Correctness test for the scaled-hole FORGE peg-insert variants.

For each clearance variant (plus the stock baseline) this:
  1. Builds the env (verifies the scaled hole *spawns* -- catches any
     non-uniform-scale / collision issues).
  2. Reports the applied in-plane scale and the resulting bore diameter, and
     checks the realized diametral clearance matches the target.
  3. Teleports the peg into the geometric success pose (concentric, fully
     inserted) and verifies ``_get_curr_successes`` returns True and the reward
     is high.
  4. Teleports the peg to a clearly-not-inserted pose and verifies success is
     False and the reward is lower -- i.e. the success/reward logic still
     discriminates correctly with the enlarged bore.

Run (headless, no rendering needed):
    conda run -n isaaclab python scripts/test_scaled_hole.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test scaled-hole FORGE peg-insert variants.")
parser.add_argument("--num_envs", type=int, default=2, help="Parallel envs per variant.")
parser.add_argument("--settle_steps", type=int, default=3, help="Env steps after teleporting the peg.")
parser.add_argument(
    "--idx",
    type=int,
    default=None,
    help="Index into VARIANTS to test (0=baseline..4=5mm). Isaac Lab hangs when a "
    "second env is built in one process, so test exactly ONE variant per process "
    "and drive the sweep from a bash loop of fresh processes.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402  registers Isaac-Forge-* ids

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import envs  # noqa: F401,E402  registers the scaled-hole ids
from envs.forge_scaled_hole_cfg import PEG_DIAMETER_NATIVE, NATIVE_CLEARANCE  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# (gym id, label, target diametral clearance in metres)
VARIANTS = [
    ("Isaac-Forge-PegInsert-Direct-v0", "baseline", NATIVE_CLEARANCE),
    ("Isaac-Forge-PegInsert-Clear0p5-Direct-v0", "0.5 mm", 0.0005),
    ("Isaac-Forge-PegInsert-Clear1p0-Direct-v0", "1.0 mm", 0.001),
    ("Isaac-Forge-PegInsert-Clear2p0-Direct-v0", "2.0 mm", 0.002),
    ("Isaac-Forge-PegInsert-Clear5p0-Direct-v0", "5.0 mm", 0.005),
]


def _teleport_peg(u, target_pos_w, quat_w=None):
    """Place the held peg root at ``target_pos_w`` (world), zero velocity."""
    n = u.num_envs
    if quat_w is None:
        quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device).repeat(n, 1)
    pose = torch.cat([target_pos_w, quat_w], dim=-1)
    u._held_asset.write_root_pose_to_sim(pose)
    u._held_asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=u.device))


def _step(env, u, n_steps):
    zero_action = torch.zeros((u.num_envs, u.cfg.action_space), device=u.device)
    reward = None
    for _ in range(n_steps):
        _, reward, _, _, _ = env.step(zero_action)
    return reward


def run_variant(task_id, label, target_clearance):
    print(f"\n{'='*72}\n[{label}]  {task_id}", flush=True)
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    u = env.unwrapped

    # --- geometry report ---------------------------------------------------
    scale = tuple(u.cfg_task.fixed_asset.spawn.scale) if u.cfg_task.fixed_asset.spawn.scale else (1.0, 1.0, 1.0)
    bore = u.cfg_task.fixed_asset_cfg.diameter
    realized_clearance = bore - PEG_DIAMETER_NATIVE
    print(f"  in-plane scale (sx,sy,sz) = {scale}", flush=True)
    print(f"  bore diameter            = {bore*1000:.4f} mm", flush=True)
    print(f"  peg diameter (constant)  = {PEG_DIAMETER_NATIVE*1000:.4f} mm", flush=True)
    print(f"  target clearance         = {target_clearance*1000:.4f} mm (diametral)", flush=True)
    print(f"  realized clearance       = {realized_clearance*1000:.4f} mm (radial gap {realized_clearance*500:.4f} mm)",
          flush=True)
    clearance_ok = abs(realized_clearance - target_clearance) < 1e-6

    env.reset()

    # --- success pose: peg root coincident with hole root ------------------
    fixed_pos_w = u._fixed_asset.data.root_pos_w.clone()  # world frame
    _teleport_peg(u, fixed_pos_w.clone())
    rew_success = _step(env, u, args.settle_steps)
    succ = u._get_curr_successes(success_threshold=u.cfg_task.success_threshold, check_rot=False)

    # measured alignment at the success pose
    held_base, _ = _held_and_target(u)
    print(f"  [success pose ] successes = {succ.tolist()}  reward = {rew_success.tolist()}", flush=True)

    # --- not-inserted pose: peg lifted 50 mm above the hole ----------------
    up = fixed_pos_w.clone()
    up[:, 2] += 0.05
    _teleport_peg(u, up)
    rew_fail = _step(env, u, args.settle_steps)
    succ_fail = u._get_curr_successes(success_threshold=u.cfg_task.success_threshold, check_rot=False)
    print(f"  [lifted 50 mm ] successes = {succ_fail.tolist()}  reward = {rew_fail.tolist()}", flush=True)

    ok = (
        clearance_ok
        and bool(succ.all())
        and not bool(succ_fail.any())
        and float(rew_success.mean()) > float(rew_fail.mean())
    )
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(clearance_ok={clearance_ok}, success@inserted={bool(succ.all())}, "
          f"success@lifted={bool(succ_fail.any())}, rew_inserted>rew_lifted="
          f"{float(rew_success.mean())>float(rew_fail.mean())})", flush=True)

    env.close()
    return ok


def _held_and_target(u):
    import isaaclab_tasks.direct.factory.factory_utils as fu
    held_base, _ = fu.get_held_base_pose(
        u.held_pos, u.held_quat, u.cfg_task.name, u.cfg_task.fixed_asset_cfg, u.num_envs, u.device
    )
    target, _ = fu.get_target_held_base_pose(
        u.fixed_pos, u.fixed_quat, u.cfg_task.name, u.cfg_task.fixed_asset_cfg, u.num_envs, u.device
    )
    return held_base, target


def main():
    if args.idx is None:
        raise SystemExit(
            "Specify --idx N (0..%d). Run ONE variant per process; drive the sweep "
            "from a bash loop (see scripts/run_scaled_hole_tests.sh)." % (len(VARIANTS) - 1)
        )
    task_id, label, clearance = VARIANTS[args.idx]
    try:
        ok = run_variant(task_id, label, clearance)
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        ok = False
    # Machine-readable line the bash driver greps for.
    print(f"\nVARIANT_RESULT idx={args.idx} label='{label}' result={'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
