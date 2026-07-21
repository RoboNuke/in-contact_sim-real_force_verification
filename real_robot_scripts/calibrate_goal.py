"""
Calibrate the real-robot peg-in-hole goal, then write it into eval_config.yaml.

The FORGE policy observes fingertip position relative to the hole-entrance obs
frame (`fixed_asset_position`) and success is judged against the seated peg-base
pose (`target_peg_base_position`). Both are physical to your fixture, so they must
be measured once per setup. This script captures them from the robot:

  1. It closes the gripper on the peg.
  2. You hand-guide the arm until the peg is FULLY SEATED in the hole (bottomed
     out), and press Enter.
  3. It reads the fingertip pose and computes, from the seated capture:
        target_peg_base_position = ee_pos + ee_to_peg_base_offset
        fixed_asset_position     = target_peg_base_position + [0, 0, hole_height]
                                   (entrance is hole_height above the seated peg
                                   base, matching sim's fixed_pos_obs_frame)
  4. It writes both back into the config (comments preserved).
  5. It retracts and drives to (goal XY, 5 cm above the entrance) so you can
     eyeball peg/hole alignment before running an eval.

Hand-guiding is enabled physically on the FR3 (guiding button / Franka Desk) —
this script does not command torques while you position the arm.

Usage:
    python real_robot_scripts/calibrate_goal.py --config real_robot_scripts/eval_config.yaml
    python real_robot_scripts/calibrate_goal.py --override robot.use_mock=true --dry
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_robot_scripts.pro_robot_interface import FrankaInterface
from real_robot_scripts.robot_interface import make_ee_target_pose
from real_robot_scripts.peg_insert_eval import load_config


def set_yaml_list(path: str, key: str, values, ndigits: int = 6) -> None:
    """Rewrite a `  key: [ ... ]` line in-place, preserving indent + trailing comment.

    Assumes `key` appears exactly once as a list-valued mapping entry (true for
    fixed_asset_position / target_peg_base_position in eval_config.yaml).
    """
    with open(path) as f:
        text = f.read()
    formatted = "[" + ", ".join(f"{float(v):.{ndigits}f}" for v in values) + "]"
    pattern = re.compile(rf"^(?P<indent>\s*){re.escape(key)}:\s*\[.*?\](?P<comment>\s*#.*)?$", re.M)
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"Key '{key}: [...]' not found in {path}")
    if len(matches) > 1:
        raise RuntimeError(f"Key '{key}' appears {len(matches)} times in {path}; refusing to edit ambiguously")

    def repl(m):
        return f"{m.group('indent')}{key}: {formatted}{m.group('comment') or ''}"

    with open(path, "w") as f:
        f.write(pattern.sub(repl, text))
    print(f"  wrote {key}: {formatted}  ->  {path}")


def main():
    p = argparse.ArgumentParser(description="Calibrate the peg-in-hole goal and write it to config")
    p.add_argument("--config", default="real_robot_scripts/eval_config.yaml")
    p.add_argument("--out", default=None, help="Config file to write (default: --config)")
    p.add_argument("--override", action="append", default=[], help="Config override key=value")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_grip", action="store_true", help="Skip closing the gripper (fixed tool)")
    p.add_argument("--no_write", action="store_true", help="Compute + print only; don't edit the config")
    p.add_argument("--no_verify", action="store_true", help="Skip the post-calibration verify move")
    p.add_argument("--dry", action="store_true", help="Auto-capture without waiting for Enter (mock testing)")
    args = p.parse_args()

    out_path = args.out or args.config
    cfg = load_config(args.config, args.override)
    device = args.device
    task = cfg["task"]
    hole_height = float(task["hole_height"])
    ee_to_peg = torch.as_tensor(task["ee_to_peg_base_offset"], device=device, dtype=torch.float32)

    print("=" * 70)
    print("  GOAL CALIBRATION")
    print("=" * 70)
    print("\nInitializing robot interface...")
    robot = FrankaInterface(cfg, device=device)

    try:
        if not args.no_grip:
            print("Closing gripper on the peg...")
            robot.close_gripper()

        print("\n>>> Hand-guide the arm until the peg is FULLY SEATED in the hole.")
        print("    Then RELEASE the guiding button so the arm holds still.")
        if args.dry:
            print("    [--dry] auto-capturing current pose without waiting.")
        else:
            input("    Press Enter to capture the pose...")

        # Briefly enter torque mode to publish state (the comm process only fills
        # shared memory once a control mode runs), read, then release. This holds
        # the arm at its current pose — the mode-switch transient is negligible for
        # a position read. Same pattern as read_state.py.
        robot.start_torque_mode()
        time.sleep(0.5)
        snap = robot.get_state_snapshot()
        robot.end_control()
        ee_pos = snap.ee_pos.to(device)
        print(f"\n  Captured seated fingertip pos: {[round(v, 5) for v in ee_pos.tolist()]}")
        print(f"  Captured fingertip quat:       {[round(v, 5) for v in snap.ee_quat.tolist()]}")

        target_peg_base = ee_pos + ee_to_peg
        # Sim anchors the obs frame to the FIXED asset (hole): entrance =
        # seated peg base + hole_height (factory_env fixed_pos_obs_frame). Build
        # it from target_peg_base so the fingertip->peg-tip distance (ee_to_peg)
        # is threaded through; using ee_pos directly drops it and puts the origin
        # |ee_to_peg| too high, starting the episode above the hole.
        fixed_asset_position = target_peg_base.clone()
        fixed_asset_position[2] += hole_height  # entrance = hole_height above seated peg base

        print(f"\n  -> fixed_asset_position   (hole entrance obs frame): "
              f"{[round(v, 5) for v in fixed_asset_position.tolist()]}")
        print(f"  -> target_peg_base_position (seated peg base):       "
              f"{[round(v, 5) for v in target_peg_base.tolist()]}")

        if args.no_write:
            print("\n  [--no_write] not editing the config.")
        else:
            print(f"\nWriting goal into {out_path} ...")
            set_yaml_list(out_path, "fixed_asset_position", fixed_asset_position.tolist())
            set_yaml_list(out_path, "target_peg_base_position", target_peg_base.tolist())

        if not args.no_verify:
            print("\nVerify: retracting and moving to (goal XY, 5 cm above the hole entrance)...")
            robot.retract_up(float(cfg["reset"]["retract_height_m"]))
            verify_pos = fixed_asset_position.clone()
            verify_pos[2] += 0.05
            orn = cfg["reset"]["hand_init_orn"]
            verify_pose = make_ee_target_pose(verify_pos.cpu().numpy(), np.array(orn))
            robot.reset_to_start_pose(verify_pose)
            vsnap = robot.get_state_snapshot()
            print(f"  At verify pose: {[round(v, 5) for v in vsnap.ee_pos.tolist()]}")
            print("  Eyeball the peg/hole alignment. Re-run calibration if it's off.")
    finally:
        robot.shutdown()

    print(f"\n{'=' * 70}\nCALIBRATION COMPLETE\n{'=' * 70}")


if __name__ == "__main__":
    main()
