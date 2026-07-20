"""
Real-robot FORGE peg-insert evaluation.

Runs a locally-trained SAC FORGE policy on a physical Franka FR3 for a batch of
peg-in-hole episodes and (optionally) logs aggregate metrics to wandb — the real
counterpart of the sim FORGE training/eval, using the vendored FrankaInterface +
the tested ControlTargets pose path.

Pipeline (per policy, per episode):
    retract -> move to start pose (above the calibrated goal, + noise)
    -> re-tare F/T -> warmup (JIT) -> torque mode
    -> 15 Hz loop { snapshot -> obs -> policy -> FORGE action map -> ControlTargets }
    -> success (geometric) / break (force) detection -> metrics.

Usage:
    # off-hardware dry run
    python real_robot_scripts/peg_insert_eval.py \
        --checkpoint runs/forge_pih/<exp> --agent 0 \
        --config real_robot_scripts/eval_config.yaml \
        --override robot.use_mock=true --no_wandb --num_episodes 2

    # print the assembled observation for one live state and exit
    python real_robot_scripts/peg_insert_eval.py --checkpoint <run> --read_state

    # on the robot
    python real_robot_scripts/peg_insert_eval.py --checkpoint <run> --agent 0 \
        --num_episodes 20
"""

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_robot_scripts.pro_robot_interface import FrankaInterface
from real_robot_scripts.robot_interface import make_ee_target_pose
from real_robot_scripts.hybrid_controller import ControlTargets
from real_robot_scripts.observation_builder import ObservationBuilder, OBS_DIM_MAP
from real_robot_scripts.forge_policy import ForgePolicy
from real_robot_scripts.forge_action_map import ForgeActionMapper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str, overrides: List[str] = None) -> dict:
    """Load the eval YAML and apply dotted key=value overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node[k]
        # parse scalar
        try:
            parsed = int(val)
        except ValueError:
            try:
                parsed = float(val)
            except ValueError:
                parsed = {"true": True, "false": False}.get(val.lower(), val)
        node[parts[-1]] = parsed
        print(f"  override: {key} = {parsed}")
    return cfg


# ---------------------------------------------------------------------------
# Control targets (pure position hold at the policy's target pose)
# ---------------------------------------------------------------------------

def build_pose_targets(target_pos, target_quat, goal_position, ctrl, device):
    """ControlTargets for a task-space PD pose hold (sel_matrix=0 -> no force axes).

    Mirrors contact_force_test.make_position_targets, parameterized by the eval
    config `control:` block.
    """
    prop = torch.as_tensor(ctrl["task_prop_gains"], device=device, dtype=torch.float32)
    deriv = torch.as_tensor(ctrl["task_deriv_gains"], device=device, dtype=torch.float32)
    return ControlTargets(
        target_pos=target_pos.to(device),
        target_quat=target_quat.to(device),
        target_force=torch.zeros(6, device=device),
        sel_matrix=torch.zeros(6, device=device),          # 0 -> pure position on every axis
        task_prop_gains=prop,
        task_deriv_gains=deriv,
        force_kp=torch.zeros(6, device=device),
        force_di_wrench=torch.zeros(6, device=device),
        pose_ki=torch.zeros(6, device=device),
        pose_integral_clamp=0.0,
        pose_integral_reset_on_target=True,
        default_dof_pos=torch.as_tensor(ctrl["default_dof_pos"], device=device, dtype=torch.float32),
        kp_null=float(ctrl["kp_null"]),
        kd_null=float(ctrl["kd_null"]),
        pos_bounds=torch.as_tensor(ctrl["pos_bounds"], device=device, dtype=torch.float32),
        goal_position=goal_position.to(device),
        ctrl_mode="force_only",                             # no-op with sel_matrix=0
        singularity_damping=0.0,
        partial_inertia_decoupling=False,
        sep_ori=False,
        mass_weighting=bool(ctrl.get("mass_weighting", False)),
    )


# ---------------------------------------------------------------------------
# Detection (matches sim FORGE / breakable-peg logic on real state)
# ---------------------------------------------------------------------------

def check_success(ee_pos, ee_to_peg_base_offset, target_peg_base_pos,
                  xy_centering_threshold, hole_height, success_threshold) -> bool:
    peg_base = ee_pos + ee_to_peg_base_offset
    xy_dist = torch.norm(peg_base[:2] - target_peg_base_pos[:2])
    z_disp = peg_base[2] - target_peg_base_pos[2]
    return bool(xy_dist < xy_centering_threshold and z_disp < hole_height * success_threshold)


def check_break(force_torque, break_force_threshold) -> bool:
    return bool(torch.norm(force_torque[:3]) >= break_force_threshold)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(episode_results: List[dict]) -> Dict[str, float]:
    n = len(episode_results)
    if n == 0:
        raise RuntimeError("No episodes to aggregate")
    m = {"total_episodes": n}
    num_success = num_breaks = num_timeouts = 0
    success_steps, break_steps = [], []
    for ep in episode_results:
        s, t = ep["succeeded"], ep["terminated"]
        if s and not t:
            num_success += 1
        elif t:
            num_breaks += 1
        else:
            num_timeouts += 1
        if s and (not t or ep["success_step"] < ep["termination_step"]):
            success_steps.append(ep["success_step"])
        if t and (not s or ep["termination_step"] <= ep["success_step"]):
            break_steps.append(ep["termination_step"])
    m["num_successful_completions"] = num_success
    m["num_breaks"] = num_breaks
    m["num_failed_timeouts"] = num_timeouts
    m["episode_length"] = sum(ep["length"] for ep in episode_results) / n
    m["avg_steps_to_success"] = (sum(success_steps) / len(success_steps)) if success_steps else 0.0
    m["avg_steps_to_break"] = (sum(break_steps) / len(break_steps)) if break_steps else 0.0
    m["ssv"] = sum(ep["ssv"] for ep in episode_results) / n
    m["ssjv"] = sum(ep["ssjv"] for ep in episode_results) / n
    m["max_force"] = max(ep["max_force"] for ep in episode_results)
    tot_fc = sum(ep["sum_force_in_contact"] for ep in episode_results)
    tot_cs = sum(ep["contact_steps"] for ep in episode_results)
    m["avg_force_in_contact"] = (tot_fc / tot_cs) if tot_cs > 0 else 0.0
    m["energy"] = sum(ep["energy"] for ep in episode_results) / n
    return m


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def sample_start_pose(cfg, goal_position, device):
    r = cfg["reset"]
    hand_init_pos = torch.as_tensor(r["hand_init_pos"], device=device, dtype=torch.float32)
    pos_noise_rng = torch.as_tensor(r["hand_init_pos_noise"], device=device, dtype=torch.float32)
    start_pos_noise = (2 * torch.rand(3, device=device) - 1) * pos_noise_rng
    target_ee_pos = goal_position + hand_init_pos + start_pos_noise
    orn = r["hand_init_orn"]
    orn_noise = r["hand_init_orn_noise"]
    yaw_noise = (2 * float(torch.rand(1).item()) - 1) * orn_noise[2]
    target_rpy = np.array([orn[0], orn[1], orn[2] + yaw_noise])
    return make_ee_target_pose(target_ee_pos.cpu().numpy(), target_rpy)


def run_episode(robot, policy, obs_builder, mapper, cfg, goal_position, device, std_scale):
    task = cfg["task"]
    ctrl = cfg["control"]
    contact_force_threshold = float(cfg["obs"].get("contact_force_threshold", 1.5))
    target_peg_base = torch.as_tensor(task["target_peg_base_position"], device=device, dtype=torch.float32)
    ee_to_peg = torch.as_tensor(task["ee_to_peg_base_offset"], device=device, dtype=torch.float32)
    xy_centering = float(task["xy_centering_threshold"])
    hole_height = float(task["hole_height"])
    success_threshold = float(task["success_threshold"])
    break_force = float(task["break_force_threshold"])
    max_steps = int(task["episode_timeout_steps"])
    terminate_on_success = bool(task["terminate_on_success"])
    action_dim = obs_builder.action_dim

    # 1. Retract + move to a fresh start pose above the hole.
    robot.retract_up(float(cfg["reset"]["retract_height_m"]))
    start_pose = sample_start_pose(cfg, goal_position, device)
    robot.reset_to_start_pose(start_pose)

    # 2. Re-tare the F/T estimate in this (free-space) start pose.
    robot.calibrate_ft_bias()

    # 3. Warmup (JIT compile the policy + control math), then reset episode state.
    mapper.reset()
    prev_actions = torch.zeros(action_dim, device=device)
    snap = robot.get_state_snapshot()
    for _ in range(3):
        w_obs = obs_builder.build_observation(snap, goal_position, prev_actions)
        w_act = policy.get_action(w_obs, std_scale=std_scale)
        mapper.step(w_act, snap.ee_pos, snap.ee_quat)
    mapper.reset()
    prev_actions = torch.zeros(action_dim, device=device)

    # 4. Torque mode + 15 Hz policy loop.
    succeeded = terminated = False
    success_step = termination_step = -1
    ssv_sum = ssjv_sum = max_force = 0.0
    sum_force_in_contact = 0.0
    contact_steps = 0
    energy_sum = 0.0
    step = 0

    robot.start_torque_mode()
    for step in range(max_steps):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)

        obs = obs_builder.build_observation(snap, goal_position, prev_actions)
        raw_action = policy.get_action(obs, std_scale=std_scale)
        target_pos, target_quat, ema_actions = mapper.step(raw_action, snap.ee_pos, snap.ee_quat)
        targets = build_pose_targets(target_pos, target_quat, goal_position, ctrl, device)
        robot.set_control_targets(targets)
        prev_actions = ema_actions

        # metrics
        ssv_sum += torch.norm(snap.ee_linvel).item()
        ssjv_sum += torch.norm(snap.joint_vel * snap.joint_vel).item()
        fmag = torch.norm(snap.force_torque[:3]).item()
        max_force = max(max_force, fmag)
        if (snap.force_torque[:3].abs() >= contact_force_threshold).any().item():
            sum_force_in_contact += fmag
            contact_steps += 1
        energy_sum += torch.sum(torch.abs(snap.joint_vel * snap.joint_torques)).item()

        if not succeeded and check_success(snap.ee_pos, ee_to_peg, target_peg_base,
                                           xy_centering, hole_height, success_threshold):
            succeeded, success_step = True, step
            print(f"    SUCCESS at step {step}")
            if terminate_on_success:
                break
        if not terminated and check_break(snap.force_torque, break_force):
            terminated, termination_step = True, step
            print(f"    BREAK at step {step} (force={fmag:.2f} N)")
            break

    robot.end_control()

    length = step + 1
    return {
        "succeeded": succeeded,
        "terminated": terminated,
        "length": length,
        "ssv": ssv_sum / length if length else 0.0,
        "ssjv": ssjv_sum / length if length else 0.0,
        "max_force": max_force,
        "sum_force_in_contact": sum_force_in_contact,
        "contact_steps": contact_steps,
        "energy": energy_sum,
        "success_step": success_step if success_step >= 0 else length,
        "termination_step": termination_step if termination_step >= 0 else length,
    }


# ---------------------------------------------------------------------------
# read-state diagnostic
# ---------------------------------------------------------------------------

def print_observation(obs, obs_builder):
    print("\n" + "=" * 70)
    print("  ASSEMBLED OBSERVATION (input to normalizer)")
    print("=" * 70)
    idx = 0
    for name in obs_builder.obs_order:
        dim = OBS_DIM_MAP[name]
        vals = [f"{v:.5f}" for v in obs[idx:idx + dim].tolist()]
        print(f"  [{idx:>2}:{idx + dim:<2}] {name:<24} {vals}")
        idx += dim
    vals = [f"{v:.5f}" for v in obs[idx:idx + obs_builder.action_dim].tolist()]
    print(f"  [{idx:>2}:{idx + obs_builder.action_dim:<2}] {'prev_actions':<24} {vals}")
    print(f"\n  obs_dim: {obs.shape[0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Real-robot FORGE peg-insert evaluation")
    p.add_argument("--checkpoint", required=True, help="Training run dir (contains <agent>/checkpoints)")
    p.add_argument("--agent", type=int, default=0, help="Block-parallel agent slot to eval")
    p.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    p.add_argument("--config", default="real_robot_scripts/eval_config.yaml")
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--std_scale", type=float, default=None, help="Override policy.std_scale")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--read_state", action="store_true",
                   help="Print one assembled observation from the live robot and exit")
    p.add_argument("--override", action="append", default=[], help="Config override key=value (repeatable)")
    args = p.parse_args()

    torch.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)

    print("=" * 80)
    print("REAL-ROBOT FORGE PEG-INSERT EVALUATION")
    print("=" * 80)
    cfg = load_config(args.config, args.override)
    device = args.device

    # goal = hole-entrance obs frame (fixed_asset_position calibrated to the mouth).
    goal_position = torch.as_tensor(cfg["task"]["fixed_asset_position"], device=device, dtype=torch.float32)
    std_scale = args.std_scale if args.std_scale is not None else float(cfg["policy"].get("std_scale", 0.0))

    # Policy + obs builder.
    policy = ForgePolicy(args.checkpoint, agent_idx=args.agent, step=args.step, device=device)
    obs_builder = ObservationBuilder(
        force_threshold=float(cfg["obs"]["force_threshold"]),
        action_dim=policy.action_dim,
        device=device,
    )
    obs_builder.validate_against_checkpoint(policy.obs_dim)

    ctrl = cfg["control"]
    mapper = ForgeActionMapper(
        goal_position=goal_position,
        ema_factor=float(ctrl["ema_factor"]),
        pos_action_bounds=ctrl["pos_action_bounds"],
        rot_action_bounds=ctrl["rot_action_bounds"],
        pos_action_threshold=ctrl["pos_action_threshold"],
        rot_action_threshold=ctrl["rot_action_threshold"],
        action_dim=policy.action_dim,
        device=device,
    )

    print("\nInitializing robot interface...")
    robot = FrankaInterface(cfg, device=device)

    try:
        # --- read-state diagnostic: assemble one obs and quit ---
        if args.read_state:
            robot.start_torque_mode()
            time.sleep(1.0)  # let the F/T EMA settle
            snap = robot.get_state_snapshot()
            robot.end_control()
            obs = obs_builder.build_observation(
                snap, goal_position, torch.zeros(policy.action_dim, device=device))
            print(f"\n  ee_pos={snap.ee_pos.tolist()}")
            print(f"  ee_quat={snap.ee_quat.tolist()}")
            print(f"  force_torque={[round(v, 3) for v in snap.force_torque.tolist()]}")
            print(f"  goal (fixed_asset_position)={goal_position.tolist()}")
            print_observation(obs, obs_builder)
            return

        print("\nClosing gripper...")
        robot.close_gripper()
        print(f"Moving to default joint config {ctrl['default_dof_pos']}...")
        robot.move_to_joint_positions(
            torch.as_tensor(ctrl["default_dof_pos"], device=device, dtype=torch.float32),
            duration_sec=3.0,
        )

        print(f"\n{'=' * 80}\nEVALUATING {args.num_episodes} EPISODES "
              f"({'deterministic' if std_scale <= 0 else f'std_scale={std_scale}'})\n{'=' * 80}")

        results = []
        for ep in range(args.num_episodes):
            print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
            res = run_episode(robot, policy, obs_builder, mapper, cfg, goal_position, device, std_scale)
            outcome = ("SUCCESS" if res["succeeded"] and not res["terminated"]
                       else "BREAK" if res["terminated"] else "TIMEOUT")
            print(f"    {outcome} (len={res['length']}, max_f={res['max_force']:.2f} N)")
            results.append(res)

        metrics = compute_metrics(results)
        print(f"\n{'=' * 80}\nRESULTS\n{'=' * 80}")
        print(f"  Successes: {metrics['num_successful_completions']}/{metrics['total_episodes']}")
        print(f"  Breaks:    {metrics['num_breaks']}/{metrics['total_episodes']}")
        print(f"  Timeouts:  {metrics['num_failed_timeouts']}/{metrics['total_episodes']}")
        print(f"  Avg length: {metrics['episode_length']:.1f}")
        print(f"  Max force:  {metrics['max_force']:.2f} N")
        print(f"  Energy:     {metrics['energy']:.2f}")

        if not args.no_wandb:
            import wandb
            wcfg = cfg.get("wandb", {})
            run_name = args.wandb_name or f"Eval_RealRobot_{os.path.basename(args.checkpoint.rstrip('/'))}_a{args.agent}"
            tags = list(wcfg.get("tags", [])) + [f"source_run:{os.path.basename(args.checkpoint.rstrip('/'))}"]
            run = wandb.init(
                project=wcfg.get("project", "forge_pih"),
                entity=wcfg.get("entity") or None,
                name=run_name,
                tags=tags,
                config={"checkpoint": args.checkpoint, "agent": args.agent,
                        "step": policy.step, "num_episodes": args.num_episodes,
                        "std_scale": std_scale, "eval_config": cfg},
            )
            wandb.log({**{f"Eval_Core/{k}": v for k, v in metrics.items()},
                       "total_steps": policy.step})
            wandb.finish()
            print(f"  Logged to wandb: {run.url}")
        else:
            print("  (wandb disabled)")
    finally:
        robot.shutdown()
    print(f"\n{'=' * 80}\nEVALUATION COMPLETE\n{'=' * 80}")


if __name__ == "__main__":
    main()
