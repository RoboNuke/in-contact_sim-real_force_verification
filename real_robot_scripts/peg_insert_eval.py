"""
Real-robot FORGE peg-insert evaluation.

Runs a locally-trained SAC FORGE policy on a physical Franka FR3 for a batch of
peg-in-hole episodes, using the vendored FrankaInterface + the tested
ControlTargets pose path.

Config sourcing:
  * Training-derived values (model architecture, break_force, force-obs mode) come
    from the run's `runtime_config.yaml` — downloaded from wandb (`--wandb_run`) or
    read from the local run dir (run_config.py).
  * FORGE task constants (obs_order, action bounds, ema, force_threshold, dof pose)
    are FORGE-fixed and live in forge_defaults.py.
  * `eval_config.yaml` holds ONLY real-robot-unique values (connection, calibrated
    goal, hardware PD gains, reset noise, camera, wandb overrides).

wandb: the eval is logged as a sibling of the training run — same project/group/
tags + a `real_robot_eval` tag — with aggregate `Eval_Core/*` metrics only. Per-step
data and videos are saved locally (`--with_step_data`), never uploaded.

Usage:
    # off-hardware dry run
    python real_robot_scripts/peg_insert_eval.py \
        --checkpoint runs/viability_test/contact_baseline --agent 0 \
        --config real_robot_scripts/eval_config.yaml \
        --override robot.use_mock=true --no_wandb --num_episodes 2

    # on the robot, with per-step data + video capture
    python real_robot_scripts/peg_insert_eval.py --checkpoint <run> --agent 0 \
        --num_episodes 20 --with_step_data
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
from real_robot_scripts import forge_defaults as FD
from real_robot_scripts.run_config import resolve_runtime_config, derive_wandb_target


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
            node = node.setdefault(k, {})
        try:
            parsed = int(val)
        except ValueError:
            try:
                parsed = float(val)
            except ValueError:
                parsed = {"true": True, "false": False, "null": None}.get(val.lower(), val)
        node[parts[-1]] = parsed
        print(f"  override: {key} = {parsed}")
    return cfg


# ---------------------------------------------------------------------------
# Control targets (pure position hold at the policy's target pose)
# ---------------------------------------------------------------------------

def build_pose_targets(target_pos, target_quat, goal_position, ctrl, device):
    """ControlTargets for a task-space PD pose hold (sel_matrix=0 -> no force axes).

    Real hardware PD gains come from eval_config `control:`; the null-space target
    joint config is the FORGE default.
    """
    prop = torch.as_tensor(ctrl["task_prop_gains"], device=device, dtype=torch.float32)
    deriv = torch.as_tensor(ctrl["task_deriv_gains"], device=device, dtype=torch.float32)
    return ControlTargets(
        target_pos=target_pos.to(device),
        target_quat=target_quat.to(device),
        target_force=torch.zeros(6, device=device),
        sel_matrix=torch.zeros(6, device=device),
        task_prop_gains=prop,
        task_deriv_gains=deriv,
        force_kp=torch.zeros(6, device=device),
        force_di_wrench=torch.zeros(6, device=device),
        pose_ki=torch.zeros(6, device=device),
        pose_integral_clamp=0.0,
        pose_integral_reset_on_target=True,
        default_dof_pos=torch.as_tensor(FD.DEFAULT_DOF_POS, device=device, dtype=torch.float32),
        kp_null=float(ctrl["kp_null"]),
        kd_null=float(ctrl["kd_null"]),
        pos_bounds=torch.as_tensor(ctrl["pos_bounds"], device=device, dtype=torch.float32),
        goal_position=goal_position.to(device),
        ctrl_mode="force_only",
        singularity_damping=0.0,
        partial_inertia_decoupling=False,
        sep_ori=False,
        mass_weighting=bool(ctrl.get("mass_weighting", False)),
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def check_success(peg_base, target_peg_base_pos, xy_centering_threshold, hole_height, success_threshold):
    xy_dist = torch.norm(peg_base[:2] - target_peg_base_pos[:2])
    z_disp = peg_base[2] - target_peg_base_pos[2]
    ok = bool(xy_dist < xy_centering_threshold and z_disp < hole_height * success_threshold)
    return ok, float(xy_dist), float(z_disp)


# ---------------------------------------------------------------------------
# Per-step record (all signals, flat)
# ---------------------------------------------------------------------------

def _vec(prefix, t, names):
    return {f"{prefix}_{n}": float(v) for n, v in zip(names, t.tolist())}


def flatten_step(step, t_mono, snap, raw_action, ema_action, target_pos, target_quat,
                 peg_base, xy_dist, z_disp, force_mag, in_contact, succeeded, terminated, obs):
    rec = {"step": step, "t_mono": t_mono, "force_mag": force_mag,
           "in_contact": bool(in_contact), "succeeded": succeeded, "terminated": terminated,
           "xy_dist_to_target": xy_dist, "z_disp": z_disp}
    rec.update(_vec("ee_pos", snap.ee_pos, "xyz"))
    rec.update(_vec("ee_quat", snap.ee_quat, "wxyz"))
    rec.update(_vec("ee_linvel", snap.ee_linvel, "xyz"))
    rec.update(_vec("ee_angvel", snap.ee_angvel, "xyz"))
    rec.update(_vec("ft", snap.force_torque, ["fx", "fy", "fz", "tx", "ty", "tz"]))
    rec.update(_vec("joint_pos", snap.joint_pos, "0123456"))
    rec.update(_vec("joint_vel", snap.joint_vel, "0123456"))
    rec.update(_vec("joint_torque", snap.joint_torques, "0123456"))
    rec.update(_vec("raw_action", raw_action, "0123456"))
    rec.update(_vec("ema_action", ema_action, "0123456"))
    rec.update(_vec("target_pos", target_pos, "xyz"))
    rec.update(_vec("target_quat", target_quat, "wxyz"))
    rec.update(_vec("peg_base", peg_base, "xyz"))
    rec.update({f"obs_{i:02d}": float(v) for i, v in enumerate(obs.tolist())})
    return rec


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


def run_episode(robot, policy, obs_builder, mapper, cfg, goal_position, device, std_scale,
                break_force, recorder=None, with_step_data=False, out_dir=None, ep_idx=0):
    from real_robot_scripts.trial_data import save_trial

    task = cfg["task"]
    ctrl = cfg["control"]
    contact_force_threshold = float(cfg.get("obs", {}).get("contact_force_threshold", 1.5))
    target_peg_base = torch.as_tensor(task["target_peg_base_position"], device=device, dtype=torch.float32)
    ee_to_peg = torch.as_tensor(task["ee_to_peg_base_offset"], device=device, dtype=torch.float32)
    xy_centering = float(task["xy_centering_threshold"])
    hole_height = float(task["hole_height"])
    success_threshold = float(task["success_threshold"])
    max_steps = int(task["episode_timeout_steps"])
    terminate_on_success = bool(task["terminate_on_success"])
    action_dim = obs_builder.action_dim

    robot.retract_up(float(cfg["reset"]["retract_height_m"]))
    robot.reset_to_start_pose(sample_start_pose(cfg, goal_position, device))
    robot.calibrate_ft_bias()

    mapper.reset()
    prev_actions = torch.zeros(action_dim, device=device)
    snap = robot.get_state_snapshot()
    for _ in range(3):  # JIT warmup
        w_obs = obs_builder.build_observation(snap, goal_position, prev_actions)
        w_act = policy.get_action(w_obs, std_scale=std_scale)
        mapper.step(w_act, snap.ee_pos, snap.ee_quat)
    mapper.reset()
    prev_actions = torch.zeros(action_dim, device=device)

    succeeded = terminated = False
    success_step = termination_step = -1
    ssv_sum = ssjv_sum = max_force = 0.0
    sum_force_in_contact = 0.0
    contact_steps = 0
    energy_sum = 0.0
    step = 0
    step_records = [] if with_step_data else None

    if with_step_data and recorder is not None:
        recorder.begin_segment()
    robot.start_torque_mode()
    for step in range(max_steps):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)
        t_mono = time.monotonic()

        obs = obs_builder.build_observation(snap, goal_position, prev_actions)
        raw_action = policy.get_action(obs, std_scale=std_scale)
        target_pos, target_quat, ema_actions = mapper.step(raw_action, snap.ee_pos, snap.ee_quat)
        robot.set_control_targets(build_pose_targets(target_pos, target_quat, goal_position, ctrl, device))
        prev_actions = ema_actions

        fmag = torch.norm(snap.force_torque[:3]).item()
        in_contact = (snap.force_torque[:3].abs() >= contact_force_threshold).any().item()
        peg_base = snap.ee_pos + ee_to_peg
        is_success, xy_dist, z_disp = check_success(
            peg_base, target_peg_base, xy_centering, hole_height, success_threshold)
        is_break = fmag >= break_force

        ssv_sum += torch.norm(snap.ee_linvel).item()
        ssjv_sum += torch.norm(snap.joint_vel * snap.joint_vel).item()
        max_force = max(max_force, fmag)
        if in_contact:
            sum_force_in_contact += fmag
            contact_steps += 1
        energy_sum += torch.sum(torch.abs(snap.joint_vel * snap.joint_torques)).item()

        if not succeeded and is_success:
            succeeded, success_step = True, step
        if not terminated and is_break:
            terminated, termination_step = True, step

        if with_step_data:
            step_records.append(flatten_step(
                step, t_mono, snap, raw_action, ema_actions, target_pos, target_quat,
                peg_base, xy_dist, z_disp, fmag, in_contact, succeeded, terminated, obs))

        if is_success:
            print(f"    SUCCESS at step {step}")
            if terminate_on_success:
                break
        if is_break:
            print(f"    BREAK at step {step} (force={fmag:.2f} N)")
            break

    robot.end_control()
    frames = recorder.end_segment() if (with_step_data and recorder is not None) else []

    length = step + 1
    outcome = "SUCCESS" if succeeded and not terminated else "BREAK" if terminated else "TIMEOUT"
    result = {
        "episode": ep_idx, "outcome": outcome,
        "succeeded": succeeded, "terminated": terminated, "length": length,
        "ssv": ssv_sum / length if length else 0.0,
        "ssjv": ssjv_sum / length if length else 0.0,
        "max_force": max_force, "sum_force_in_contact": sum_force_in_contact,
        "contact_steps": contact_steps, "energy": energy_sum,
        "success_step": success_step if success_step >= 0 else length,
        "termination_step": termination_step if termination_step >= 0 else length,
    }

    if with_step_data:
        cam = cfg.get("camera", {}) or {}
        paths = save_trial(step_records, frames, out_dir, ep_idx, break_force, outcome,
                           overlay=bool(cam.get("overlay", True)),
                           fallback_fps=int(cam.get("fps", 30)))
        result["mp4"] = paths.get("mp4")
        result["data_pkl"] = paths.get("pkl")

    return result


# ---------------------------------------------------------------------------
# read-state diagnostic
# ---------------------------------------------------------------------------

def print_observation(obs, obs_builder):
    print("\n" + "=" * 70 + "\n  ASSEMBLED OBSERVATION (input to normalizer)\n" + "=" * 70)
    idx = 0
    for name in obs_builder.obs_order:
        dim = OBS_DIM_MAP[name]
        print(f"  [{idx:>2}:{idx + dim:<2}] {name:<24} {[f'{v:.5f}' for v in obs[idx:idx + dim].tolist()]}")
        idx += dim
    print(f"  [{idx:>2}:{idx + obs_builder.action_dim:<2}] {'prev_actions':<24} "
          f"{[f'{v:.5f}' for v in obs[idx:idx + obs_builder.action_dim].tolist()]}")
    print(f"\n  obs_dim: {obs.shape[0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Real-robot FORGE peg-insert evaluation")
    p.add_argument("--checkpoint", required=True, help="Training run dir (contains <agent>/checkpoints)")
    p.add_argument("--agent", type=int, default=0)
    p.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    p.add_argument("--config", default="real_robot_scripts/eval_config.yaml")
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--std_scale", type=float, default=None)
    p.add_argument("--read_state", action="store_true", help="Print one live observation and exit")
    p.add_argument("--with_step_data", action="store_true",
                   help="Capture ALL per-step data + a RealSense video per trial (saved locally)")
    p.add_argument("--data_dir", default=None, help="Local dir for per-step data/videos")
    p.add_argument("--override", action="append", default=[], help="Config override key=value")
    # runtime_config sourcing from wandb (else read local run dir)
    p.add_argument("--wandb_run", default=None, help="Source wandb run id (download runtime_config.yaml)")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()

    torch.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)

    print("=" * 80 + "\nREAL-ROBOT FORGE PEG-INSERT EVALUATION\n" + "=" * 80)
    cfg = load_config(args.config, args.override)
    device = args.device
    goal_position = torch.as_tensor(cfg["task"]["fixed_asset_position"], device=device, dtype=torch.float32)
    std_scale = args.std_scale if args.std_scale is not None else float(cfg.get("policy", {}).get("std_scale", 0.0))

    # 1. Resolve the training config (wandb download or local run dir).
    train_cfg, cfg_src, run_id = resolve_runtime_config(
        args.checkpoint, args.agent, wandb_run=args.wandb_run,
        wandb_entity=args.wandb_entity, wandb_project=args.wandb_project)

    # 2. Policy (break_force + model come from the training config).
    policy = ForgePolicy(args.checkpoint, train_cfg, agent_idx=args.agent, step=args.step, device=device)
    break_force = policy.break_force
    override_bf = cfg.get("task", {}).get("break_force_threshold")
    if override_bf is not None and abs(float(override_bf) - break_force) > 1e-6:
        print(f"  [warn] eval_config task.break_force_threshold={override_bf} overrides "
              f"training break_force={break_force} N")
        break_force = float(override_bf)
    print(f"  Using break_force = {break_force} N")

    # 3. Obs builder + FORGE-default action mapper.
    force_threshold = float(cfg.get("obs", {}).get("force_threshold", FD.FORCE_THRESHOLD_DEFAULT))
    obs_builder = ObservationBuilder(force_threshold=force_threshold,
                                     action_dim=policy.action_dim, device=device)
    obs_builder.validate_against_checkpoint(policy.obs_dim)
    ctrl = cfg["control"]
    mapper = ForgeActionMapper(
        goal_position=goal_position,
        ema_factor=float(cfg.get("control", {}).get("ema_factor", FD.EMA_FACTOR_DEFAULT)),
        pos_action_bounds=FD.POS_ACTION_BOUNDS, rot_action_bounds=FD.ROT_ACTION_BOUNDS,
        pos_action_threshold=FD.POS_ACTION_THRESHOLD, rot_action_threshold=FD.ROT_ACTION_THRESHOLD,
        action_dim=policy.action_dim, device=device)

    # 4. Robot + optional recorder.
    print("\nInitializing robot interface...")
    robot = FrankaInterface(cfg, device=device)
    recorder = None
    out_dir = None
    if args.with_step_data:
        from real_robot_scripts.camera import make_recorder
        from real_robot_scripts.trial_data import save_summary
        data_root = args.data_dir or cfg.get("data_dir", "data/real_robot_eval")
        run_name = f"{os.path.basename(args.checkpoint.rstrip('/'))}_a{args.agent}_s{policy.step}"
        out_dir = os.path.join(data_root, run_name)
        recorder = make_recorder(cfg.get("camera", {}), use_mock=bool(cfg["robot"].get("use_mock", False)))
        recorder.start()
        print(f"[with_step_data] per-trial data/videos -> {out_dir}")

    try:
        if args.read_state:
            robot.start_torque_mode()
            time.sleep(1.0)
            snap = robot.get_state_snapshot()
            robot.end_control()
            obs = obs_builder.build_observation(snap, goal_position, torch.zeros(policy.action_dim, device=device))
            print(f"\n  ee_pos={snap.ee_pos.tolist()}")
            print(f"  force_torque={[round(v, 3) for v in snap.force_torque.tolist()]}")
            print(f"  goal={goal_position.tolist()}")
            print_observation(obs, obs_builder)
            return

        print("\nClosing gripper...")
        robot.close_gripper()
        print(f"Moving to default joint config {FD.DEFAULT_DOF_POS}...")
        robot.move_to_joint_positions(
            torch.as_tensor(FD.DEFAULT_DOF_POS, device=device, dtype=torch.float32), duration_sec=3.0)

        print(f"\n{'=' * 80}\nEVALUATING {args.num_episodes} EPISODES "
              f"({'deterministic' if std_scale <= 0 else f'std_scale={std_scale}'})\n{'=' * 80}")

        results = []
        for ep in range(args.num_episodes):
            print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
            res = run_episode(robot, policy, obs_builder, mapper, cfg, goal_position, device,
                              std_scale, break_force, recorder=recorder,
                              with_step_data=args.with_step_data, out_dir=out_dir, ep_idx=ep)
            print(f"    {res['outcome']} (len={res['length']}, max_f={res['max_force']:.2f} N)")
            results.append(res)

        metrics = compute_metrics(results)
        print(f"\n{'=' * 80}\nRESULTS\n{'=' * 80}")
        print(f"  Successes: {metrics['num_successful_completions']}/{metrics['total_episodes']}")
        print(f"  Breaks:    {metrics['num_breaks']}/{metrics['total_episodes']}")
        print(f"  Timeouts:  {metrics['num_failed_timeouts']}/{metrics['total_episodes']}")
        print(f"  Max force: {metrics['max_force']:.2f} N   Energy: {metrics['energy']:.2f}")

        if args.with_step_data:
            save_summary(results, out_dir)

        if not args.no_wandb:
            import wandb
            wcfg = cfg.get("wandb", {}) or {}
            target = derive_wandb_target(
                train_cfg, args.agent, extra_tags=wcfg.get("extra_tags"),
                project_override=wcfg.get("project"), entity_override=wcfg.get("entity"))
            run = wandb.init(
                project=target["project"], entity=target["entity"] or None,
                group=target["group"], tags=target["tags"], name=target["name"],
                config={"checkpoint": args.checkpoint, "agent": args.agent, "step": policy.step,
                        "source_run_id": run_id, "num_episodes": args.num_episodes,
                        "std_scale": std_scale, "break_force": break_force},
            )
            # Aggregate metrics only — never per-step data or videos.
            wandb.log({**{f"Eval_Core/{k}": v for k, v in metrics.items()}, "total_steps": policy.step})
            wandb.finish()
            print(f"  Logged to wandb ({target['project']}/{target['group']}, "
                  f"tags={target['tags']}): {run.url}")
        else:
            print("  (wandb disabled)")
    finally:
        if recorder is not None:
            recorder.close()
        robot.shutdown()
    print(f"\n{'=' * 80}\nEVALUATION COMPLETE\n{'=' * 80}")


if __name__ == "__main__":
    main()
