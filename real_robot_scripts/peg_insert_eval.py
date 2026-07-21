"""
Real-robot FORGE peg-insert policy evaluation (process-based FrankaInterface).

Structure/robustness ported from RoboNuke's eval/pro_real_robot_eval.py — the
proven real-robot runner — with our stack swapped in:
  * policy loading  : ForgePolicy (local checkpoints; not SimBaNet/wandb-cache)
  * action scaling  : ForgeActionMapper + build_pose_targets (FORGE incremental map)
  * training cfg    : run_config.resolve_runtime_config (break_force, model, force-obs)
  * logging         : our Eval_Core/* wandb (sibling of the training run) + optional
                      per-step DataFrame/mp4 via camera.py/trial_data.py (--with_step_data)

Robustness kept from RoboNuke: non-blocking keyboard control (skip/pause/calibrate/
resume/quit), per-episode 5x retry with robot.error_recovery() (clears FR3 Reflex),
reset retry inside the episode, pause/calibrate loop, try/finally shutdown.

FORGE is position-control only; the settle/torque-mode re-tare are intentionally NOT
present here (RoboNuke does neither — it calibrates ft_bias at the start pose).

Usage:
    python real_robot_scripts/peg_insert_eval.py --checkpoint <run_dir> --agent 0 \
        --num_episodes 20 [--with_step_data] [--no_wandb]
"""

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_robot_scripts.pro_robot_interface import FrankaInterface
from real_robot_scripts.robot_interface import make_ee_target_pose, SafetyViolation
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
        singularity_damping=float(ctrl.get("singularity_damping", 0.0)),
        partial_inertia_decoupling=False,
        sep_ori=False,
        mass_weighting=bool(ctrl.get("mass_weighting", False)),
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def check_success(peg_base, target_peg_base_pos, xy_centering_threshold, hole_height, threshold):
    """Peg-insert success/engage proxy (matches sim _get_curr_successes geometry).

    `threshold` is success_threshold or engage_threshold. Returns (ok, xy_dist, z_disp).
    """
    xy_dist = torch.norm(peg_base[:2] - target_peg_base_pos[:2])
    z_disp = peg_base[2] - target_peg_base_pos[2]
    ok = bool(xy_dist < xy_centering_threshold and z_disp < hole_height * threshold)
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
    m["num_breaks_engaged"] = sum(1 for ep in episode_results if ep["terminated"] and ep["engaged"])
    # Breaks caused by a panda/robot error during the rollout (not a force violation).
    m["num_breaks_panda_error"] = sum(1 for ep in episode_results if ep["terminated"] and ep.get("panda_error"))
    m["num_timeouts_engaged"] = sum(
        1 for ep in episode_results if not ep["succeeded"] and not ep["terminated"] and ep["engaged"])
    m["episode_length"] = sum(ep["length"] for ep in episode_results) / n
    m["avg_steps_to_success"] = (sum(success_steps) / len(success_steps)) if success_steps else 0.0
    m["avg_steps_to_break"] = (sum(break_steps) / len(break_steps)) if break_steps else 0.0
    m["ssv"] = sum(ep["ssv"] for ep in episode_results) / n
    m["ssjv"] = sum(ep["ssjv"] for ep in episode_results) / n
    tot_f = sum(ep["sum_force"] for ep in episode_results)
    tot_len = sum(ep["length"] for ep in episode_results)
    m["avg_force"] = (tot_f / tot_len) if tot_len > 0 else 0.0
    m["max_force"] = max(ep["max_force"] for ep in episode_results)
    tot_fc = sum(ep["sum_force_in_contact"] for ep in episode_results)
    tot_cs = sum(ep["contact_steps"] for ep in episode_results)
    m["avg_force_in_contact"] = (tot_fc / tot_cs) if tot_cs > 0 else 0.0
    m["energy"] = sum(ep["energy"] for ep in episode_results) / n
    return m


# ---------------------------------------------------------------------------
# Non-blocking keyboard controller (ported verbatim from pro_real_robot_eval.py)
# ---------------------------------------------------------------------------

class EvalKeyboardController:
    """Non-blocking keyboard listener for eval control.

    Keys during an episode:  's' skip (end as BREAK), 'p' pause (finish then pause).
    Keys while paused:        'c' calibrate (goal XY, 5cm above goal Z), Enter resume.
    Any time:                 ESC quit (end episode + shut down).

    Disabled automatically when stdin is not a TTY (mock/dry runs) — all should_*
    properties return False so the loop runs unattended.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._skip = False
        self._pause = False
        self._resume = False
        self._calibrate = False
        self._quit = False
        self._paused = False
        self._stop = threading.Event()
        self._old_settings = None
        self._enabled = False

    def start(self):
        """Save terminal settings, set raw mode, start listener thread (TTY only)."""
        if not sys.stdin.isatty():
            self._enabled = False
            return
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except (termios.error, ValueError):
            self._enabled = False
            return
        self._enabled = True
        self._stop.clear()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def stop(self):
        """Restore terminal settings."""
        self._stop.set()
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    @property
    def should_skip(self) -> bool:
        with self._lock:
            val = self._skip
            self._skip = False
            return val

    @property
    def should_pause(self) -> bool:
        with self._lock:
            return self._pause

    @property
    def should_calibrate(self) -> bool:
        with self._lock:
            val = self._calibrate
            self._calibrate = False
            return val

    @property
    def should_resume(self) -> bool:
        with self._lock:
            val = self._resume
            self._resume = False
            return val

    @property
    def should_quit(self) -> bool:
        with self._lock:
            return self._quit

    def set_paused(self, paused: bool):
        with self._lock:
            self._paused = paused
            if not paused:
                self._pause = False
                self._resume = False
                self._calibrate = False

    @staticmethod
    def raw_print(msg: str):
        """Print with CR+LF so output isn't garbled in raw terminal mode."""
        sys.stdout.write(msg + "\r\n")
        sys.stdout.flush()

    def _read_loop(self):
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                ch = sys.stdin.read(1)
                with self._lock:
                    if ch == '\x1b':
                        self._quit = True
                        self._skip = True
                    elif ch.lower() == 's' and not self._paused:
                        self._skip = True
                    elif ch.lower() == 'p' and not self._paused:
                        self._pause = True
                    elif ch.lower() == 'c' and self._paused:
                        self._calibrate = True
                    elif ch in ('\r', '\n') and self._paused:
                        self._resume = True


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(robot, policy, obs_builder, mapper, cfg, ctrl, goal_position,
                target_peg_base, ee_to_peg, break_force, contact_force_threshold,
                episode_noise, hand_init_pos, hand_init_orn, keyboard, device,
                std_scale=0.0, recorder=None, with_step_data=False, out_dir=None, ep_idx=0,
                log_trajectory=False, traj_dir=None, continue_on_break=False):
    """Run one evaluation episode. Returns a result dict, or None if the reset
    motion could not recover after retries (caller aborts remaining episodes).

    If continue_on_break is True, the rollout keeps running to max_steps (collecting
    step data / trajectory), but the METRICS freeze at the first terminal event
    (success/force-break) so tracking still sees the logical episode. Only a panda
    error or keyboard skip ends the rollout early."""
    task = cfg["task"]
    xy_centering = float(task["xy_centering_threshold"])
    hole_height = float(task["hole_height"])
    success_threshold = float(task["success_threshold"])
    engage_threshold = float(task.get("engage_threshold", success_threshold))
    max_steps = int(task["episode_timeout_steps"])
    terminate_on_success = bool(task["terminate_on_success"])
    action_dim = obs_builder.action_dim
    rp = EvalKeyboardController.raw_print

    # --- Per-episode noise: noisy action/obs frame + start-pose jitter ---
    noisy_goal = goal_position + episode_noise["goal_pos_noise"]
    start_pos_noise = episode_noise["start_pos_noise"]
    target_ee_pos = goal_position + hand_init_pos + start_pos_noise
    target_pose = make_ee_target_pose(target_ee_pos.cpu().numpy(), np.array(hand_init_orn))

    # FORGE action-frame origin follows the (noisy) goal for this episode.
    mapper.goal_position = noisy_goal.to(device, dtype=torch.float32)

    # --- PRE-ROLLOUT (reset + tare + warmup), retried up to MAX_MOTION_RETRIES ---
    #     Everything BEFORE the rollout is retried (the FR3 intermittently rejects the
    #     reset move / reflexes out). If it can't recover after all attempts, return
    #     None and the caller aborts this policy. Errors DURING the rollout are NOT
    #     retried — they count as a break (see the rollout try/except below).
    MAX_MOTION_RETRIES = 5
    retract_height = float(cfg["reset"]["retract_height_m"])
    prepared = False
    for attempt in range(MAX_MOTION_RETRIES):
        try:
            robot.retract_up(retract_height)
            robot.reset_to_start_pose(target_pose)
            # Tare F/T at the start pose (free space, JointImpedance;
            # window = robot.ft_calibration_duration_sec).
            robot.calibrate_ft_bias()
            # Warmup (JIT): policy inference + one snapshot read; no rollout.
            mapper.reset()
            obs_builder.reset()
            snap = robot.get_state_snapshot()
            zero_pa = torch.zeros(action_dim, device=device)
            for _ in range(3):
                w_obs = obs_builder.build_observation(snap, noisy_goal, zero_pa)
                w_act = policy.get_action(w_obs, std_scale=std_scale)
                mapper.step(w_act, snap.ee_pos, snap.ee_quat)
            mapper.reset()
            obs_builder.reset()   # zero the force-history buffer before the rollout
            prepared = True
            break
        except Exception as e:
            rp(f"  [PRE-ROLLOUT RETRY {attempt + 1}/{MAX_MOTION_RETRIES}] {type(e).__name__}: {e}")
            try:
                robot.end_control()
            except Exception:
                pass
            try:
                robot.error_recovery()
            except Exception:
                pass
            time.sleep(1.0)
    if not prepared:
        rp(f"  [MOTION FAILED] pre-rollout failed all {MAX_MOTION_RETRIES} attempts")
        return None

    prev_actions = torch.zeros(action_dim, device=device)

    # --- Episode tracking ---
    succeeded = engaged = terminated = False
    panda_error = False   # break caused by a robot/comm fault during the rollout (not force)
    success_step = termination_step = -1
    # "logical" episode end (first terminal event). Metrics freeze here; with
    # continue_on_break the rollout keeps collecting step data past it to max_steps.
    logical_done = False
    logical_end_step = -1
    ssv_sum = ssjv_sum = 0.0
    max_force = sum_force = sum_force_in_contact = 0.0
    contact_steps = 0
    energy_sum = 0.0
    step_records = [] if with_step_data else None

    if with_step_data and recorder is not None:
        recorder.begin_segment()
    robot.start_torque_mode(log_trajectory=log_trajectory)

    step = 0
    try:
        for step in range(max_steps):
            robot.wait_for_policy_step()
            snap = robot.get_state_snapshot()
            robot.check_safety(snap)

            # Keyboard skip ('s' / ESC) — end the episode immediately as a BREAK.
            if keyboard.should_skip:
                terminated = True
                if not logical_done:
                    logical_end_step = step
                logical_done = True
                break

            t_mono = time.monotonic()
            obs = obs_builder.build_observation(snap, noisy_goal, prev_actions)
            raw_action = policy.get_action(obs, std_scale=std_scale)
            target_pos, target_quat, ema_actions = mapper.step(raw_action, snap.ee_pos, snap.ee_quat)
            robot.set_control_targets(build_pose_targets(target_pos, target_quat, noisy_goal, ctrl, device))
            prev_actions = ema_actions

            fmag = torch.norm(snap.force_torque[:3]).item()
            in_contact = (snap.force_torque[:3].abs() >= contact_force_threshold).any().item()
            peg_base = snap.ee_pos + ee_to_peg

            # ---- Metrics: accumulate only while the LOGICAL episode is still running ----
            if not logical_done:
                ssv_sum += torch.norm(snap.ee_linvel).item()
                ssjv_sum += torch.norm(snap.joint_vel * snap.joint_vel).item()
                sum_force += fmag
                max_force = max(max_force, fmag)
                if in_contact:
                    sum_force_in_contact += fmag
                    contact_steps += 1
                energy_sum += torch.sum(torch.abs(snap.joint_vel * snap.joint_torques)).item()

            # ---- Detection (against the TRUE seated peg base, not the noisy goal) ----
            is_success, xy_dist, z_disp = check_success(
                peg_base, target_peg_base, xy_centering, hole_height, success_threshold)
            if not logical_done and not engaged:
                is_engaged, _, _ = check_success(
                    peg_base, target_peg_base, xy_centering, hole_height, engage_threshold)
                if is_engaged:
                    engaged = True

            # ---- Per-step data ALWAYS collected (through max_steps) ----
            if with_step_data:
                step_records.append(flatten_step(
                    step, t_mono, snap, raw_action, ema_actions, target_pos, target_quat,
                    peg_base, xy_dist, z_disp, fmag, in_contact, succeeded, terminated, obs))

            # ---- Terminal events: register the FIRST one (the logical outcome) ----
            if not logical_done:
                if not succeeded and is_success:
                    succeeded = True
                    success_step = step
                    if terminate_on_success:
                        logical_done = True
                        logical_end_step = step
                if not logical_done and not terminated and fmag >= break_force:
                    terminated = True
                    termination_step = step
                    logical_done = True
                    logical_end_step = step

            # Stop at the logical end UNLESS we're continuing to collect data.
            if logical_done and not continue_on_break:
                break
    except Exception as e:
        # Any error DURING the rollout is NOT retried. If it is the first terminal event
        # it counts as a BREAK flagged panda_error (robot/comm fault, not force); if a
        # force-break/success already ended the logical episode, it just stops the
        # data-collection extension and does not change the outcome.
        rp(f"  [ROLLOUT ERROR] step {step}: {type(e).__name__}: {e}  -> BREAK (panda error)")
        if not logical_done:
            terminated = True
            panda_error = True
            logical_end_step = step
        logical_done = True

    try:
        robot.end_control()
    except Exception:
        pass
    if panda_error:
        try:
            robot.error_recovery()   # clear the reflex so the next episode's reset can run
        except Exception:
            pass
    if log_trajectory:
        traj = robot.get_last_trajectory()
        if traj is not None and traj_dir is not None:
            os.makedirs(traj_dir, exist_ok=True)
            tpath = os.path.join(traj_dir, f"traj_{ep_idx:03d}.npz")
            np.savez_compressed(tpath, **traj)
            EvalKeyboardController.raw_print(
                f"    [traj] 1kHz -> {tpath} ({len(traj['time_ms'])} samples)")
    frames = recorder.end_segment() if (with_step_data and recorder is not None) else []

    # Metric (logical) length = where the episode logically ended (first terminal
    # event); data_length = steps actually run/collected (== max_steps when
    # continue_on_break kept the rollout going).
    data_length = step + 1
    length = (logical_end_step + 1) if logical_end_step >= 0 else data_length
    outcome = "SUCCESS" if succeeded and not terminated else "BREAK" if terminated else "TIMEOUT"
    result = {
        "episode": ep_idx, "outcome": outcome,
        "succeeded": succeeded, "engaged": engaged, "terminated": terminated,
        "panda_error": panda_error, "length": length, "data_length": data_length,
        "ssv": ssv_sum / length if length else 0.0,
        "ssjv": ssjv_sum / length if length else 0.0,
        "max_force": max_force, "sum_force": sum_force,
        "sum_force_in_contact": sum_force_in_contact, "contact_steps": contact_steps,
        "energy": energy_sum,
        "success_step": success_step if success_step >= 0 else length,
        "termination_step": termination_step if termination_step >= 0 else length,
    }

    if with_step_data:
        from real_robot_scripts.trial_data import save_trial
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

def build_policy_specs(args):
    """List of {label, method, ckpt, agent} to run.

    --sweep_root <root>: walk <root>/<method>/<agent#>/ -> one spec per (method, agent).
    Otherwise a single spec from --checkpoint/--agent.
    """
    if args.sweep_root:
        root = args.sweep_root
        specs = []
        for method in sorted(os.listdir(root)):
            mdir = os.path.join(root, method)
            if not os.path.isdir(mdir):
                continue
            agents = sorted((x for x in os.listdir(mdir)
                             if x.isdigit() and os.path.isdir(os.path.join(mdir, x))), key=int)
            for a in agents:
                specs.append({"label": f"{method}/a{a}", "method": method,
                              "ckpt": mdir, "agent": int(a)})
        if not specs:
            raise SystemExit(f"No <method>/<agent#>/ dirs found under {root}")
        return specs
    base = os.path.basename(args.checkpoint.rstrip("/"))
    return [{"label": f"{base}/a{args.agent}", "method": base,
             "ckpt": args.checkpoint, "agent": args.agent}]


def load_policy_for_spec(spec, cfg, args, device):
    """Resolve runtime config + load policy + obs builder + mapper + break_force for one spec.

    The eval_config (cfg) is authoritative for everything eval-side; the runtime config
    only supplies model architecture, force-obs mode, and break_force (which the eval
    task.break_force_threshold may override).
    """
    train_cfg, cfg_src, run_id = resolve_runtime_config(
        spec["ckpt"], spec["agent"], wandb_run=args.wandb_run,
        wandb_entity=args.wandb_entity, wandb_project=args.wandb_project)
    policy = ForgePolicy(spec["ckpt"], train_cfg, agent_idx=spec["agent"], step=args.step, device=device)
    break_force = policy.break_force
    override_bf = cfg["task"].get("break_force_threshold")
    if override_bf is not None and abs(float(override_bf) - break_force) > 1e-6:
        print(f"  [warn] eval_config task.break_force_threshold={override_bf} overrides "
              f"training break_force={break_force} N")
        break_force = float(override_bf)
    force_threshold = float(cfg.get("obs", {}).get("force_threshold", FD.FORCE_THRESHOLD_DEFAULT))
    obs_builder = ObservationBuilder(force_threshold=force_threshold,
                                     action_dim=policy.action_dim,
                                     history_length=policy.history_length, device=device)
    obs_builder.validate_against_checkpoint(policy.obs_dim)
    goal = torch.as_tensor(cfg["task"]["fixed_asset_position"], device=device, dtype=torch.float32)
    mapper = ForgeActionMapper(
        goal_position=goal,
        ema_factor=float(cfg.get("control", {}).get("ema_factor", FD.EMA_FACTOR_DEFAULT)),
        pos_action_bounds=FD.POS_ACTION_BOUNDS, rot_action_bounds=FD.ROT_ACTION_BOUNDS,
        pos_action_threshold=FD.POS_ACTION_THRESHOLD, rot_action_threshold=FD.ROT_ACTION_THRESHOLD,
        action_dim=policy.action_dim, device=device)
    return {"policy": policy, "obs_builder": obs_builder, "mapper": mapper,
            "break_force": break_force, "train_cfg": train_cfg, "run_id": run_id}


def run_policy(robot, loaded, keyboard, episode_noises, cfg, ctrl, goal_position, target_peg_base,
               ee_to_peg, contact_force_threshold, hand_init_pos, hand_init_orn, cal_pose,
               retract_height, device, std_scale, recorder, with_step_data, out_dir,
               log_trajectory, traj_dir, label, continue_on_break=False):
    """Run all episodes for one loaded policy. Returns (episode_results, quit_requested)."""
    rp = EvalKeyboardController.raw_print
    policy, obs_builder = loaded["policy"], loaded["obs_builder"]
    mapper, break_force = loaded["mapper"], loaded["break_force"]
    total_episodes = len(episode_noises)

    rp("=" * 80)
    rp(f"POLICY {label}  |  {total_episodes} eps  break_force={break_force} N  step={policy.step}  "
       f"{'deterministic' if std_scale <= 0 else f'std_scale={std_scale}'}")
    rp("  's'=skip(BREAK)  'p'=pause  'c'=calibrate(paused)  Enter=resume  ESC=quit")
    rp("=" * 80)

    episode_results = []
    running_s = running_b = running_bp = running_be = running_t = running_te = 0
    quit_requested = False
    for ep_idx in range(total_episodes):
        # run_episode handles pre-rollout retries (5x) and rollout errors internally.
        # It returns None ONLY if the pre-rollout couldn't recover -> abort this policy;
        # a rollout error comes back as a break with panda_error=True.
        result = run_episode(
            robot, policy, obs_builder, mapper, cfg, ctrl, goal_position,
            target_peg_base, ee_to_peg, break_force, contact_force_threshold,
            episode_noises[ep_idx], hand_init_pos, hand_init_orn, keyboard, device,
            std_scale=std_scale, recorder=recorder,
            with_step_data=with_step_data, out_dir=out_dir, ep_idx=ep_idx,
            log_trajectory=log_trajectory, traj_dir=traj_dir, continue_on_break=continue_on_break)

        if result is None:
            sys.stdout.write("\r\n")
            rp(f"  [ABORT] {label}: pre-rollout failed after retries — skipping rest of this policy")
            break

        if result["succeeded"] and not result["terminated"]:
            running_s += 1
        elif result["terminated"]:
            running_b += 1
            if result.get("panda_error"):
                running_bp += 1
            if result["engaged"]:
                running_be += 1
        else:
            running_t += 1
            if result["engaged"]:
                running_te += 1

        status = (f"  [{label}] [{ep_idx + 1}/{total_episodes}] {result['outcome']}"
                  f"{'(panda)' if result.get('panda_error') else ''} "
                  f"len={result['length']} max_f={result['max_force']:.2f}N | "
                  f"S:{running_s} B:{running_b}(pe:{running_bp},eng:{running_be}) T:{running_t}")
        sys.stdout.write(f"\r\x1b[K{status}\r\n")
        sys.stdout.flush()
        episode_results.append(result)

        if keyboard.should_quit:
            rp("  [QUIT] Shutting down...")
            quit_requested = True
            break

        if keyboard.should_pause:
            keyboard.set_paused(True)
            rp("  [PAUSED] 'c' = calibrate, Enter = resume, ESC = quit")
            while True:
                if keyboard.should_quit:
                    rp("  [QUIT] Shutting down...")
                    keyboard.set_paused(False)
                    quit_requested = True
                    break
                if keyboard.should_calibrate:
                    rp("  [CALIBRATING] Moving to goal XY, 5cm above goal Z...")
                    robot.retract_up(retract_height)
                    robot.reset_to_start_pose(cal_pose)
                    snap = robot.get_state_snapshot()
                    rp(f"  [CALIBRATED] xyz=[{snap.ee_pos[0].item():.4f}, "
                       f"{snap.ee_pos[1].item():.4f}, {snap.ee_pos[2].item():.4f}]")
                    rp("  [PAUSED] 'c' = calibrate, Enter = resume, ESC = quit")
                if keyboard.should_resume:
                    keyboard.set_paused(False)
                    rp("  [RESUMED]")
                    break
                time.sleep(0.05)
            if quit_requested:
                break

    return episode_results, quit_requested


def main():
    p = argparse.ArgumentParser(description="Real-robot FORGE peg-insert evaluation")
    p.add_argument("--checkpoint", default=None, help="Training run dir (contains <agent>/checkpoints)")
    p.add_argument("--sweep_root", default=None,
                   help="Root dir laid out as <root>/<method>/<agent#>/ ; runs every "
                        "method+agent in one keyboard session, each logged as its own wandb sibling")
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
    p.add_argument("--continue_on_break", action="store_true",
                   help="Keep the rollout going to max_steps after a force break (for data "
                        "collection); metrics still freeze at the break. Only a panda error "
                        "or skip ends early. Pair with --with_step_data / --log_trajectory.")
    p.add_argument("--data_dir", default=None, help="Local dir for per-step data/videos")
    p.add_argument("--log_trajectory", action="store_true",
                   help="Record the 1 kHz torque-loop trajectory per episode to .npz (diagnostics)")
    p.add_argument("--trajectory_dir", default=None, help="Dir for 1 kHz trajectory .npz files")
    p.add_argument("--override", action="append", default=[], help="Config override key=value")
    p.add_argument("--wandb_run", default=None, help="Source wandb run id (download runtime_config.yaml)")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()
    if not args.checkpoint and not args.sweep_root:
        p.error("provide --checkpoint <run_dir> or --sweep_root <root>")
    if args.checkpoint and args.sweep_root:
        p.error("use either --checkpoint or --sweep_root, not both")

    torch.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)

    print("=" * 80 + "\nREAL-ROBOT FORGE PEG-INSERT EVALUATION\n" + "=" * 80)
    cfg = load_config(args.config, args.override)
    device = args.device
    use_mock = bool(cfg["robot"].get("use_mock", False))
    interactive = sys.stdin.isatty() and not use_mock
    task = cfg["task"]
    goal_position = torch.as_tensor(task["fixed_asset_position"], device=device, dtype=torch.float32)
    target_peg_base = torch.as_tensor(task["target_peg_base_position"], device=device, dtype=torch.float32)
    ee_to_peg = torch.as_tensor(task["ee_to_peg_base_offset"], device=device, dtype=torch.float32)
    std_scale = args.std_scale if args.std_scale is not None else float(cfg.get("policy", {}).get("std_scale", 0.0))

    # 1. Policies to run (single --checkpoint, or every <method>/<agent#> under --sweep_root).
    specs = build_policy_specs(args)
    print(f"  {len(specs)} policy(ies) to run: " + ", ".join(s["label"] for s in specs))
    ctrl = cfg["control"]
    contact_force_threshold = float(cfg.get("obs", {}).get("contact_force_threshold", 1.5))

    # 2. Robot + optional shared RealSense recorder (per-policy out/traj dirs built in the loop).
    print("\nInitializing robot interface...")
    robot = FrankaInterface(cfg, device=device)
    recorder = None
    if args.with_step_data:
        from real_robot_scripts.camera import make_recorder
        recorder = make_recorder(cfg.get("camera", {}), use_mock=use_mock)
        recorder.start()

    # --- read-state diagnostic: print one live observation (first policy) and exit ---
    if args.read_state:
        loaded = load_policy_for_spec(specs[0], cfg, args, device)
        try:
            robot.start_torque_mode()
            time.sleep(1.0)
            snap = robot.get_state_snapshot()
            robot.end_control()
            obs = loaded["obs_builder"].build_observation(
                snap, goal_position, torch.zeros(loaded["policy"].action_dim, device=device))
            print(f"\n  ee_pos={snap.ee_pos.tolist()}")
            print(f"  force_torque={[round(v, 3) for v in snap.force_torque.tolist()]}")
            print(f"  goal={goal_position.tolist()}")
            print_observation(obs, loaded["obs_builder"])
        finally:
            if recorder is not None:
                recorder.close()
            robot.shutdown()
        return

    print("\nClosing gripper...")
    robot.close_gripper()

    reset_cfg = cfg["reset"]
    hand_init_pos = torch.as_tensor(reset_cfg["hand_init_pos"], device=device, dtype=torch.float32)
    hand_init_pos_noise = torch.as_tensor(reset_cfg["hand_init_pos_noise"], device=device, dtype=torch.float32)
    hand_init_orn = list(reset_cfg["hand_init_orn"])
    retract_height = float(reset_cfg["retract_height_m"])
    noise_cfg = cfg.get("noise", {}) or {}
    goal_pos_noise_scale = torch.as_tensor(noise_cfg.get("goal_pos_noise", [0.0, 0.0, 0.0]),
                                           device=device, dtype=torch.float32)

    # 5. Move to a calibration pose (goal XY, 5cm above goal Z) for eyeballing.
    cal_goal = goal_position.clone()
    cal_goal[2] += 0.05
    cal_pose = make_ee_target_pose(cal_goal.cpu().numpy(), np.array(hand_init_orn))
    print("\nMoving to calibration pose (goal XY, 5cm above goal Z)...")
    robot.retract_up(retract_height)
    robot.reset_to_start_pose(cal_pose)
    snap = robot.get_state_snapshot()
    print(f"  Calibration pose: xyz=[{snap.ee_pos[0].item():.4f}, "
          f"{snap.ee_pos[1].item():.4f}, {snap.ee_pos[2].item():.4f}]")
    if interactive:
        input("  Press Enter to begin experiments...")

    # 6. Pre-generate per-episode noise (deterministic under --eval_seed).
    total_episodes = args.num_episodes
    episode_noises = []
    for _ in range(total_episodes):
        episode_noises.append({
            "goal_pos_noise": torch.randn(3, device=device) * goal_pos_noise_scale,
            "start_pos_noise": (2 * torch.rand(3, device=device) - 1) * hand_init_pos_noise,
        })

    # 7. Keyboard-controlled sweep over all policies (one session; auto-advances).
    keyboard = EvalKeyboardController()
    keyboard.start()

    outcomes = []  # {spec, loaded, results, out_dir}
    try:
        for spec in specs:
            loaded = load_policy_for_spec(spec, cfg, args, device)
            out_dir = traj_dir = None
            if args.with_step_data or args.log_trajectory:
                data_root = args.data_dir or cfg.get("data_dir", "data/real_robot_eval")
                # Nested layout: <data_root>/<method>/<agent>/  (mirrors --sweep_root).
                base = os.path.join(data_root, spec["method"], str(spec["agent"]))
                if args.with_step_data:
                    out_dir = base
                if args.log_trajectory:
                    traj_dir = args.trajectory_dir or os.path.join(base, "trajectories")
            results, quit_req = run_policy(
                robot, loaded, keyboard, episode_noises, cfg, ctrl, goal_position,
                target_peg_base, ee_to_peg, contact_force_threshold, hand_init_pos,
                hand_init_orn, cal_pose, retract_height, device, std_scale, recorder,
                args.with_step_data, out_dir, args.log_trajectory, traj_dir, spec["label"],
                continue_on_break=args.continue_on_break)
            outcomes.append({"spec": spec, "loaded": loaded, "results": results, "out_dir": out_dir})
            if quit_req:
                break
    finally:
        keyboard.stop()

    # --- Per-policy summary + wandb (keyboard stopped) + cross-policy table ---
    rows = []
    for oc in outcomes:
        spec, loaded, results, out_dir = oc["spec"], oc["loaded"], oc["results"], oc["out_dir"]
        if not results:
            print(f"\n[{spec['label']}] NO EPISODES COMPLETED")
            continue
        m = compute_metrics(results)
        print("\n" + "=" * 80 + f"\nRESULTS  {spec['label']}\n" + "=" * 80)
        print(f"  Successes: {m['num_successful_completions']}/{m['total_episodes']}")
        print(f"  Breaks:    {m['num_breaks']}/{m['total_episodes']} "
              f"({m['num_breaks_panda_error']} panda-error, {m['num_breaks_engaged']} engaged)")
        print(f"  Timeouts:  {m['num_failed_timeouts']}/{m['total_episodes']} ({m['num_timeouts_engaged']} engaged)")
        print(f"  Avg length: {m['episode_length']:.1f}   SSV: {m['ssv']:.4f}   "
              f"Avg force: {m['avg_force']:.2f} N   Max force: {m['max_force']:.2f} N   Energy: {m['energy']:.2f}")
        rows.append((spec, m))

        if args.with_step_data and out_dir:
            from real_robot_scripts.trial_data import save_summary
            save_summary(results, out_dir)

        if not args.no_wandb:
            import wandb
            wcfg = cfg.get("wandb", {}) or {}
            extra = list(wcfg.get("extra_tags") or []) + [f"method:{spec['method']}"]
            target = derive_wandb_target(loaded["train_cfg"], spec["agent"], extra_tags=extra,
                                         project_override=wcfg.get("project"),
                                         entity_override=wcfg.get("entity"))
            run = wandb.init(
                project=target["project"], entity=target["entity"] or None,
                group=target["group"], tags=target["tags"], name=target["name"],
                reinit="create_new",
                config={"method": spec["method"], "agent": spec["agent"], "checkpoint": spec["ckpt"],
                        "step": loaded["policy"].step, "source_run_id": loaded["run_id"],
                        "num_episodes": len(results), "std_scale": std_scale,
                        "break_force": loaded["break_force"]})
            # Log through the run handle (reinit="create_new" does NOT set the
            # module-global wandb.run, so wandb.log()/wandb.finish() have no target).
            run.log({**{f"Eval_Core/{k}": v for k, v in m.items()}, "total_steps": loaded["policy"].step})
            run.finish()
            print(f"  wandb: {run.url}")

    if len(rows) > 1:
        print("\n" + "=" * 80 + "\nSUMMARY — ALL POLICIES\n" + "=" * 80)
        print(f"  {'policy':<26} {'N':>3} {'succ':>5} {'brk':>4} {'brkPE':>5} {'to':>4} {'maxF':>6} {'avgF':>6}")
        for spec, m in rows:
            print(f"  {spec['label']:<26} {m['total_episodes']:>3} "
                  f"{m['num_successful_completions']:>5} {m['num_breaks']:>4} {m['num_breaks_panda_error']:>5} "
                  f"{m['num_failed_timeouts']:>4} {m['max_force']:>6.2f} {m['avg_force']:>6.2f}")

    if recorder is not None:
        recorder.close()
    robot.shutdown()
    print("\n" + "=" * 80 + "\nEVALUATION COMPLETE\n" + "=" * 80)


if __name__ == "__main__":
    main()
