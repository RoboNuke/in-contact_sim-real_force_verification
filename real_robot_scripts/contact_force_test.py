"""Real-robot contact-force push test (FR3).

The hardware analog of the sim sweep (``scripts/param_sweep_run.py`` driving
``Isaac-ContactForceTest-Direct-v0``). The surface reference (the EE/fingertip
position at which the peg tip contacts the surface) comes from ``surface_pos``
in the config, or is captured from a hand-set pose when that is ``null``. The
orientation is leveled exactly upright (tool axis straight down). Then, for each
target force and each repeat, mirroring the sim reset:

  1. centers ``approach_height_m`` (default 2 cm) above the surface point,
  2. re-tares the F/T estimate in that free space (``calibrate_ft_bias``),
  3. lowers to ``tip_gap_m`` above the surface (the sim's ``tip_gap`` standoff),
  4. switches to torque **position control** and holds the standoff pose still
     for ``settle_seconds`` (default 0.5 s, unlogged) so the mode-switch
     transient settles, then
  5. pushes straight down to ``z = surface_ref_z - force/gain`` (gain = the z
     proportional gain, default 565 N/m) for the ``hold_seconds`` (2 s) episode.
     The surface is rigid, so the controller settles at ``force = gain * depth``.
     Then it returns to (1) for the next rep.

Force targets ``[1, 2, 5, 10, 15]`` N x 10 reps = 50 pushes by default. All the
data the sim logs is collected (see the module docstring of
``envs/contact_force_test_env.py``): joint pos/vel, measured joint torques,
mass matrix, Jacobian, EE pose/vel, the F/T reading (real analog of the sim
contact sensor), and the two joint-torque-derived force estimates
``pinv(J^T).tau`` and dynamically-consistent ``Jbar^T.tau``, plus the 1 kHz
trajectory buffer.

Usage (dry run, no hardware)::

    python real_robot_scripts/contact_force_test.py \
        --config real_robot_scripts/config.yaml --mock \
        --forces 1,2 --reps 2 --hold 0.5

On the robot: position the peg, then::

    python real_robot_scripts/contact_force_test.py --config real_robot_scripts/config.yaml
"""

import argparse
import datetime
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_robot_scripts.pro_robot_interface import FrankaInterface
from real_robot_scripts.hybrid_controller import ControlTargets
from real_robot_scripts.robot_interface import (
    SafetyViolation,
    _rotation_matrix_to_quat_wxyz,
)


# --------------------------------------------------------------------------- #
# Quaternion / force-mapping helpers (mirror envs/contact_force_test_env.py so
# the produced fields are directly comparable to the sim log).
# --------------------------------------------------------------------------- #
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a (w,x,y,z) quaternion."""
    return torch.stack([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` (3,) by quaternion ``q`` (w,x,y,z)."""
    w, xyz = q[0], q[1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_to_rot_matrix(q: torch.Tensor) -> np.ndarray:
    """(w,x,y,z) quaternion -> 3x3 rotation matrix (numpy)."""
    w, x, y, z = [float(c) for c in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def level_quat_to_vertical(quat: torch.Tensor):
    """Rotate ``quat`` so the tool/push axis (EE +z) points exactly straight
    down (world -z), preserving the spin about that axis (yaw).

    The peg is pushed straight down in world z, so it should be perfectly
    vertical; this removes any roll/pitch tilt from the hand-set start pose
    (which would otherwise tilt the push off world-z). The minimal (geodesic)
    rotation that aligns the tool axis leaves the twist about that axis
    unchanged, so the finger / x-y heading you set by hand is preserved.

    Returns ``(leveled_quat (w,x,y,z), tilt_removed_deg)``.
    """
    R = torch.from_numpy(quat_to_rot_matrix(quat)).to(quat.dtype)  # world <- EE
    a = R[:, 2] / torch.linalg.norm(R[:, 2])           # tool axis (EE +z) in world
    t = torch.tensor([0.0, 0.0, -1.0], dtype=quat.dtype)  # straight down
    c = float(torch.dot(a, t).clamp(-1.0, 1.0))
    tilt_deg = math.degrees(math.acos(c))
    v = torch.cross(a, t, dim=-1)
    s = float(torch.linalg.norm(v))
    if s < 1e-8:                                        # already vertical (or degenerate)
        return quat.clone(), tilt_deg
    vx = torch.tensor([[0.0, -v[2], v[1]],
                       [v[2], 0.0, -v[0]],
                       [-v[1], v[0], 0.0]], dtype=quat.dtype)
    g = torch.eye(3, dtype=quat.dtype) + vx + vx @ vx * ((1.0 - c) / (s * s))
    return _rotation_matrix_to_quat_wxyz(g @ R), tilt_deg


def map_torque_to_ee_force_pinv(jacobian, tau, ee_quat):
    """F = pinv(J^T) . tau, in world and EE frames. Mirrors the sim's
    ``_map_torque_to_ee_force`` (contact_force_test_env.py)."""
    jacobian_T = jacobian.transpose(0, 1)          # (7, 6)
    wrench = torch.linalg.pinv(jacobian_T) @ tau   # (6,)
    force_w = wrench[0:3]
    force_ee = quat_rotate(quat_conjugate(ee_quat), force_w)
    return force_w, force_ee


def map_torque_to_ee_force_dyn(jacobian, mass_matrix, tau, ee_quat):
    """F via the dynamically-consistent inverse Jbar^T = (J M^-1 J^T)^-1 J M^-1.
    Mirrors the sim's ``_map_torque_to_ee_force_dyn``."""
    J = jacobian                                   # (6, 7)
    M_inv = torch.inverse(mass_matrix)             # (7, 7)
    J_Minv = J @ M_inv                             # (6, 7)
    lambda_task = torch.inverse(J_Minv @ J.transpose(0, 1))  # (6, 6)
    Jbar_T = lambda_task @ J_Minv                  # (6, 7)
    wrench = Jbar_T @ tau                          # (6,)
    force_w = wrench[0:3]
    force_ee = quat_rotate(quat_conjugate(ee_quat), force_w)
    return force_w, force_ee


def deriv_gains(prop_gains: torch.Tensor) -> torch.Tensor:
    """Critical damping 2*sqrt(kp) (== factory_utils.get_deriv_gains)."""
    return 2.0 * torch.sqrt(prop_gains)


# --------------------------------------------------------------------------- #
# ControlTargets builder: pure position hold at a fixed z depth.
# --------------------------------------------------------------------------- #
def make_position_targets(target_pos, target_quat, start_joint_q, exp):
    """Build ControlTargets for a pure position hold (no force axes).

    sel_matrix and force gains are zero, so the wrench is the task-space PD
    toward target_pos/target_quat -- exactly the sim's position controller.
    """
    prop = torch.tensor(exp["prop_gains"], dtype=torch.float32)
    return ControlTargets(
        target_pos=target_pos.clone(),
        target_quat=target_quat.clone(),
        target_force=torch.zeros(6),
        sel_matrix=torch.zeros(6),                 # 0 -> pure position on every axis
        task_prop_gains=prop,
        task_deriv_gains=deriv_gains(prop),
        force_kp=torch.zeros(6),
        force_di_wrench=torch.zeros(6),
        pose_ki=torch.zeros(6),                    # pose integral disabled
        pose_integral_clamp=0.0,
        pose_integral_reset_on_target=True,
        default_dof_pos=start_joint_q.clone(),
        kp_null=float(exp["kp_null"]),
        kd_null=float(exp["kd_null"]),
        pos_bounds=torch.tensor(exp["pos_bounds"], dtype=torch.float32),
        goal_position=target_pos[:3].clone(),
        ctrl_mode="force_only",                    # rotation kept by pose PD; no-op with sel=0
        singularity_damping=0.0,
        partial_inertia_decoupling=False,
        sep_ori=False,
        mass_weighting=bool(exp.get("mass_weighting", True)),
    )


def pose_to_matrix(ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> np.ndarray:
    """Row-major 4x4 homogeneous transform from EE pos + quat (for reset)."""
    T = np.eye(4)
    T[:3, :3] = quat_to_rot_matrix(ee_quat)
    T[:3, 3] = ee_pos.cpu().numpy()
    return T


# --------------------------------------------------------------------------- #
# One push (single force target, single rep).
# --------------------------------------------------------------------------- #
def run_one_push(robot, exp, force, surf_pos, surf_quat, standoff_pos,
                 start_joint_q, approach_pose_4x4, standoff_pose_4x4,
                 n_steps, settle_steps):
    """Stage above the surface, re-tare, lower to the sim standoff, then switch
    to torque control, settle in place, and push to depth for the episode.

    Mirrors the sim reset: each rep centers ``approach_height_m`` above the
    surface point, re-tares the F/T in that free space, lowers to ``tip_gap_m``
    above the surface (the sim's ``tip_gap`` standoff), and only then switches to
    torque control. It first holds the standoff pose still for ``settle_steps``
    (``settle_seconds``) — unlogged — so the mode-switch transient dies out, then
    commands the push and runs the 2 s ``force = gain * depth`` episode. Logs at
    15 Hz (snapshot) and 1 kHz (built-in buffer).
    """
    gain = float(exp["gain"])
    depth = force / gain                            # meters below the surface ref
    surface_ref_z = float(surf_pos[2])

    # (1) center approach_height_m above the surface point (clearance / return pose)
    robot.reset_to_start_pose(approach_pose_4x4)

    # (2) re-tare the F/T estimate in free space up here
    tare_bias = robot.calibrate_ft_bias()

    # (3) lower to the sim standoff: tip sits tip_gap_m above the surface
    robot.reset_to_start_pose(standoff_pose_4x4)

    # (4) switch to torque control and settle: hold the standoff pose still for
    # settle_steps so the mode-switch transient (the shake) dies out. Not logged
    # in the 15 Hz snapshot; the 1 kHz buffer keeps this prefix (see settle_steps
    # in the metadata) so it can be trimmed in analysis.
    hold_targets = make_position_targets(standoff_pos, surf_quat, start_joint_q, exp)
    robot.start_torque_mode(log_trajectory=True)
    for _ in range(settle_steps):
        robot.set_control_targets(hold_targets)
        robot.wait_for_policy_step()
        robot.check_safety(robot.get_state_snapshot())

    # (5) command the push; z target = surface_ref - depth (same x/y/orn) and
    # collect the episode at 15 Hz (snapshot) + 1 kHz (built-in buffer).
    target_pos = surf_pos.clone()
    target_pos[2] = surface_ref_z - depth
    targets = make_position_targets(target_pos, surf_quat, start_joint_q, exp)

    samples = []
    for _ in range(n_steps):
        robot.set_control_targets(targets)
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)

        tau = snap.joint_torques                    # measured tau_J (7,)
        _, est_pinv_ee = map_torque_to_ee_force_pinv(snap.jacobian, tau, snap.ee_quat)
        _, est_dyn_ee = map_torque_to_ee_force_dyn(
            snap.jacobian, snap.mass_matrix, tau, snap.ee_quat)

        samples.append({
            "joint_pos": snap.joint_pos.cpu().numpy(),
            "joint_vel": snap.joint_vel.cpu().numpy(),
            "joint_torques_meas": tau.cpu().numpy(),
            "mass_matrix": snap.mass_matrix.cpu().numpy(),
            "jacobian": snap.jacobian.cpu().numpy(),
            "ee_pos": snap.ee_pos.cpu().numpy(),
            "ee_quat": snap.ee_quat.cpu().numpy(),
            "ee_linvel": snap.ee_linvel.cpu().numpy(),
            "ee_angvel": snap.ee_angvel.cpu().numpy(),
            "ft_ee": snap.force_torque.cpu().numpy(),          # real analog of contact sensor
            "est_force_ee_pinv": est_pinv_ee.cpu().numpy(),
            "est_force_ee_dyn": est_dyn_ee.cpu().numpy(),
            # gravity torque g(q); lets the analysis gravity-remove the est's properly
            # (replaces the step-0 hack). Zeros if the interface didn't capture it.
            "gravity": (snap.gravity.cpu().numpy() if snap.gravity is not None
                        else np.zeros(7, dtype=np.float32)),
        })

    robot.end_control()
    traj_1khz = robot.get_last_trajectory()         # dict of numpy arrays (variable length)

    return samples, np.asarray(tare_bias, dtype=np.float64), depth, traj_1khz


def stack_samples(all_samples):
    """all_samples[f][r] -> list of per-step dicts. Returns {field: (F,R,T,...)}."""
    n_f = len(all_samples)
    n_r = len(all_samples[0])
    fields = list(all_samples[0][0][0].keys())
    out = {}
    for name in fields:
        # (F, R, T, ...) — every push has the same T (= n_steps)
        out[name] = np.stack([
            np.stack([
                np.stack([step[name] for step in all_samples[f][r]], axis=0)
                for r in range(n_r)
            ], axis=0)
            for f in range(n_f)
        ], axis=0)
    return out


def main():
    p = argparse.ArgumentParser(description="Real-robot contact-force push test.")
    p.add_argument("--config", default="real_robot_scripts/config.yaml")
    p.add_argument("--mock", action="store_true", help="Force use_mock=true (dry run).")
    p.add_argument("--forces", default=None, help="Override force list, e.g. '1,2,5'.")
    p.add_argument("--reps", type=int, default=None, help="Override reps per force.")
    p.add_argument("--hold", type=float, default=None, help="Override hold seconds.")
    p.add_argument("--out_dir", default=None, help="Override output directory.")
    p.add_argument("--no_grip", action="store_true", help="Skip close_gripper().")
    p.add_argument("--no_upright", action="store_true",
                   help="Hold the hand-set orientation instead of leveling it upright.")
    args = p.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    if args.mock:
        config["robot"]["use_mock"] = True

    exp = config["experiment"]
    if args.forces is not None:
        exp["forces_n"] = [float(x) for x in args.forces.split(",") if x.strip()]
    if args.reps is not None:
        exp["reps"] = args.reps
    if args.hold is not None:
        exp["hold_seconds"] = args.hold
    out_dir = args.out_dir or exp["out_dir"]

    forces = [float(f) for f in exp["forces_n"]]
    reps = int(exp["reps"])
    rate = float(config["robot"]["control_rate_hz"])
    n_steps = int(round(exp["hold_seconds"] * rate))
    settle_seconds = float(exp.get("settle_seconds", 0.5))
    settle_steps = int(round(settle_seconds * rate))

    print("=" * 70)
    print("  REAL-ROBOT CONTACT-FORCE PUSH TEST")
    print(f"  robot={config['robot']['ip']} mock={config['robot']['use_mock']}")
    print(f"  forces={forces} N   reps={reps}   settle={settle_seconds}s "
          f"({settle_steps} steps)   hold={exp['hold_seconds']}s "
          f"({n_steps} steps @ {rate} Hz)")
    print(f"  gain={exp['gain']}  depths(mm)="
          f"{[round(f / float(exp['gain']) * 1000, 2) for f in forces]}")
    _mw = bool(exp.get("mass_weighting", True))
    print(f"  mass_weighting={_mw} "
          f"({'J^T @ Lambda (inertia-weighted)' if _mw else 'plain J^T (sim-matched)'})")
    print("=" * 70)

    robot = FrankaInterface(config, device="cpu")
    try:
        if exp.get("close_gripper", True) and not args.no_grip:
            robot.close_gripper()

        # Capture the current state. A brief zero-torque session packs the
        # current state into shared memory. Used for the joint config, and for
        # the orientation (and, if surface_pos is not set in config, the
        # surface reference position).
        robot.start_torque_mode()
        snap0 = robot.get_state_snapshot()
        robot.end_control()
        cur_pos = snap0.ee_pos.clone()
        start_quat_raw = snap0.ee_quat.clone()
        start_joint_q = snap0.joint_pos.clone()

        # Force the held peg exactly upright: level the current orientation so
        # the tool axis points straight down (world -z), keeping yaw. Guarantees
        # the push is purely vertical regardless of small tilts in the pose.
        # Disable with force_upright=false / --no_upright.
        if exp.get("force_upright", True) and not args.no_upright:
            start_quat, tilt_deg = level_quat_to_vertical(start_quat_raw)
            warn = ("  *** exceeds warn threshold — re-check the pose ***"
                    if tilt_deg > float(exp.get("upright_warn_deg", 5.0)) else "")
            print(f"[start] leveling to upright: removed {tilt_deg:.2f} deg of tilt"
                  f"{warn}")
        else:
            start_quat = start_quat_raw
            print("[start] force_upright disabled — holding the current orientation")

        # Surface reference position (EE/fingertip pos at which the peg tip
        # contacts the surface). From config if surface_pos is set, else captured
        # from the current (hand-set) pose.
        surf_cfg = exp.get("surface_pos", None)
        if surf_cfg is not None:
            surf_pos = torch.tensor([float(v) for v in surf_cfg], dtype=cur_pos.dtype)
            print(f"[start] surface_pos from config: {surf_pos.tolist()}")
        else:
            surf_pos = cur_pos
            print(f"[start] surface_pos captured (hand-set): {surf_pos.tolist()}")
        surface_ref_z = float(surf_pos[2])

        # Staging heights: center approach_height_m above the surface, then lower
        # to tip_gap_m above it (the sim's tip_gap) before switching to control.
        approach_h = float(exp.get("approach_height_m", 0.02))
        tip_gap = float(exp.get("tip_gap_m", 0.0002))
        approach_pos = surf_pos.clone()
        approach_pos[2] = surface_ref_z + approach_h
        standoff_pos = surf_pos.clone()
        standoff_pos[2] = surface_ref_z + tip_gap
        approach_pose_4x4 = pose_to_matrix(approach_pos, start_quat)
        standoff_pose_4x4 = pose_to_matrix(standoff_pos, start_quat)
        print(f"[start] approach z={float(approach_pos[2]):.5f} (+{approach_h * 100:.1f} cm)  "
              f"standoff z={float(standoff_pos[2]):.5f} (+{tip_gap * 1000:.2f} mm)  "
              f"surface_ref_z={surface_ref_z:.5f}")

        all_samples, all_bias = [], []
        t0 = time.perf_counter()
        traj_1khz = np.empty((len(forces), reps), dtype=object)
        for fi, force in enumerate(forces):
            per_rep = []
            for r in range(reps):
                print(f"[push] force={force:5.1f} N  rep={r + 1}/{reps} ...", flush=True)
                samples, bias, depth, traj = run_one_push(
                    robot, exp, force, surf_pos, start_quat, standoff_pos,
                    start_joint_q, approach_pose_4x4, standoff_pose_4x4,
                    n_steps, settle_steps)
                per_rep.append(samples)
                all_bias.append(bias)
                traj_1khz[fi, r] = traj
                fz = np.mean([s["ft_ee"][2] for s in samples[-max(1, n_steps // 4):]])
                print(f"        settled ft_ee Fz~{fz:+.3f} N  depth={depth * 1000:.2f} mm")
            all_samples.append(per_rep)
        wall = time.perf_counter() - t0

        robot.retract_up(float(exp["retract_height_m"]))

        # ---- assemble + save ----
        stacked = stack_samples(all_samples)
        out = dict(stacked)
        out["traj_1khz"] = traj_1khz                # object array of per-push dicts
        out["force_targets"] = np.asarray(forces, dtype=np.float64)
        out["depths"] = np.asarray([f / float(exp["gain"]) for f in forces], dtype=np.float64)
        out["tare_bias"] = np.asarray(all_bias, dtype=np.float64).reshape(len(forces), reps, 6)
        out["meta_gain"] = np.asarray(float(exp["gain"]))
        out["mass_weighting"] = np.asarray(bool(exp.get("mass_weighting", True)))
        out["prop_gains"] = np.asarray(exp["prop_gains"], dtype=np.float64)
        out["control_rate_hz"] = np.asarray(rate)
        out["hold_seconds"] = np.asarray(float(exp["hold_seconds"]))
        out["n_steps"] = np.asarray(n_steps)
        out["settle_seconds"] = np.asarray(settle_seconds)
        out["settle_steps"] = np.asarray(settle_steps)
        out["reps"] = np.asarray(reps)
        out["surface_pos"] = surf_pos.cpu().numpy()              # surface reference (EE frame)
        out["start_ee_pos"] = surf_pos.cpu().numpy()             # alias (kept for compat)
        out["start_ee_quat"] = start_quat.cpu().numpy()          # leveled (commanded)
        out["start_ee_quat_raw"] = start_quat_raw.cpu().numpy()  # as captured
        out["start_joint_q"] = start_joint_q.cpu().numpy()
        out["surface_ref_z"] = np.asarray(surface_ref_z)
        out["approach_height_m"] = np.asarray(approach_h)
        out["tip_gap_m"] = np.asarray(tip_gap)
        out["wall_time_total"] = np.asarray(wall)
        out["use_mock"] = np.asarray(bool(config["robot"]["use_mock"]))

        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(out_dir, f"contact_force_real_{stamp}.npz")
        np.savez_compressed(fpath, **out)
        print(f"\n[done] saved {fpath}  ({len(forces)} forces x {reps} reps, "
              f"wall {wall:.1f}s)")

    except (SafetyViolation, KeyboardInterrupt) as e:
        print(f"\n[abort] {type(e).__name__}: {e}. Robot stops on comm timeout.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
