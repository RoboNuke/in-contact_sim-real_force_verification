"""Real-robot contact-force push test (FR3).

The hardware analog of the sim sweep (``scripts/param_sweep_run.py`` driving
``Isaac-ContactForceTest-Direct-v0``). You position the peg tip at the surface by
hand; this script then, for each target force and each repeat:

  1. returns to the pose you set (the surface / force=0 reference),
  2. re-tares the F/T estimate in free space (``calibrate_ft_bias``),
  3. pushes straight down in **position control** to ``z = surface_ref_z -
     force/gain`` (gain = the z proportional gain, default 565 N/m). The surface
     is rigid, so the controller settles at ``force = gain * depth``.
  4. holds for ``hold_seconds`` while logging, then retracts to (1) for the next
     rep.

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
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_robot_scripts.pro_robot_interface import FrankaInterface
from real_robot_scripts.hybrid_controller import ControlTargets
from real_robot_scripts.robot_interface import SafetyViolation


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
def run_one_push(robot, exp, force, start_pose_4x4, start_pos, start_quat,
                 start_joint_q, surface_ref_z, n_steps):
    """Return-to-start, re-tare, push to depth, hold n_steps, log 15 Hz + 1 kHz."""
    gain = float(exp["gain"])
    depth = force / gain                            # meters below the surface ref

    # (1) return to the surface/force=0 reference in free space
    robot.reset_to_start_pose(start_pose_4x4)

    # (2) re-tare the F/T estimate in free space at the start pose
    tare_bias = robot.calibrate_ft_bias()

    # (3) position target: same x/y/orientation, z = surface_ref - depth
    target_pos = start_pos.clone()
    target_pos[2] = surface_ref_z - depth
    targets = make_position_targets(target_pos, start_quat, start_joint_q, exp)

    # (4) push + hold, logging at 15 Hz (snapshot) and 1 kHz (built-in buffer)
    robot.start_torque_mode(log_trajectory=True)
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

    print("=" * 70)
    print("  REAL-ROBOT CONTACT-FORCE PUSH TEST")
    print(f"  robot={config['robot']['ip']} mock={config['robot']['use_mock']}")
    print(f"  forces={forces} N   reps={reps}   hold={exp['hold_seconds']}s "
          f"({n_steps} steps @ {rate} Hz)")
    print(f"  gain={exp['gain']}  depths(mm)="
          f"{[round(f / float(exp['gain']) * 1000, 2) for f in forces]}")
    print("=" * 70)

    robot = FrankaInterface(config, device="cpu")
    try:
        if exp.get("close_gripper", True) and not args.no_grip:
            robot.close_gripper()

        # Capture the start pose = the surface / force=0 reference. A brief
        # zero-torque session packs the current state into shared memory.
        robot.start_torque_mode()
        snap0 = robot.get_state_snapshot()
        robot.end_control()
        start_pos = snap0.ee_pos.clone()
        start_quat = snap0.ee_quat.clone()
        start_joint_q = snap0.joint_pos.clone()
        surface_ref_z = float(start_pos[2])
        start_pose_4x4 = pose_to_matrix(start_pos, start_quat)
        print(f"[start] ee_pos={start_pos.tolist()}  surface_ref_z={surface_ref_z:.5f}")

        all_samples, all_bias = [], []
        t0 = time.perf_counter()
        traj_1khz = np.empty((len(forces), reps), dtype=object)
        for fi, force in enumerate(forces):
            per_rep = []
            for r in range(reps):
                print(f"[push] force={force:5.1f} N  rep={r + 1}/{reps} ...", flush=True)
                samples, bias, depth, traj = run_one_push(
                    robot, exp, force, start_pose_4x4, start_pos, start_quat,
                    start_joint_q, surface_ref_z, n_steps)
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
        out["prop_gains"] = np.asarray(exp["prop_gains"], dtype=np.float64)
        out["control_rate_hz"] = np.asarray(rate)
        out["hold_seconds"] = np.asarray(float(exp["hold_seconds"]))
        out["n_steps"] = np.asarray(n_steps)
        out["reps"] = np.asarray(reps)
        out["start_ee_pos"] = start_pos.cpu().numpy()
        out["start_ee_quat"] = start_quat.cpu().numpy()
        out["start_joint_q"] = start_joint_q.cpu().numpy()
        out["surface_ref_z"] = np.asarray(surface_ref_z)
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
