"""
FORGE action → end-effector pose target, for real-robot evaluation.

Replicates the exact target-pose computation the FORGE env applies to policy
actions, so the same policy drives the real FR3 the way it drove sim:

  * Action EMA smoothing (factory `_pre_physics_step`):
        actions = ema_factor * raw + (1 - ema_factor) * actions
    Both `actions` and `prev_actions` are zeroed at episode reset
    (`factory_env.py` reset), so we seed the EMA state to zero.

  * Position (`forge_env._apply_action`): the action is an INCREMENTAL step from
    the CURRENT fingertip, scaled by pos_action_threshold; the absolute target is
    then clipped to +/- pos_action_bounds around the fixed (hole) frame —
        target = fingertip_pos + action[0:3] * pos_action_threshold
        target = fixed_frame + clip(target - fixed_frame, +/- pos_action_bounds)
    (IsaacLab naming is inverted: pos_action_threshold is the per-step SCALE and
    pos_action_bounds is the CLIP box, not the other way around.)

  * Orientation: roll/pitch actions are zeroed; yaw is remapped into the joint-safe
    band  yaw = -180deg + 270deg * (a+1)/2, composed with a 180deg-about-x flip
    (gripper points down), then the per-axis euler delta is clipped to the rotation
    thresholds about the current orientation.

Pure PyTorch; reuses the vendored quaternion helpers in ``hybrid_controller``.
"""

import numpy as np
import torch

from real_robot_scripts.hybrid_controller import (
    quat_mul,
    quat_from_euler_xyz,
    get_euler_xyz,
)


def _wrap_yaw(angle: torch.Tensor) -> torch.Tensor:
    """factory_utils.wrap_yaw: keep yaw in the joint-safe band (< 235 deg)."""
    return torch.where(angle > np.deg2rad(235.0), angle - 2 * np.pi, angle)


def _s(x, device):
    """0-dim float tensor helper."""
    return torch.as_tensor(x, device=device, dtype=torch.float32).reshape(())


class ForgeActionMapper:
    """Maps EMA-smoothed FORGE actions to (target_pos, target_quat) EE targets.

    Args:
        goal_position: [3] fixed/hole obs-frame position (world). The FORGE
            position action frame origin.
        ema_factor: Fixed action EMA factor for eval (sim randomizes
            ema_factor_range=[0.025, 0.1]; use the midpoint 0.0625).
        pos_action_bounds: [3] position action scale (m). FORGE default [0.05]*3.
        rot_action_bounds: [3] rotation action scale. FORGE default [1, 1, 1].
        pos_action_threshold: [3] per-step position delta clip (m). Default [0.02]*3.
        rot_action_threshold: [3] per-step euler delta clip (rad). Default [0.097]*3.
        action_dim: 7 for FORGE.
        device: Torch device.
    """

    def __init__(
        self,
        goal_position: torch.Tensor,
        ema_factor: float,
        pos_action_bounds,
        rot_action_bounds,
        pos_action_threshold,
        rot_action_threshold,
        action_dim: int = 7,
        device: str = "cpu",
    ):
        self.device = device
        self.action_dim = int(action_dim)
        self.goal_position = goal_position.to(device, dtype=torch.float32)
        self.ema_factor = float(ema_factor)
        self.pos_action_bounds = torch.as_tensor(pos_action_bounds, device=device, dtype=torch.float32)
        self.rot_action_bounds = torch.as_tensor(rot_action_bounds, device=device, dtype=torch.float32)
        self.pos_action_threshold = torch.as_tensor(pos_action_threshold, device=device, dtype=torch.float32)
        self.rot_action_threshold = torch.as_tensor(rot_action_threshold, device=device, dtype=torch.float32)
        self.ema_actions = torch.zeros(self.action_dim, device=device)

    def reset(self):
        """Zero the EMA action state (matches FORGE/factory reset)."""
        self.ema_actions = torch.zeros(self.action_dim, device=self.device)

    def step(self, raw_action: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor):
        """Smooth the raw action and return the clipped EE pose target.

        Args:
            raw_action: [action_dim] policy action in [-1, 1].
            ee_pos: [3] current fingertip position (world).
            ee_quat: [4] current fingertip quaternion (w, x, y, z).

        Returns:
            (target_pos [3], target_quat [4], ema_actions [action_dim]).
        """
        raw = raw_action.to(self.device, dtype=torch.float32)
        self.ema_actions = self.ema_factor * raw + (1.0 - self.ema_factor) * self.ema_actions
        act = self.ema_actions
        ee_pos = ee_pos.to(self.device, dtype=torch.float32)
        ee_quat = ee_quat.to(self.device, dtype=torch.float32)

        # ---- position: FORGE semantics (forge_env._apply_action). act[0:3] is an
        #      ABSOLUTE setpoint relative to the fixed (hole) frame, scaled by
        #      pos_action_bounds; the per-step move from the current fingertip is then
        #      clipped to +/- pos_action_threshold. NOTE: FORGE OVERRIDES Factory's
        #      _apply_action — do not use the Factory (incremental) form here:
        #        pos_actions = actions[:, 0:3] * pos_action_bounds
        #        preclipped  = fixed_frame + pos_actions
        #        target      = fingertip + clip(preclipped - fingertip, +/- pos_action_threshold)
        pos_actions = act[0:3] * self.pos_action_bounds
        preclipped_pos = self.goal_position + pos_actions
        delta_pos = preclipped_pos - ee_pos
        delta_pos = torch.clip(delta_pos, -self.pos_action_threshold, self.pos_action_threshold)
        target_pos = ee_pos + delta_pos

        # ---- orientation: yaw-only, joint-safe remap, then euler delta-clip ----
        yaw = np.deg2rad(-180.0) + np.deg2rad(270.0) * (act[5].item() + 1.0) / 2.0
        bolt_frame_quat = quat_from_euler_xyz(_s(0.0, self.device), _s(0.0, self.device), _s(yaw, self.device))
        quat_bolt_to_ee = quat_from_euler_xyz(_s(np.pi, self.device), _s(0.0, self.device), _s(0.0, self.device))
        preclipped_quat = quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        curr_roll, curr_pitch, curr_yaw = get_euler_xyz(ee_quat)
        des_roll, des_pitch, des_yaw = get_euler_xyz(preclipped_quat)
        curr_roll, curr_pitch = curr_roll.reshape(()), curr_pitch.reshape(())
        des_roll, des_pitch = des_roll.reshape(()), des_pitch.reshape(())

        # yaw
        curr_yaw = _wrap_yaw(curr_yaw.reshape(()))
        des_yaw = _wrap_yaw(des_yaw.reshape(()))
        clipped_yaw = torch.clip(des_yaw - curr_yaw, -self.rot_action_threshold[2], self.rot_action_threshold[2])
        out_yaw = curr_yaw + clipped_yaw

        # roll
        des_roll = torch.where(des_roll < 0.0, des_roll + 2 * np.pi, des_roll)
        clipped_roll = torch.clip(des_roll - curr_roll, -self.rot_action_threshold[0], self.rot_action_threshold[0])
        out_roll = curr_roll + clipped_roll

        # pitch
        curr_pitch = torch.where(curr_pitch > np.pi, curr_pitch - 2 * np.pi, curr_pitch)
        des_pitch = torch.where(des_pitch > np.pi, des_pitch - 2 * np.pi, des_pitch)
        clipped_pitch = torch.clip(des_pitch - curr_pitch, -self.rot_action_threshold[1], self.rot_action_threshold[1])
        out_pitch = curr_pitch + clipped_pitch

        target_quat = quat_from_euler_xyz(out_roll, out_pitch, out_yaw).reshape(4)
        return target_pos, target_quat, self.ema_actions.clone()
