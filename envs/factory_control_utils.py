"""Task-wrench -> joint-torque mapping, split out from FactoryEnv.

``factory_control.compute_dof_torque`` builds the task wrench from a pose error
*internally* and then maps it to joint torques (J^T + null-space posture). For
hybrid position/force control we need to build the wrench ourselves (a blend of a
pose-PD wrench and a force-error wrench), so this exposes just the wrench->torque
half.

This is the second half of ``isaaclab_tasks.direct.factory.factory_control.
compute_dof_torque`` verbatim, matching the reference repo's
``wrappers/controllers/factory_control_utils.compute_dof_torque_from_wrench``.
"""

from __future__ import annotations

import math

import torch


def compute_dof_torque_from_wrench(cfg, dof_pos, dof_vel, task_wrench, jacobian, arm_mass_matrix, device):
    """Map a task-space (world-frame) wrench to arm joint torques.

    ``tau = J^T * wrench`` plus null-space posture control toward the default arm
    configuration; final torques are clamped to ``[-100, 100]``.
    """
    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)

    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Null-space posture control (identical to FactoryEnv).
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    arm_mass_matrix_task = torch.inverse(jacobian @ arm_mass_matrix_inv @ jacobian_T)
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (2 * math.pi) - math.pi
    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - jacobian_T @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench
