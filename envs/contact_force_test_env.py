"""Contact-force test env (``Isaac-ContactForceTest-Direct-v0``).

A non-learning rig that pushes a grasped cylinder straight down into a fixed box
and reports the resulting contact force two ways:

  1. **Contact sensor** between the held cylinder and the fixed box (full x/y/z
     force, world frame).
  2. **Joint-torque estimate**: ``pinv(J^T) · tau`` mapped into the end-effector
     frame — what a real robot (no wrist F/T sensor) would compute.

It subclasses :class:`FactoryEnv` to reuse the task-space PD control law, the
IK-based grasp reset, and the Jacobian / joint-torque intermediates. The action
is ``[delta_z_target, kp_z]`` (see :meth:`_apply_action`); x/y and orientation
targets are frozen at their reset values, so the cylinder only moves up/down.
"""

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.direct.factory import factory_control, factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from . import factory_control_utils
from .contact_force_test_env_cfg import ContactForceTestEnvCfg
from .physics_logger import PhysicsDataLogger


class ContactForceTestEnv(FactoryEnv):
    cfg: ContactForceTestEnvCfg

    def __init__(self, cfg: ContactForceTestEnvCfg, render_mode: str | None = None, **kwargs):
        # Our observation is three 3-vector force readings (contact sensor,
        # pinv(J^T)·tau_measured, and dyn-consistent·tau_measured); no asymmetric
        # critic state. Set the spaces directly and bypass FactoryEnv.__init__'s
        # obs_order-based recompute (we don't use the Factory obs layout).
        cfg.observation_space = 9
        cfg.state_space = 0
        self.cfg_task = cfg.task

        # Keep the box "tip" offset consistent with the full box z-extent (the
        # primitive box's origin is its center, so origin -> top = box_size_z/2).
        box_z = cfg.task.fixed_asset_cfg.box_size[2]
        cfg.task.fixed_asset_cfg.height = box_z / 2.0
        cfg.task.fixed_asset_cfg.base_height = 0.0

        # Grip the cylinder at its center (get_handheld_asset_relative_pose returns
        # rel_z=0), so the tip hangs height/2 below the fingertip. Place the
        # fingertip at box_top + tip_gap + height/2, minus a small empirical
        # grasp residual, so the tip lands exactly tip_gap above the box.
        cyl_h = cfg.task.held_asset_cfg.height
        cfg.task.hand_init_pos = [0.0, 0.0, cfg.task.tip_gap + cyl_h / 2.0 - cfg.task.grasp_tip_offset]

        # Skip FactoryEnv.__init__ (which recomputes obs/state from obs_order) and
        # replicate the parts we need.
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)

        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

        # Optional per-physics-step data logger (records inside _apply_action).
        self.physics_logger = PhysicsDataLogger() if cfg.log_physics else None

    # ------------------------------------------------------------------
    # Scene: robot + parametric cylinder/box primitives + contact sensor
    # ------------------------------------------------------------------
    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table", table_cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._held_asset = RigidObject(self._build_held_cfg())
        self._fixed_asset = RigidObject(self._build_fixed_cfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["held_asset"] = self._held_asset
        self.scene.rigid_objects["fixed_asset"] = self._fixed_asset

        # Contact sensor on the held cylinder, filtered to the fixed box, so
        # data.force_matrix_w reports the pairwise held<->fixed force.
        self._contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/HeldAsset",
                filter_prim_paths_expr=["/World/envs/env_.*/FixedAsset"],
                update_period=0.0,
                history_length=0,
            )
        )
        self.scene.sensors["held_fixed_contact"] = self._contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _set_default_dynamics_parameters(self):
        """Set gains / thresholds and robot friction.

        Overrides Factory's version, which also calls ``set_friction`` on the
        held and fixed assets via ``root_physx_view.get_material_properties()`` —
        an ArticulationView API our ``RigidObject`` primitives don't share. Their
        friction is already baked into the spawn's ``physics_material``, so we set
        only the robot's here.
        """
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        factory_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _build_held_cfg(self) -> RigidObjectCfg:
        held = self.cfg_task.held_asset_cfg
        return RigidObjectCfg(
            prim_path="/World/envs/env_.*/HeldAsset",
            spawn=sim_utils.CylinderCfg(
                radius=held.diameter / 2.0,
                height=held.height,
                axis="Z",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=held.max_depenetration_velocity,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=held.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=held.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=held.friction, dynamic_friction=held.friction, restitution=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
        )

    def _build_fixed_cfg(self) -> RigidObjectCfg:
        fixed = self.cfg_task.fixed_asset_cfg
        return RigidObjectCfg(
            prim_path="/World/envs/env_.*/FixedAsset",
            spawn=sim_utils.CuboidCfg(
                size=tuple(fixed.box_size),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    max_depenetration_velocity=fixed.max_depenetration_velocity,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=fixed.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=fixed.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=fixed.friction, dynamic_friction=fixed.friction, restitution=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(self.cfg_task.fixed_asset_pos), rot=(1.0, 0.0, 0.0, 0.0)),
        )

    # ------------------------------------------------------------------
    # Tensors
    # ------------------------------------------------------------------
    def _init_tensors(self):
        super()._init_tensors()
        z3 = lambda: torch.zeros((self.num_envs, 3), device=self.device)  # noqa: E731
        # Force readings (populated in _compute_intermediate_values).
        self.contact_force_w = z3()
        self.contact_force_ee = z3()
        self.est_force_w = z3()
        self.est_force_ee = z3()
        self.est_force_w_meas = z3()
        self.est_force_ee_meas = z3()
        self.est_force_w_cmd = z3()
        self.est_force_ee_cmd = z3()
        self.est_force_w_dyn = z3()
        self.est_force_ee_dyn = z3()
        # Commanded control torque (set by generate_ctrl_signals); needed for the
        # "commanded" joint-torque source before the first control step runs.
        self.joint_torque = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        # Frozen x/y/orientation targets, captured at reset.
        self.init_fingertip_pos = z3()
        self.init_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        # Fingertip z at which the tip sits exactly on the box top surface
        # (surface-referenced setpoint for absolute z control). Captured at reset.
        self.surface_fingertip_ref = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Intermediate values: add the two force readings
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)
        self.contact_force_w = self._read_contact_force()
        # Contact force rotated into the EE frame (so all three force readings
        # share a frame in the observation).
        self.contact_force_ee = torch_utils.quat_apply(
            torch_utils.quat_conjugate(self.fingertip_midpoint_quat), self.contact_force_w
        )

        # Two joint-torque signals mapped to an EE force via pinv(J^T)·tau:
        #   measured  = get_dof_projected_joint_forces() — the physics solver's
        #               joint force in the DOF motion direction, i.e. the actual
        #               reaction INCLUDING external contact. This is the
        #               real-robot analog (joint torque sensors) and needs no
        #               null-space reasoning: it is the true joint loading.
        #   commanded = applied actuator effort (robot.data.applied_torque). This
        #               is what the controller asked for; mapping it back returns
        #               the commanded task wrench plus null-space leakage.
        # Full per-joint torques (kept for the physics logger); the arm 7 are used
        # for the EE force estimates.
        self.joint_torque_measured = self._robot.root_physx_view.get_dof_projected_joint_forces()
        self.joint_torque_applied = self._robot.data.applied_torque
        tau_measured = self.joint_torque_measured[:, 0:7]
        tau_commanded = self.joint_torque_applied[:, 0:7]

        self.est_force_w_meas, self.est_force_ee_meas = self._map_torque_to_ee_force(tau_measured)
        self.est_force_w_cmd, self.est_force_ee_cmd = self._map_torque_to_ee_force(tau_commanded)
        # Diagnostic: dynamically-consistent inverse of the MEASURED torque, which
        # annihilates the controller's null-space term exactly (unlike pinv(J^T)).
        self.est_force_w_dyn, self.est_force_ee_dyn = self._map_torque_to_ee_force_dyn(tau_measured)

        # The observation uses the source selected in cfg (default: measured).
        if self.cfg.ctrl.joint_torque_source == "commanded":
            self.est_force_w, self.est_force_ee = self.est_force_w_cmd, self.est_force_ee_cmd
        else:
            self.est_force_w, self.est_force_ee = self.est_force_w_meas, self.est_force_ee_meas

    def _read_contact_force(self) -> torch.Tensor:
        """Net contact force the held cylinder exerts ON the fixed box (world frame).

        ``force_matrix_w`` reports the force ON the held body FROM the fixed body
        (the reaction, box->cylinder). We negate it so its sign matches the
        joint-torque estimates: positive = pressing into the surface.
        """
        fmat = self._contact_sensor.data.force_matrix_w
        if fmat is None:
            return torch.zeros((self.num_envs, 3), device=self.device)
        # (num_envs, num_bodies, num_filters, 3) -> (num_envs, 3); negate to the
        # "force applied to the surface" convention.
        return -fmat.sum(dim=(1, 2))

    def _map_torque_to_ee_force(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a 7-DOF arm joint torque to an EE force: tau = J^T·F => F = pinv(J^T)·tau.

        Returns the linear force in the world frame and rotated into the EE frame.
        """
        jacobian_T = self.fingertip_midpoint_jacobian.transpose(1, 2)  # (N, 7, 6)
        wrench = torch.linalg.pinv(jacobian_T) @ tau.unsqueeze(-1)  # (N, 6, 1)
        force_w = wrench.squeeze(-1)[:, 0:3]
        force_ee = torch_utils.quat_apply(
            torch_utils.quat_conjugate(self.fingertip_midpoint_quat), force_w
        )
        return force_w, force_ee

    def _map_torque_to_ee_force_dyn(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map joint torque to EE force via the dynamically-consistent inverse.

        Jbar^T = (J M^-1 J^T)^-1 J M^-1. Unlike pinv(J^T), this annihilates the
        controller's null-space regularization torque exactly, so it recovers the
        task-space wrench without null-space leakage.
        """
        J = self.fingertip_midpoint_jacobian  # (N, 6, 7)
        M_inv = torch.inverse(self.arm_mass_matrix)  # (N, 7, 7)
        J_Minv = J @ M_inv  # (N, 6, 7)
        lambda_task = torch.inverse(J_Minv @ J.transpose(1, 2))  # (N, 6, 6) = (J M^-1 J^T)^-1
        Jbar_T = lambda_task @ J_Minv  # (N, 6, 7)
        wrench = (Jbar_T @ tau.unsqueeze(-1)).squeeze(-1)  # (N, 6)
        force_w = wrench[:, 0:3]
        force_ee = torch_utils.quat_apply(
            torch_utils.quat_conjugate(self.fingertip_midpoint_quat), force_w
        )
        return force_w, force_ee

    # ------------------------------------------------------------------
    # Grasp geometry override (center-origin cylinder)
    # ------------------------------------------------------------------
    def get_handheld_asset_relative_pose(self):
        """EEF->held-object transform: grasp the cylinder at its center.

        The ``panda_fingertip_centered`` frame is made to coincide with the
        cylinder's origin, which for a primitive ``CylinderCfg`` is its geometric
        center. So the tip (bottom) hangs ``height/2`` below the fingertip, and
        ``hand_init_pos[2] = tip_gap + height/2`` puts the tip ``tip_gap`` above
        the box top.

        NOTE: Factory's peg uses ``rel_z = height - fingerpad`` because that USD's
        origin is at the peg *base* and it grips near the top. Using that (or the
        earlier ``height/2 - fingerpad``) here is inconsistent with a
        center-origin primitive and mis-places the tip by ~height/2.
        """
        rel_pos = torch.zeros((self.num_envs, 3), device=self.device)
        rel_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        return rel_pos, rel_quat

    # ------------------------------------------------------------------
    # Action: [a_z, kp].  a_z is a z position goal (control_mode="position")
    # or a z force target (control_mode="force"); kp is the stiffness / force
    # gain. x/y and orientation are always held at their reset pose.
    # ------------------------------------------------------------------
    def _apply_action(self):
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        goal = self.actions[:, 0]  # z position goal (rel surface) OR z force target
        kp = self.actions[:, 1]  # stiffness / force gain

        if self.cfg.ctrl.control_mode == "force":
            self._apply_force_control(force_target=goal, kp_force=kp)
        else:
            self._apply_position_control(a_z=goal, kp_z=kp)

        # Log this physics substep (state at substep start + the action applied).
        if self.physics_logger is not None:
            self._log_physics_step()

    def _apply_position_control(self, a_z, kp_z):
        """z position PD: target z = z reference + clip(a_z*scale); x/y/orn frozen."""
        dz = torch.clip(
            a_z * self.cfg.ctrl.z_action_scale, -self.cfg.ctrl.z_action_bound, self.cfg.ctrl.z_action_bound
        )
        target_pos = self.init_fingertip_pos.clone()
        z_ref = self.surface_fingertip_ref if self.cfg.ctrl.z_target_absolute else self.fingertip_midpoint_pos[:, 2]
        target_pos[:, 2] = z_ref + dz
        target_quat = self.init_fingertip_quat.clone()

        prop_gains = self.default_gains.clone()
        prop_gains[:, 2] = kp_z
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        if not self.cfg.ctrl.mass_weighting:
            # Plain J^T (default): the IsaacLab compute_dof_torque path.
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                ctrl_target_gripper_dof_pos=0.0,
            )
        else:
            # J^T @ Lambda (task-space inertia weighted): build the same pose-PD
            # wrench compute_dof_torque would, then map it through the local util
            # with mass_weighting on.
            task_wrench = self._pose_wrench(target_pos, target_quat, prop_gains, self.task_deriv_gains)
            self.joint_torque, self.applied_wrench = factory_control_utils.compute_dof_torque_from_wrench(
                cfg=self.cfg,
                dof_pos=self.joint_pos,
                dof_vel=self.joint_vel,
                task_wrench=task_wrench,
                jacobian=self.fingertip_midpoint_jacobian,
                arm_mass_matrix=self.arm_mass_matrix,
                device=self.device,
                mass_weighting=True,
            )
            self.ctrl_target_joint_pos[:, 7:9] = 0.0
            self.joint_torque[:, 7:9] = 0.0
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
            self._robot.set_joint_effort_target(self.joint_torque)

    def _apply_force_control(self, force_target, kp_force):
        """z force control: z task wrench = kp * (f_target - f_measured).

        Pure-proportional force control (the reference hybrid-force-position law).
        x/y and orientation are held at the reset pose by the position PD; only
        the vertical axis is replaced by the force term.
        """
        # (1) Pose-PD wrench holding x/y + reset orientation (world frame, 6D).
        target_pos = self.init_fingertip_pos.clone()
        target_quat = self.init_fingertip_quat.clone()
        prop_gains = self.default_gains.clone()  # x/y/rot position gains
        deriv_gains = factory_utils.get_deriv_gains(prop_gains)
        task_wrench = self._pose_wrench(target_pos, target_quat, prop_gains, deriv_gains)

        # (2) Force wrench along the pressing axis (EE +z), saturated so contact
        # acquisition can't slam/tunnel, then rotated to world.
        f_meas = self._force_feedback()[:, 2]  # EE z, + = pressing
        bound = self.cfg.ctrl.force_wrench_bound
        force_ee = torch.zeros((self.num_envs, 3), device=self.device)
        force_ee[:, 2] = torch.clip(kp_force * (force_target - f_meas), -bound, bound)
        force_world = torch_utils.quat_apply(self.fingertip_midpoint_quat, force_ee)  # (N, 3)

        # (3) Replace the world-vertical linear wrench with the force term; keep
        # x/y and orientation from the pose PD (EE z ~ world z for the upright tool).
        task_wrench[:, 2] = force_world[:, 2]

        self.task_prop_gains = prop_gains
        self.task_deriv_gains = deriv_gains
        self.joint_torque, self.applied_wrench = factory_control_utils.compute_dof_torque_from_wrench(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            task_wrench=task_wrench,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            device=self.device,
            mass_weighting=self.cfg.ctrl.mass_weighting,
        )
        # Keep the gripper closed (position-controlled), like generate_ctrl_signals.
        self.ctrl_target_joint_pos[:, 7:9] = 0.0
        self.joint_torque[:, 7:9] = 0.0
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _pose_wrench(self, target_pos, target_quat, prop_gains, deriv_gains):
        """Task-space PD wrench (world frame, 6D) toward a target pose."""
        pos_error, axis_angle_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_pose = torch.cat((pos_error, axis_angle_error), dim=1)
        return factory_control._apply_task_space_gains(
            delta_pose, self.fingertip_midpoint_linvel, self.fingertip_midpoint_angvel, prop_gains, deriv_gains
        )

    def _force_feedback(self):
        """The force signal (EE frame) that drives the force controller."""
        source = self.cfg.ctrl.force_feedback
        if source == "pinv":
            return self.est_force_ee_meas
        if source == "dyn":
            return self.est_force_ee_dyn
        return self.contact_force_ee

    def _log_physics_step(self):
        """Record one physics-step sample into the data logger (per-env tensors)."""
        self.physics_logger.record(
            self._robot._data._sim_timestamp,
            joint_pos=self.joint_pos,  # (N, 9)
            joint_vel=self.joint_vel,  # (N, 9)
            joint_torque_applied=self.joint_torque_applied,  # (N, 9) actuator effort
            joint_torque_measured=self.joint_torque_measured,  # (N, 9) projected joint force
            mass_matrix=self.arm_mass_matrix,  # (N, 7, 7)
            jacobian=self.fingertip_midpoint_jacobian,  # (N, 6, 7)
            ee_pos=self.fingertip_midpoint_pos,  # (N, 3)
            ee_quat=self.fingertip_midpoint_quat,  # (N, 4)
            ee_linvel=self.fingertip_midpoint_linvel,  # (N, 3)
            ee_angvel=self.fingertip_midpoint_angvel,  # (N, 3)
            contact_force_ee=self.contact_force_ee,  # (N, 3)
            est_force_ee_pinv=self.est_force_ee_meas,  # (N, 3)
            est_force_ee_dynconsistent=self.est_force_ee_dyn,  # (N, 3)
            action_goal=self.actions[:, 0:1],  # (N, 1) z position goal or force target
            action_stiffness=self.actions[:, 1:2],  # (N, 1) stiffness / force gain
        )

    # ------------------------------------------------------------------
    # Observations / rewards / reset
    # ------------------------------------------------------------------
    def _get_observations(self):
        # Three force readings, all in the EE frame (each x/y/z): contact sensor,
        # pinv(J^T)·tau_measured, dyn-consistent·tau_measured.
        obs = torch.cat(
            [self.contact_force_ee, self.est_force_ee_meas, self.est_force_ee_dyn], dim=-1
        )
        return {"policy": obs}

    def _get_rewards(self):
        # Non-learning test — no reward signal.
        self.prev_actions = self.actions.clone()
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Freeze the x/y and orientation targets at the post-grasp pose.
        self.init_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.init_fingertip_quat = self.fingertip_midpoint_quat.clone()
        # Surface-referenced setpoint: the fingertip z that places the tip on the
        # box top. tip = fingertip - (fingertip - tip); box_top = fixed + box/2.
        # So fingertip-for-tip-on-surface = box_top + (fingertip - tip).
        box_top_z = self.fixed_pos[:, 2] + self.cfg_task.fixed_asset_cfg.box_size[2] / 2.0
        tip_z = self.held_pos[:, 2] - self.cfg_task.held_asset_cfg.height / 2.0
        self.surface_fingertip_ref = box_top_z + (self.fingertip_midpoint_pos[:, 2] - tip_z)
