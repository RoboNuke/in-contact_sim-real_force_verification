"""Config for the contact-force test env (``Isaac-ContactForceTest-Direct-v0``).

This is a **non-learning** test rig built on top of Isaac Lab's FORGE/Factory
direct env. A cylinder (held, in the Franka's grasp) is pushed straight down
into a fixed box to generate a contact force. It runs at the same physics /
policy rate as FORGE (``sim.dt = 1/120``, ``decimation = 8`` → 15 Hz policy).

Everything that defines the physics, the controller, and the geometry is
exposed here as a config field so it can be changed without touching code
(directly, or via ``runner_cfg.env_cfg_overrides`` dotted paths).

Only the **position** action space is implemented for now (a task-space PD
controller on the EE z position); the force-control action space is deferred.
"""

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Reuse Factory's scene / robot / sim / physx config and the task dataclasses so
# we inherit FORGE-parity physics and the grasp-reset machinery.
from isaaclab_tasks.direct.factory.factory_env_cfg import CtrlCfg, FactoryEnvCfg
from isaaclab_tasks.direct.factory.factory_tasks_cfg import (
    FactoryTask,
    FixedAssetCfg,
    HeldAssetCfg,
)


@configclass
class ContactTestCtrlCfg(CtrlCfg):
    """Task-space PD controller settings.

    The z proportional gain is supplied *every step* by ``action[:, 1]``; its
    derivative gain is the critically-damped value ``2*sqrt(kp_z)`` (see
    ``factory_utils.get_deriv_gains``). The remaining axes (x, y and the three
    rotational DOFs) are held at ``default_task_prop_gains`` below.
    """

    # FORGE-parity default gains for the axes NOT driven by the action.
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]

    # No action smoothing — the test wants the commanded action applied directly.
    ema_factor = 1.0

    # z position action shaping: target tip height = reference + clip(scale*a_z, ±bound).
    z_action_scale: float = 1.0  # meters of tip-target per unit action
    z_action_bound: float = 0.05  # max |tip target offset| (m)
    # Reference for the z target:
    #   True (absolute, default): the box TOP SURFACE. a_z=0 => tip on the surface;
    #        a_z is the tip height RELATIVE TO THE CONTACT SURFACE (negative = press
    #        into it => contact force ~ kp * penetration). A fixed absolute setpoint
    #        => position feedback => settles to static force equilibrium.
    #   False (delta): current fingertip z; error is always Δz (constant-force
    #        command, no position stiffness; the FORGE/Factory convention).
    z_target_absolute: bool = True

    # Source of the joint torque used for the pinv(J^T)·tau EE-force estimate:
    #   "measured"  -> get_dof_projected_joint_forces(): the physics solver's
    #                  joint force in the DOF motion direction (the actual
    #                  reaction, incl. external contact). Real-robot analog.
    #   "commanded" -> robot.data.applied_torque: the applied actuator effort.
    joint_torque_source: str = "measured"


@configclass
class CylinderHeldAssetCfg(HeldAssetCfg):
    """The held cylinder (a parametric primitive, spawned in ``_setup_scene``)."""

    diameter: float = 0.008  # cylinder diameter (m); also sets the gripper width
    height: float = 0.05  # cylinder length (m)
    mass: float = 0.019
    friction: float = 0.75


@configclass
class BoxFixedAssetCfg(FixedAssetCfg):
    """The fixed box (a parametric primitive, kinematic so it never moves).

    ``height`` is kept in sync with ``box_size[2] / 2`` in the env ``__init__`` so
    that Factory's "fixed tip" (origin + height + base_height) lands on the box's
    top surface (the box primitive's origin is its geometric center).
    """

    box_size: tuple = (0.04, 0.04, 0.04)  # full x, y, z extents (m)
    height: float = 0.02  # == box_size[2] / 2 (auto-synced in env __init__)
    base_height: float = 0.0
    mass: float = 0.05
    friction: float = 0.75
    # The box is a kinematic rigid body (immovable). See _build_fixed_cfg.


@configclass
class ContactForceTestTask(FactoryTask):
    """Task cfg: cylinder-into-box, all randomization disabled.

    ``name = "peg_insert"`` so Factory's geometry helpers take their simplest
    (zero-offset) branch. The reward/success code that is peg-specific never runs
    because the env overrides ``_get_rewards`` / ``_get_dones``.
    """

    name: str = "peg_insert"
    fixed_asset_cfg: BoxFixedAssetCfg = BoxFixedAssetCfg()
    held_asset_cfg: CylinderHeldAssetCfg = CylinderHeldAssetCfg()

    # Cylinder tip sits this far above the box top surface at reset (m).
    tip_gap: float = 0.003

    # Empirical fingertip-placement residual: with rel_z=0 the grasp settles the
    # cylinder center ~this far below the fingertip frame, so the tip lands
    # ~this much high. Subtracted from hand_init_pos so the tip is exactly
    # tip_gap above the box. Deterministic (no reset randomization).
    grasp_tip_offset: float = 0.00043

    # World pose of the fixed box (origin = box center). Reachable by the Franka.
    fixed_asset_pos: tuple = (0.6, 0.0, 0.05)

    # ---- Deterministic reset: zero ALL randomization. ----
    # hand_init_pos[2] is overwritten in the env __init__ from tip_gap + cyl/2.
    hand_init_pos: list = [0.0, 0.0, 0.028]
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]
    hand_init_orn: list = [3.14159, 0.0, 0.0]  # gripper pointing straight down
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]

    fixed_asset_init_pos_noise: list = [0.0, 0.0, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 0.0

    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0


@configclass
class ContactForceTestEnvCfg(FactoryEnvCfg):
    """Top-level env cfg. Inherits Factory's sim / physx / robot / scene."""

    # action = [a_z, kp_z]. obs/state dims are set in the env __init__.
    action_space: int = 2
    # [contact_force_w(3), est_force_ee_meas(3, pinv), est_force_ee_dyn(3, dyn-consistent)]
    observation_space: int = 9
    state_space: int = 0

    obs_order: list = []
    state_order: list = []

    ctrl: ContactTestCtrlCfg = ContactTestCtrlCfg()
    task: ContactForceTestTask = ContactForceTestTask()

    # Our held/fixed assets are parametric primitives spawned as RigidObjects;
    # Fabric cloning errors on them ("Failed to clone in Fabric"). USD cloning
    # works and is correct for every env count.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)

    # Same episode length as FORGE PegInsert (10 s @ 15 Hz = 150 steps).
    episode_length_s: float = 10.0

    # When True, record per-physics-step data into env.physics_logger (120 Hz,
    # i.e. every substep, not every policy step). See envs/physics_logger.py.
    log_physics: bool = False
