"""FORGE peg/hole contact-force test env (``Isaac-ForgeContactTest-Direct-v0``).

Subclasses :class:`ContactForceTestEnv` — inheriting ALL of its control modes
(position/force, absolute/delta, force feedback, wrench clamp) and the physics
logger — and only swaps in the FORGE peg (held) and hole (fixed) USD assets, and
places the peg OFF-CENTER so its tip lands on the hole's flat top.
"""

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from .contact_force_test_env import ContactForceTestEnv
from .forge_contact_test_env_cfg import ForgeContactTestEnvCfg
from .physics_logger import PhysicsDataLogger

# Diagnostic: print the peg/hole geometry (placement + tip gap) once per reset.
# Flip to True to inspect off-center / tip-gap calibration.
_DEBUG_GEOM = False

# Prim paths (relative to each env) of the peg/hole *collision bodies*. Unlike the
# cylinder/box (RigidObjects, whose prim IS the body), the FORGE assets are
# articulations, so the collision body is nested one level under HeldAsset/FixedAsset.
# The contact sensor MUST target these bodies, not the articulation roots (pointing
# at the root — which has no collision geometry — breaks articulation init).
_PEG_BODY = "forge_round_peg_8mm"
_HOLE_BODY = "forge_hole_8mm"


class ForgeContactTestEnv(ContactForceTestEnv):
    cfg: ForgeContactTestEnvCfg

    def __init__(self, cfg: ForgeContactTestEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = 9
        cfg.state_space = 0
        self.cfg_task = cfg.task

        # NOTE: the FORGE hole USD is already a FIXED-BASE articulation (it has an
        # internal fixed joint to the world), so it is immovable by construction.
        # Do NOT set kinematic_enabled — that conflicts with the fixed joint
        # ("cannot create a joint between static bodies").

        # The peg USD origin is at its base (== the insertion tip / bottom), and
        # Factory grips it near the top: fingertip - held_pos = height - fingerpad.
        # So to put the peg tip tip_gap above the hole flat top:
        #   fingertip = flat_top + tip_gap + (height - fingerpad)
        # and hand_init_pos[2] (relative to the flat top) is that minus the height/2 = ...
        fingerpad = cfg.task.robot_cfg.franka_fingerpad_length
        cfg.task.hand_init_pos = [
            0.0,
            0.0,
            cfg.task.tip_gap + cfg.task.held_asset_cfg.height - fingerpad - cfg.task.grasp_tip_offset,
        ]

        # Skip ContactForceTestEnv.__init__ (box/cylinder-specific) and
        # FactoryEnv.__init__ (obs recompute); replicate what we need.
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self.physics_logger = PhysicsDataLogger() if cfg.log_physics else None

    # ------------------------------------------------------------------
    # Scene: robot + FORGE peg/hole USD assets + contact sensor
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
        self._held_asset = Articulation(self.cfg_task.held_asset)  # FORGE peg
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)  # FORGE hole

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["held_asset"] = self._held_asset
        self.scene.articulations["fixed_asset"] = self._fixed_asset

        self._contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/HeldAsset/{_PEG_BODY}",
                filter_prim_paths_expr=[f"/World/envs/env_.*/FixedAsset/{_HOLE_BODY}"],
                update_period=0.0,
                history_length=0,
            )
        )
        self.scene.sensors["held_fixed_contact"] = self._contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _set_default_dynamics_parameters(self):
        """Gains/thresholds + friction (Factory's, since peg/hole are Articulations)."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        factory_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _read_contact_force(self) -> torch.Tensor:
        """Net contact force the peg exerts on the hole (world frame).

        The base env reads the *filtered* peg<->hole ``force_matrix_w``. That works
        for the cylinder/box (RigidObjects) but is empty here: the hole is a
        FIXED-BASE articulation, so PhysX attributes the peg's contact to the
        static/world partner rather than the ``forge_hole_8mm`` link, and the
        filter never matches. The peg's *net* contact force is correct instead —
        the hole is the only body it touches that registers a force (the symmetric
        finger grip cancels, so net force is ~0 until the peg hits the hole), so
        the net equals the peg<->hole force. Negated to match the base env's
        "positive = pressing into the surface" convention.
        """
        net = self._contact_sensor.data.net_forces_w  # (N, num_bodies, 3): force ON the peg
        if net is None:
            return torch.zeros((self.num_envs, 3), device=self.device)
        return -net.sum(dim=1)

    # Use Factory's peg grasp transform (revert ContactForceTestEnv's center-origin
    # cylinder override): peg USD origin is at its base.
    def get_handheld_asset_relative_pose(self):
        return FactoryEnv.get_handheld_asset_relative_pose(self)

    # ------------------------------------------------------------------
    # Reset: Factory grasp -> shift off-center onto the flat top -> capture refs
    # ------------------------------------------------------------------
    def set_pos_inverse_kinematics(self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids):
        """Offset the grasp IK target by ``off_center`` in x, so the peg is grasped
        directly over the hole's flat top (not the opening). This is the ONLY
        caller of set_pos_inverse_kinematics (Factory's grasp loop), so the peg
        lands off-center by construction — no post-grasp shifting.
        """
        target = ctrl_target_fingertip_midpoint_pos.clone()
        target[:, 0] += self.cfg_task.off_center
        return FactoryEnv.set_pos_inverse_kinematics(self, target, ctrl_target_fingertip_midpoint_quat, env_ids)

    def _reset_idx(self, env_ids):
        FactoryEnv._reset_idx(self, env_ids)  # grasp the peg off-center over the flat top

        self.init_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.init_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Surface-referenced setpoint: fingertip z that puts the peg tip on the
        # hole's flat top. flat_top_z = fixed_pos + height + base_height (Factory
        # "fixed tip"); peg tip z from _peg_tip_z().
        flat_top_z = (
            self.fixed_pos[:, 2] + self.cfg_task.fixed_asset_cfg.height + self.cfg_task.fixed_asset_cfg.base_height
        )
        tip_z = self._peg_tip_z()
        self.surface_fingertip_ref = flat_top_z + (self.fingertip_midpoint_pos[:, 2] - tip_z)

        if _DEBUG_GEOM:
            print(
                f"[forge-geom] fingertip_z={self.fingertip_midpoint_pos[:, 2].mean().item():.5f} "
                f"held_pos_z={self.held_pos[:, 2].mean().item():.5f} "
                f"held_pos_x={self.held_pos[:, 0].mean().item():.5f} "
                f"fixed_pos_x={self.fixed_pos[:, 0].mean().item():.5f} "
                f"flat_top_z={flat_top_z.mean().item():.5f} tip_z={tip_z.mean().item():.5f} "
                f"tip-above-flat(mm)={((tip_z - flat_top_z) * 1000).mean().item():.3f}",
                flush=True,
            )

    def _peg_tip_z(self):
        """World z of the peg tip (bottom / pressing end).

        The peg USD origin is at its base (the insertion tip), i.e. held_pos IS the
        tip — confirmed by the [forge-geom] diagnostic (get_held_base_pos_local is
        0 for peg_insert).
        """
        return self.held_pos[:, 2]
