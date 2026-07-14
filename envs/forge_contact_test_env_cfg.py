"""Config for the FORGE peg/hole contact-force test env.

Same rig as ContactForceTestEnv (all control modes + physics logger), but with the
FORGE peg (held) and hole (fixed) USD assets instead of the parametric cylinder/box.
The peg is placed OFF-CENTER over the hole so its tip lands on the hole's flat top
(not over the opening) — see ForgeContactTestEnv._reset_idx.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import PegInsert

from .contact_force_test_env_cfg import ContactForceTestEnvCfg


@configclass
class ForgeContactTestTask(PegInsert):
    """FORGE peg-into-hole assets, but contact-on-flat-top (deterministic reset).

    Inherits the Peg8mm held-asset and Hole8mm fixed-asset ArticulationCfgs and
    the peg-insert geometry from :class:`PegInsert`.
    """

    # Lateral offset (m) baked into the grasp IK target (ForgeContactTestEnv.
    # set_pos_inverse_kinematics) so the peg is grasped over the hole's flat top,
    # not the opening. To land the peg (radius ~4.0 mm) fully clear of the bore
    # (radius ~4.05 mm) it must exceed ~8.05 mm; the IK undershoots ~9%, so 0.009
    # commanded -> ~8.2 mm actual. Tuned empirically.
    off_center: float = 0.009

    # (The FORGE hole USD is already a fixed-base articulation — immovable by
    # construction — so no kinematic flag is needed, unlike the primitive box.)

    # Peg tip sits this far above the flat top at reset (m).
    tip_gap: float = 0.0002

    # Empirical fingertip-placement residual (like the cylinder's): subtracted from
    # hand_init_pos in the env __init__ so the peg tip lands near tip_gap above the
    # flat top. NOTE: the peg grasp settles with ~0.4 mm run-to-run variance (larger
    # than the cylinder's), so the effective tip gap is ~0.5 mm and cannot be driven
    # reliably to 0.2 mm without risking some envs starting in contact. This value
    # keeps the mean comfortably positive (tip ~0.45-0.6 mm above, small runway).
    grasp_tip_offset: float = 0.0004

    # Deterministic reset: zero ALL randomization.
    fixed_asset_init_pos_noise: list = [0.0, 0.0, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 0.0
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]


@configclass
class ForgeContactTestEnvCfg(ContactForceTestEnvCfg):
    """Top-level cfg: inherits ContactForceTestEnvCfg (ctrl/obs/action) but with
    the FORGE peg/hole task."""

    task: ForgeContactTestTask = ForgeContactTestTask()
