"""Scaled-hole FORGE peg-insertion task variants.

These configs reuse the stock ``Isaac-Forge-PegInsert-Direct-v0`` task unchanged
except that the *fixed* asset (the hole/socket) is scaled in-plane so the
peg-to-bore clearance is enlarged to a chosen value. The *held* asset (the peg)
is never touched, so peg geometry, gripper width, grasp pose and mass are
identical across every variant.

Why this works without any other edits
--------------------------------------
* The hole USD is scaled by ``(s, s, 1.0)`` -- X/Y only, Z left at 1.0. Because
  the in-plane factor is uniform (sx == sy) the circular bore stays circular and
  simply grows in diameter: ``bore_diameter = BORE_DIAMETER_NATIVE * s``. Z is
  left untouched so the hole *height* is unchanged.
* ``fixed_asset_cfg.height`` / ``base_height`` (Z quantities) are what the task
  uses for the insertion-depth success threshold and the fixed-tip position, so
  keeping Z at 1.0 keeps the success/reward geometry identical to baseline.
* ``fixed_asset_cfg.diameter`` is metadata only -- grep of the factory/forge code
  shows it is never read (only ``held_asset_cfg.diameter`` drives gripper width).
  We still update it so it faithfully reports the bore size.
* Success is ``xy_dist < 2.5 mm`` (a fixed lateral tolerance, not tied to bore
  size) AND inserted-depth, so enlarging the bore does not change the success
  criterion.

Caveat: the scale is applied to the whole hole USD, so the entire socket (base
plate, mounting-bolt holes, outer boss) grows in-plane with ``s`` -- not just the
bore. For training this is harmless (the socket is a fixed obstacle) but if you
need the outer footprint held constant you would have to author a custom mesh.

Clearance convention
--------------------
``clearance`` here is the **diametral** clearance in metres, i.e.
``bore_diameter - peg_diameter``. Radial gap = clearance / 2. The stock task
ships with a diametral clearance of ``0.0081 - 0.007986 = 0.000114 m`` (0.114 mm).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg

# Native geometry of the stock peg-insert assets (metres). See
# isaaclab_tasks/direct/factory/factory_tasks_cfg.py :: Peg8mm / Hole8mm.
PEG_DIAMETER_NATIVE = 0.007986   # Peg8mm.diameter (held asset -- kept constant)
BORE_DIAMETER_NATIVE = 0.0081    # Hole8mm.diameter (fixed asset -- scaled)
NATIVE_CLEARANCE = BORE_DIAMETER_NATIVE - PEG_DIAMETER_NATIVE  # 0.000114 m


def scale_for_clearance(clearance_m: float) -> float:
    """In-plane USD scale factor that yields a given diametral clearance."""
    return (PEG_DIAMETER_NATIVE + clearance_m) / BORE_DIAMETER_NATIVE


@configclass
class ForgePegInsertScaledHoleCfg(ForgeTaskPegInsertCfg):
    """Base variant: identical to ForgeTaskPegInsertCfg but with an enlarged bore.

    Set ``target_clearance_m`` (diametral, metres) in a subclass. ``__post_init__``
    applies the corresponding in-plane scale to the hole spawn and updates the
    bore-diameter metadata.
    """

    # Diametral clearance target in metres. Default reproduces the stock bore.
    target_clearance_m: float = NATIVE_CLEARANCE

    def __post_init__(self):
        # No parent in the MRO defines __post_init__, but guard anyway so this
        # keeps working if one is ever added upstream.
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()

        s = scale_for_clearance(self.target_clearance_m)
        # Enlarge the bore in-plane only; leave hole height (Z) unchanged.
        self.task.fixed_asset.spawn.scale = (s, s, 1.0)
        # Keep the (metadata-only) bore diameter honest.
        self.task.fixed_asset_cfg.diameter = BORE_DIAMETER_NATIVE * s


@configclass
class ForgePegInsertClear0p5Cfg(ForgePegInsertScaledHoleCfg):
    """0.5 mm diametral clearance."""

    target_clearance_m: float = 0.0005


@configclass
class ForgePegInsertClear1p0Cfg(ForgePegInsertScaledHoleCfg):
    """1.0 mm diametral clearance."""

    target_clearance_m: float = 0.001


@configclass
class ForgePegInsertClear2p0Cfg(ForgePegInsertScaledHoleCfg):
    """2.0 mm diametral clearance."""

    target_clearance_m: float = 0.002


@configclass
class ForgePegInsertClear5p0Cfg(ForgePegInsertScaledHoleCfg):
    """5.0 mm diametral clearance."""

    target_clearance_m: float = 0.005


# Convenient registry: gym id suffix -> (cfg class, diametral clearance in metres).
SCALED_HOLE_VARIANTS = {
    "Clear0p5": (ForgePegInsertClear0p5Cfg, 0.0005),
    "Clear1p0": (ForgePegInsertClear1p0Cfg, 0.001),
    "Clear2p0": (ForgePegInsertClear2p0Cfg, 0.002),
    "Clear5p0": (ForgePegInsertClear5p0Cfg, 0.005),
}
