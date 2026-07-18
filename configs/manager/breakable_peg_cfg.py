"""Config for the breakable-peg wrapper (:class:`wrappers.breakable_peg.BreakablePegWrapper`).

Kept as a plain dataclass with no gym/torch/Isaac imports so the YAML config
manager can load it without the Sim app running. The runner constructs the
wrapper from this and applies it to the raw FORGE env (below the skrl wrapper).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class BreakablePegCfg:
    """Breakable-peg mechanics for a stock FORGE env."""

    enabled: bool = False
    """Master toggle. When False the runner does not apply the wrapper at all
    (stock, unbreakable FORGE peg). The wrapper itself does not read this flag —
    once constructed it is active, and breakable whenever ``break_force > 0``."""

    break_force: float = 25.0
    """Max contact force (N) the peg survives. Once the smoothed contact-force
    magnitude ``||force_sensor_smooth[:, :3]||`` exceeds this, the peg breaks and
    the episode terminates. Must exceed FORGE's contact-penalty threshold: the
    wrapper validates ``break_force > max(contact_penalty_threshold_range) +
    force_margin`` at construction and raises otherwise. ``0`` / negative ->
    unbreakable (wrapper becomes a pass-through)."""

    peg_break_rew: float = -10.0
    """Reward added to an env on the step it breaks. Negative = penalty."""

    force_margin: float = 1.0
    """Required gap (N) between FORGE's maximum contact-penalty threshold and
    ``break_force``, enforced at construction so the penalty band always precedes
    breaking."""
