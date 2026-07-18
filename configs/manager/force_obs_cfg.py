"""Config for the FORGE force-observation wrapper (:class:`wrappers.force_obs_wrapper.ForceObsWrapper`).

Kept as a plain dataclass with no Isaac imports so the YAML config manager can load
it without the Sim app running. The runner constructs the wrapper from this and
applies it to the raw FORGE env (below the skrl wrapper).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ForceObsCfg:
    """Controls force-source injection and force history on a stock FORGE env."""

    enabled: bool = False
    """Master toggle. When False the runner does not apply the wrapper at all
    (FORGE's stock ``ft_force`` observation is used unchanged)."""

    force_source: str = "contact"
    """Which force feeds the ``ft_force`` observation term:
      ``"contact"``  -> FORGE's force_sensor reading, unchanged.
      ``"dyn_pinv"`` -> inject F = (J M^-1 J^T)^-1 J M^-1 . tau_measured (world
                        frame, raw / no EMA smoothing) in place of ``ft_force``.
    Observation-only: FORGE's contact_penalty reward is never affected."""

    history_enabled: bool = False
    """Append the last ``history_length`` force vectors (newest-first) to the
    policy and critic observations."""

    history_length: int = 0
    """Number of past force vectors to stack when ``history_enabled``. Grows each
    observation by ``history_length * 3``."""
