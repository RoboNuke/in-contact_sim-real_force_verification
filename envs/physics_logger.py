"""Physics-rate data logger.

Collects per-**physics-step** (not per-policy-step) tensors with timestamps and
dumps them to a compressed ``.npz``. The contact-force test env records into this
from ``_apply_action``, which runs once per physics substep (``decimation`` times
per policy step). Records are moved to the CPU as they arrive so GPU memory does
not grow during a run.

Each recorded field is a per-env tensor of shape ``(num_envs, ...)``; after a run
``as_dict()`` / ``save()`` stack them along a leading time axis, giving
``(T, num_envs, ...)`` arrays plus a ``t`` vector of shape ``(T,)``.
"""

from __future__ import annotations

import numpy as np
import torch


class PhysicsDataLogger:
    """Append-only, physics-rate buffer of timestamped per-env tensors."""

    def __init__(self) -> None:
        self._t: list[float] = []
        self._fields: dict[str, list[torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self._t)

    def record(self, timestamp: float, **fields: torch.Tensor) -> None:
        """Append one physics step. ``fields`` are per-env tensors (num_envs, ...)."""
        self._t.append(float(timestamp))
        for name, value in fields.items():
            self._fields.setdefault(name, []).append(value.detach().to("cpu", copy=True))

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Stack the buffer into ``{field: (T, num_envs, ...)}`` plus ``t: (T,)``."""
        out: dict[str, torch.Tensor] = {"t": torch.tensor(self._t)}
        for name, seq in self._fields.items():
            out[name] = torch.stack(seq, dim=0)
        return out

    def save(self, path: str) -> None:
        """Write the buffer to a compressed ``.npz`` (numpy arrays)."""
        data = self.as_dict()
        np.savez_compressed(path, **{k: v.numpy() for k, v in data.items()})

    def clear(self) -> None:
        self._t.clear()
        self._fields.clear()
