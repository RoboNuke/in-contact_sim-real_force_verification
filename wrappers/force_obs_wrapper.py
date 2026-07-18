"""Force-observation wrapper for the stock FORGE env.

A standalone :class:`gymnasium.Wrapper` around an Isaac Lab FORGE ``DirectRLEnv``
(``ForgeEnv``). It has two independent, config-driven features and depends on
*nothing* from this repo's bespoke envs -- only stock ``ForgeEnv`` attributes.

1. **Force source injection.** FORGE's ``ft_force`` observation term is read from
   the ``force_sensor`` link (contact), EMA-smoothed by ``cfg.ft_smoothing_factor``
   and reframed to world orientation (see ``ForgeEnv._compute_intermediate_values``).
   This wrapper can instead inject a force estimate obtained from the *joint
   torques* via the dynamically-consistent Jacobian inverse::

       F = (J M^-1 J^T)^-1 J M^-1 . tau          (world-frame linear part)

   The estimate overwrites the ``ft_force`` slice in the policy *and* critic
   observation vectors. It is written straight in, with no exponential smoothing --
   this is the "set the EMA to zero" behavior requested for the pinv source (the
   raw estimate replaces the smoothed value each step).

   The injection is **observation-only**: FORGE's contact reward
   (``contact_penalty``, which reads ``force_sensor_smooth``) is deliberately left
   untouched, so swapping the observed force never changes the reward physics.

2. **Force history.** Appends the last ``history_length`` force vectors
   (newest-first, i.e. ``[t-1, t-2, ... t-n]``, each 3-D) to the policy and critic
   observations. The current-step force stays in its usual ``ft_force`` slot, so the
   history is purely additive (no redundancy). Buffers are zeroed per-env in
   ``_reset_idx`` -- FORGE auto-resets envs inside ``step()`` and never calls the
   gym ``reset()`` during training, so ``_reset_idx`` is the only correct hook.

Implementation note: the two runtime hooks (``_reset_idx`` and
``_get_observations``) are installed by rebinding them on ``env.unwrapped``. That
way they still fire when the RL runner drives the base env via ``env.unwrapped``
(bypassing this wrapper object), and the resized observation/state spaces -- which
downstream RL wrappers read off ``env.unwrapped`` -- stay consistent.

Sign/frame caveat: the injected linear force is expressed in world orientation to
match FORGE's ``ft_force`` slot. Its sign/scale need not match the contact-sensor
reading -- comparing the two sources is the point of the toggle.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

# The runner-facing config is a plain, Isaac-free dataclass so YAML loading never
# needs the Sim app; see configs/manager/force_obs_cfg.py.
from configs.manager.force_obs_cfg import ForceObsCfg

# These module-level dicts carry the per-term obs widths. forge_env_cfg mutates
# them in place (adding ``ft_force``: 3) at import, so importing them from the
# forge module guarantees that update has run.
from isaaclab_tasks.direct.forge.forge_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG

_FORCE_TERM = "ft_force"
_FORCE_DIM = 3


class ForceObsWrapper(gym.Wrapper):
    """Inject an alternative force estimate and/or append force history to a
    stock FORGE env's observations. See the module docstring."""

    def __init__(self, env: Any, cfg: ForceObsCfg | None = None) -> None:
        super().__init__(env)
        self.force_cfg = cfg or ForceObsCfg()
        base = env.unwrapped
        self._base = base
        self.device = base.device
        self.num_envs = base.num_envs

        src = self.force_cfg.force_source
        if src not in ("contact", "dyn_pinv"):
            raise ValueError(
                f"force_source must be 'contact' or 'dyn_pinv', got {src!r}"
            )

        # Offset of the 3-wide ft_force slice within each collapsed obs vector.
        self._pol_off = self._force_offset(base.cfg.obs_order, OBS_DIM_CFG)
        self._has_critic = "critic" in base.single_observation_space.spaces
        self._crit_off = (
            self._force_offset(base.cfg.state_order, STATE_DIM_CFG)
            if self._has_critic
            else None
        )

        # History buffers hold the PREVIOUS n forces (newest-first); the current
        # force stays in the ft_force slot, so histories are purely additive.
        n = self.force_cfg.history_length if self.force_cfg.history_enabled else 0
        if n < 0:
            raise ValueError(f"history_length must be >= 0, got {n}")
        self._n = n
        if n > 0:
            self._pol_hist = torch.zeros((self.num_envs, n, _FORCE_DIM), device=self.device)
            if self._has_critic:
                self._crit_hist = torch.zeros((self.num_envs, n, _FORCE_DIM), device=self.device)
            # Resize the base env's spaces in place so downstream RL wrappers
            # (which read env.unwrapped.*) see the extended observations.
            self._grow_spaces(n * _FORCE_DIM)

        # Install runtime hooks on the base env (see module docstring).
        self._orig_get_obs = base._get_observations
        self._orig_reset_idx = base._reset_idx
        base._get_observations = self._get_observations
        base._reset_idx = self._reset_idx

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _force_offset(order: list[str], dims: dict[str, int]) -> int:
        """Cumulative width of the terms preceding ``ft_force`` in ``order``."""
        off = 0
        for name in order:
            if name == _FORCE_TERM:
                return off
            off += dims[name]
        raise ValueError(f"{_FORCE_TERM!r} not found in obs order {order}")

    def _grow_spaces(self, extra: int) -> None:
        """Extend the base env's policy (and critic) obs spaces by ``extra`` dims."""
        b = self._base
        b.cfg.observation_space += extra
        b.single_observation_space["policy"] = self._grow_box(
            b.single_observation_space["policy"], extra
        )
        b.observation_space = gym.vector.utils.batch_space(
            b.single_observation_space["policy"], self.num_envs
        )
        if self._has_critic:
            b.cfg.state_space += extra
            b.single_observation_space["critic"] = self._grow_box(
                b.single_observation_space["critic"], extra
            )
            b.state_space = gym.vector.utils.batch_space(
                b.single_observation_space["critic"], self.num_envs
            )
        # Keep this wrapper's advertised spaces in sync with the base env.
        self.observation_space = b.observation_space
        self.single_observation_space = b.single_observation_space

    @staticmethod
    def _grow_box(box: gym.spaces.Box, extra: int) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(box.shape[0] + extra,), dtype=box.dtype
        )

    # ------------------------------------------------------------------
    # Runtime hooks (rebound onto the base env)
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids) -> None:
        self._orig_reset_idx(env_ids)
        if self._n > 0:
            self._pol_hist[env_ids] = 0.0
            if self._has_critic:
                self._crit_hist[env_ids] = 0.0

    def _get_observations(self) -> dict:
        obs = self._orig_get_obs()
        pol = obs["policy"]
        crit = obs.get("critic") if self._has_critic else None

        # (1) Force source injection: overwrite the ft_force slice (obs-only).
        if self.force_cfg.force_source == "dyn_pinv":
            f = self._dyn_pinv_force()
            pol[:, self._pol_off : self._pol_off + _FORCE_DIM] = f
            if crit is not None:
                crit[:, self._crit_off : self._crit_off + _FORCE_DIM] = f

        # (2) Force history: append previous-n forces, then roll in the current one.
        if self._n > 0:
            cur_pol = pol[:, self._pol_off : self._pol_off + _FORCE_DIM]
            obs["policy"] = torch.cat(
                [pol, self._pol_hist.reshape(self.num_envs, -1)], dim=-1
            )
            self._pol_hist = torch.cat(
                [cur_pol.unsqueeze(1), self._pol_hist[:, :-1]], dim=1
            )
            if crit is not None:
                cur_crit = crit[:, self._crit_off : self._crit_off + _FORCE_DIM]
                obs["critic"] = torch.cat(
                    [crit, self._crit_hist.reshape(self.num_envs, -1)], dim=-1
                )
                self._crit_hist = torch.cat(
                    [cur_crit.unsqueeze(1), self._crit_hist[:, :-1]], dim=1
                )
        return obs

    # ------------------------------------------------------------------
    # Force estimate
    # ------------------------------------------------------------------
    def _dyn_pinv_force(self) -> torch.Tensor:
        """World-frame EE force from measured joint torque via the dynamically-
        consistent Jacobian inverse: Jbar^T = (J M^-1 J^T)^-1 J M^-1; F = Jbar^T tau.

        Uses the projected joint forces (the solver's DOF-direction reaction, incl.
        external contact -- the real-robot joint-torque-sensor analog).
        """
        b = self._base
        tau = b._robot.root_physx_view.get_dof_projected_joint_forces()[:, 0:7]  # (N, 7)
        J = b.fingertip_midpoint_jacobian  # (N, 6, 7)
        M_inv = torch.inverse(b.arm_mass_matrix)  # (N, 7, 7)
        J_Minv = J @ M_inv  # (N, 6, 7)
        lambda_task = torch.inverse(J_Minv @ J.transpose(1, 2))  # (N, 6, 6)
        Jbar_T = lambda_task @ J_Minv  # (N, 6, 7)
        wrench = (Jbar_T @ tau.unsqueeze(-1)).squeeze(-1)  # (N, 6)
        return wrench[:, 0:_FORCE_DIM]  # world-frame linear force
