"""Joint-torque observation wrapper for the stock FORGE env.

A standalone :class:`gymnasium.Wrapper` around an Isaac Lab FORGE ``DirectRLEnv``
that *appends* the Franka arm joint torques to the observation, without touching
IsaacLab. It implements the "proprio_jnts" ablation: proprioception (no wrist
force) plus the joint torques the robot already reports.

The base FORGE env never exposes joint torques to the policy — ``obs_order`` can
only reference terms present in ``factory``'s ``obs_dict``, and joint torque is
not one of them. Rather than add the term inside IsaacLab, this wrapper reads the
torque straight off the base env each step and concatenates it to the collapsed
observation vector, growing the advertised obs/state spaces to match.

Two torque sources (see :class:`JointObsCfg`):
  * ``"measured"``  — ``root_physx_view.get_dof_projected_joint_forces()[:, :n]``,
    the solver's DOF-direction reaction including external contact. This is the
    real-robot joint-torque-sensor analog (the same signal the dyn_pinv force
    estimate in :mod:`wrappers.force_obs_wrapper` uses).
  * ``"commanded"`` — ``env.joint_torque[:, :n]``, the impedance controller's
    commanded arm torque. Zeros until the first control step has run.

Implementation mirrors :class:`wrappers.force_obs_wrapper.ForceObsWrapper`: the
``_get_observations`` hook is rebound on ``env.unwrapped`` so it still fires when
the RL runner drives the base env via ``env.unwrapped``, and the resized spaces —
read off ``env.unwrapped`` by downstream RL wrappers — stay consistent. Rebinding
captures whatever ``_get_observations`` is currently installed, so this composes
with ForceObsWrapper when both are applied (apply this one second).

The append is purely additive and observation-only: no reward term reads it, and
no existing obs slot is modified.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

# Runner-facing config is a plain, Isaac-free dataclass so YAML loading never
# needs the Sim app; see configs/manager/joint_obs_cfg.py.
from configs.manager.joint_obs_cfg import JointObsCfg


class JointObsWrapper(gym.Wrapper):
    """Append arm joint torques to a stock FORGE env's observations. See the
    module docstring."""

    def __init__(self, env: Any, cfg: JointObsCfg | None = None) -> None:
        super().__init__(env)
        self.joint_cfg = cfg or JointObsCfg()
        base = env.unwrapped
        self._base = base
        self.device = base.device
        self.num_envs = base.num_envs

        src = self.joint_cfg.source
        if src not in ("measured", "commanded"):
            raise ValueError(
                f"joint torque source must be 'measured' or 'commanded', got {src!r}"
            )

        self._nj = int(self.joint_cfg.num_joints)
        if self._nj <= 0:
            raise ValueError(f"num_joints must be >= 1, got {self._nj}")

        self._has_critic = "critic" in base.single_observation_space.spaces
        self._include_critic = bool(self.joint_cfg.include_critic) and self._has_critic

        # Grow the base env's spaces in place so downstream RL wrappers (which read
        # env.unwrapped.*) see the extended observations.
        self._grow_spaces(self._nj, self._include_critic)

        # Install the runtime hook on the base env, chaining any already-installed
        # _get_observations (e.g. ForceObsWrapper's) rather than the original.
        self._orig_get_obs = base._get_observations
        base._get_observations = self._get_observations

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _grow_spaces(self, extra: int, grow_critic: bool) -> None:
        b = self._base
        b.cfg.observation_space += extra
        b.single_observation_space["policy"] = self._grow_box(
            b.single_observation_space["policy"], extra
        )
        b.observation_space = gym.vector.utils.batch_space(
            b.single_observation_space["policy"], self.num_envs
        )
        if grow_critic:
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
    # Runtime hook (rebound onto the base env)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        obs = self._orig_get_obs()
        tau = self._joint_torque()
        obs["policy"] = torch.cat([obs["policy"], tau], dim=-1)
        if self._include_critic and "critic" in obs:
            obs["critic"] = torch.cat([obs["critic"], tau], dim=-1)
        return obs

    def _joint_torque(self) -> torch.Tensor:
        """The (N, num_joints) arm joint-torque tensor for the configured source."""
        b = self._base
        if self.joint_cfg.source == "measured":
            return b._robot.root_physx_view.get_dof_projected_joint_forces()[:, 0 : self._nj]
        # "commanded": the controller writes env.joint_torque during stepping; it
        # may not exist yet on the first observation (before any control step).
        tau = getattr(b, "joint_torque", None)
        if tau is None:
            return torch.zeros((self.num_envs, self._nj), device=self.device)
        return tau[:, 0 : self._nj]
