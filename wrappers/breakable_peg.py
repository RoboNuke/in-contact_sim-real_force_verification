"""Make the base Isaac Lab FORGE peg breakable during training.

A self-contained :class:`gymnasium.Wrapper` around the *stock* FORGE env
(``isaaclab_tasks.direct.forge.ForgeEnv`` / ``Isaac-Forge-*-Direct-v0``). It
imports only ``torch``, ``gymnasium`` and Isaac Lab — nothing project-local — so
it drops onto an unmodified FORGE task. Its config is duck-typed: any object
exposing ``break_force`` / ``peg_break_rew`` / ``force_margin`` works (the
project supplies :class:`configs.manager.breakable_peg_cfg.BreakablePegCfg`);
standalone callers can instead pass those three as keyword arguments.

What it does
------------
1. **Breakable peg.** After each step, the smoothed contact-force magnitude
   ``||force_sensor_smooth[:, :3]||`` (the exact scalar FORGE already uses for
   its contact penalty) is compared against ``break_force``. Envs above
   threshold "break": they are ``terminated`` this step and a ``peg_break_rew``
   penalty is added to their reward. A break is a failure, not a success.

2. **Efficient per-env resets.** FORGE/Factory ``_reset_idx`` re-grasps the peg
   by *stepping the whole scene's physics* (IK loop + settle) and explicitly
   "assumes all envs will always be reset at the same time". A peg that breaks
   mid-episode therefore can't use the native reset for its subset without
   corrupting the other envs. So on the first synchronized (all-env) reset we
   let the real grasp run and cache the resulting scene + FORGE bookkeeping
   state; when a subset breaks we restore that cached state into just those
   envs via ``scene.reset_to(...)`` + tensor copies — no physics stepping, no
   re-grasp. Full (all-env) resets always re-run the real grasp and refresh the
   cache.

3. **Threshold invariant.** ``break_force`` must sit strictly above FORGE's own
   contact-penalty threshold — the force at which the contact penalty turns on,
   randomized per episode from ``cfg.task.contact_penalty_threshold_range``. The
   break force must exceed the *largest* value that range can produce (plus a
   margin); this is validated at construction and raises on violation.

Hooking
-------
Only ``env.unwrapped``'s ``_get_dones`` / ``_get_rewards`` / ``_reset_idx`` are
monkeypatched (never ``reset()`` — FORGE does not call it during training). The
patches call the originals and layer break/reset behavior on top, so the wrapper
composes with any outer vec-env wrappers used by the trainer.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

# Isaac Lab (``isaaclab.envs.DirectRLEnv``) is imported lazily inside
# :meth:`BreakablePegWrapper._efficient_reset` — not at module top — so this
# module can be imported without booting the Sim app.


# Per-env FORGE/Factory tensors that ``_reset_idx`` (re)initializes on the
# object and that must be replayed when we skip the real reset. Physics state
# (robot joints, peg/hole poses + velocities) is restored separately via
# ``scene.reset_to()``; these are the derived/randomized attributes living on
# the env. Missing attrs (version differences) are silently skipped, and only
# tensors whose leading dim is ``num_envs`` are cached.
_RESET_STATE_ATTRS: tuple[str, ...] = (
    # finite-difference history
    "prev_joint_pos",
    "prev_fingertip_pos",
    "prev_fingertip_quat",
    "ee_linvel_fd",
    "ee_angvel_fd",
    # action / EMA state
    "actions",
    "prev_actions",
    "ema_factor",
    # fixed-asset observation frames
    "fixed_pos_obs_frame",
    "init_fixed_pos_obs_noise",
    # controller gains + action-clip thresholds
    "task_prop_gains",
    "task_deriv_gains",
    "pos_threshold",
    "rot_threshold",
    # FORGE randomized thresholds
    "contact_penalty_thresholds",
    "dead_zone_thresholds",
    # force smoothing + obs-flip randomization
    "force_sensor_world_smooth",
    "force_sensor_world",
    "flip_quats",
    # joint controller target
    "ctrl_target_joint_pos",
)


class BreakablePegWrapper(gym.Wrapper):
    """Breakable-peg + efficient-reset wrapper for the stock FORGE env."""

    def __init__(
        self,
        env: gym.Env,
        cfg: Any = None,
        *,
        break_force: float = 25.0,
        peg_break_rew: float = -10.0,
        force_margin: float = 1.0,
    ) -> None:
        """Wrap ``env`` with breakable-peg mechanics.

        :param cfg: Optional config object exposing ``break_force`` /
            ``peg_break_rew`` / ``force_margin`` (e.g.
            :class:`configs.manager.breakable_peg_cfg.BreakablePegCfg`). When
            given it takes precedence over the keyword arguments.
        :param break_force: Standalone fallback for ``cfg.break_force`` (N).
        :param peg_break_rew: Standalone fallback for ``cfg.peg_break_rew``.
        :param force_margin: Standalone fallback for ``cfg.force_margin`` (N).
        """
        super().__init__(env)
        # Duck-type the config; fall back to the keyword arguments when no cfg.
        self._break_force_n = float(getattr(cfg, "break_force", break_force))
        self._peg_break_rew = float(getattr(cfg, "peg_break_rew", peg_break_rew))
        self._force_margin = float(getattr(cfg, "force_margin", force_margin))

        u = self.unwrapped
        self._u = u
        self._device = u.device
        self._num_envs = u.num_envs

        self._breakable = self._break_force_n > 0.0

        # Validate the threshold invariant up front (fail loud on a bad config).
        if self._breakable:
            self._validate_break_force()

        # Per-env break force (kept as a tensor so callers can override per env
        # later, e.g. per-agent break forces in multi-agent training).
        self._break_force = torch.full(
            (self._num_envs,), self._break_force_n, device=self._device
        )
        # Break mask for the current step (published on info + consumed by the
        # reward hook). Set inside the _get_dones hook.
        self._broke = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        # Efficient-reset cache, populated on the first synchronized full reset.
        self._cached_scene_state: dict[str, dict[str, dict[str, torch.Tensor]]] | None = None
        self._cached_reset_tensors: dict[str, torch.Tensor] = {}

        # Save + install method hooks on the unwrapped env.
        self._orig_get_dones = u._get_dones
        self._orig_get_rewards = u._get_rewards
        self._orig_reset_idx = u._reset_idx
        u._get_dones = self._hooked_get_dones
        u._get_rewards = self._hooked_get_rewards
        u._reset_idx = self._hooked_reset_idx

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_break_force(self) -> None:
        """Ensure ``break_force`` exceeds FORGE's max contact-penalty threshold."""
        try:
            thresh_range = self._u.cfg.task.contact_penalty_threshold_range
        except AttributeError as exc:  # not a FORGE task
            raise TypeError(
                "BreakablePegWrapper requires a FORGE env exposing "
                "cfg.task.contact_penalty_threshold_range; got "
                f"{self._u.__class__.__name__}."
            ) from exc

        max_thresh = float(max(thresh_range))
        min_allowed = max_thresh + self._force_margin
        if self._break_force_n <= min_allowed:
            raise ValueError(
                "break_force must stay above FORGE's contact-penalty threshold: "
                f"break_force={self._break_force_n} N must exceed "
                f"max(contact_penalty_threshold_range)={max_thresh} N + "
                f"force_margin={self._force_margin} N = {min_allowed} N. "
                "Raise break_force or lower contact_penalty_threshold_range."
            )

    # ------------------------------------------------------------------
    # _get_dones hook — break the peg + terminate
    # ------------------------------------------------------------------
    def _hooked_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # FORGE's _get_dones refreshes intermediate values (incl. the smoothed
        # force), so read the force AFTER calling it.
        terminated, truncated = self._orig_get_dones()

        if not self._breakable:
            self._broke = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            return terminated, truncated

        contact_force = torch.linalg.vector_norm(self._u.force_sensor_smooth[:, 0:3], dim=-1)
        self._broke = contact_force > self._break_force
        # Break is a real termination (failure), not a timeout/truncation.
        terminated = terminated | self._broke
        return terminated, truncated

    # ------------------------------------------------------------------
    # _get_rewards hook — apply break penalty
    # ------------------------------------------------------------------
    def _hooked_get_rewards(self) -> torch.Tensor:
        rew = self._orig_get_rewards()
        if self._breakable and bool(self._broke.any()):
            rew = rew + self._broke.to(rew.dtype) * self._peg_break_rew
        return rew

    # ------------------------------------------------------------------
    # _reset_idx hook — full grasp (+cache) vs efficient restore
    # ------------------------------------------------------------------
    def _hooked_reset_idx(self, env_ids) -> None:
        env_ids = self._as_long(env_ids)

        # All envs reset together (episode timeout, or the trainer's initial
        # reset): run FORGE's real synchronized grasp, then refresh the cache.
        if env_ids.numel() == self._num_envs:
            self._orig_reset_idx(env_ids)
            self._cache_reset_state()
            return

        # A subset broke mid-episode: restore cached state without re-grasping.
        if self._cached_scene_state is None:
            raise RuntimeError(
                "BreakablePegWrapper: efficient (subset) reset requested before "
                "any full reset cached a template state. Ensure env.reset() runs "
                "once (all envs) before per-env breaking can occur."
            )
        self._efficient_reset(env_ids)

    def _cache_reset_state(self) -> None:
        """Snapshot the post-grasp scene + FORGE bookkeeping for later restore."""
        # is_relative=True stores poses relative to env origins so they can be
        # written back into the matching env slots via reset_to(is_relative=True).
        self._cached_scene_state = self._u.scene.get_state(is_relative=True)
        self._cached_reset_tensors = {}
        for name in _RESET_STATE_ATTRS:
            t = getattr(self._u, name, None)
            if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == self._num_envs:
                self._cached_reset_tensors[name] = t.clone()

    def _efficient_reset(self, env_ids: torch.Tensor) -> None:
        """Restore cached reset state into ``env_ids`` — no physics stepping."""
        # Lazy import: keep the module importable without the Sim app.
        from isaaclab.envs import DirectRLEnv

        # (1) Lightweight base reset: scene.reset (buffers/actuators), reset-mode
        #     events, obs/action noise, and episode_length_buf[env_ids] = 0.
        #     Deliberately the grandparent DirectRLEnv path, NOT Factory's grasp.
        DirectRLEnv._reset_idx(self._u, env_ids)

        # (2) Restore the cached post-grasp physics (robot joints, peg/hole
        #     poses + velocities) into just these envs.
        sliced = self._slice_state(self._cached_scene_state, env_ids)
        self._u.scene.reset_to(sliced, env_ids=env_ids, is_relative=True)

        # (3) Replay the cached FORGE bookkeeping tensors for these envs.
        for name, cached in self._cached_reset_tensors.items():
            getattr(self._u, name)[env_ids] = cached[env_ids]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _as_long(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self._num_envs, device=self._device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self._device)
        return env_ids.to(device=self._device, dtype=torch.long).view(-1)

    @staticmethod
    def _slice_state(
        state: dict[str, dict[str, dict[str, torch.Tensor]]], env_ids: torch.Tensor
    ) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """Index every tensor in a get_state() dict down to ``env_ids``."""
        return {
            kind: {
                name: {comp: val[env_ids] for comp, val in comps.items()}
                for name, comps in entities.items()
            }
            for kind, entities in state.items()
        }

    # ------------------------------------------------------------------
    # gym.Wrapper API — publish break info
    # ------------------------------------------------------------------
    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # _broke was set inside the _get_dones hook during this step().
        if isinstance(info, dict):
            info["peg_broke"] = self._broke.clone()
            info["peg_broke_rate"] = self._broke.float().mean()
        return obs, reward, terminated, truncated, info
