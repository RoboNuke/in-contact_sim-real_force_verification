"""Config for the joint-torque observation wrapper (:class:`wrappers.joint_obs_wrapper.JointObsWrapper`).

Kept as a plain dataclass with no gym/torch/Isaac imports so the YAML config
manager can load it without the Sim app running. The runner constructs the
wrapper from this and applies it to the raw FORGE env (below the skrl wrapper),
appending the arm joint torques to the policy (and optionally critic) obs.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class JointObsCfg:
    """Append the Franka arm joint torques to a stock FORGE env's observations."""

    enabled: bool = False
    """Master toggle. When False the runner does not apply the wrapper at all
    (observations are left unchanged)."""

    source: str = "measured"
    """Which joint-torque signal to append:
      ``"measured"``  -> ``get_dof_projected_joint_forces()`` — the solver's
                         DOF-direction reaction incl. external contact. This is
                         the real-robot joint-torque-sensor analog (same source
                         the dyn_pinv force estimate uses).
      ``"commanded"`` -> ``env.joint_torque`` — the impedance controller's
                         commanded arm torque (zeros before the first control
                         step)."""

    num_joints: int = 7
    """Number of arm joints appended (Franka has 7). Each adds one obs dim; the
    observation grows by ``num_joints`` on the policy (and critic if enabled)."""

    include_critic: bool = True
    """Also append the joint torques to the critic ``state`` observation. The
    critic is already privileged (asymmetric actor-critic); set False to feed the
    torques to the policy only."""
