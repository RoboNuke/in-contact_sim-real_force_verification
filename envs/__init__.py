"""Project-local Isaac Lab environments.

Importing this package registers the gym ids below, so ``gym.make(...)`` (and
``isaaclab_tasks.utils.parse_env_cfg``) can resolve them. Import it only *after*
the Isaac Lab ``AppLauncher`` has booted.
"""

import gymnasium as gym

from .contact_force_test_env_cfg import ContactForceTestEnvCfg

gym.register(
    id="Isaac-ContactForceTest-Direct-v0",
    entry_point="envs.contact_force_test_env:ContactForceTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ContactForceTestEnvCfg,
    },
)
