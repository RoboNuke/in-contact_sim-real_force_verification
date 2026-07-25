"""Project-local Isaac Lab environments.

Importing this package registers the gym ids below, so ``gym.make(...)`` (and
``isaaclab_tasks.utils.parse_env_cfg``) can resolve them. Import it only *after*
the Isaac Lab ``AppLauncher`` has booted.
"""

import gymnasium as gym

from .contact_force_test_env_cfg import ContactForceTestEnvCfg
from .forge_contact_test_env_cfg import ForgeContactTestEnvCfg
from .forge_scaled_hole_cfg import SCALED_HOLE_VARIANTS

gym.register(
    id="Isaac-ContactForceTest-Direct-v0",
    entry_point="envs.contact_force_test_env:ContactForceTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ContactForceTestEnvCfg,
    },
)

gym.register(
    id="Isaac-ForgeContactTest-Direct-v0",
    entry_point="envs.forge_contact_test_env:ForgeContactTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeContactTestEnvCfg,
    },
)

# Scaled-hole FORGE peg-insert variants: same task as
# Isaac-Forge-PegInsert-Direct-v0 but with an enlarged bore clearance. Reuses the
# upstream ForgeEnv entry point; only the fixed-asset spawn scale differs.
for _suffix, (_cfg_cls, _clearance_m) in SCALED_HOLE_VARIANTS.items():
    gym.register(
        id=f"Isaac-Forge-PegInsert-{_suffix}-Direct-v0",
        entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _cfg_cls,
        },
    )
