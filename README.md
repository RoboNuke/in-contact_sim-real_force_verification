# in-contact_sim-real_force_verification

Sim-to-real force verification for contact-rich robotic manipulation.

This project trains reinforcement-learning policies for contact-rich manipulation
in NVIDIA Isaac Lab / Isaac Sim and studies how faithfully the contact forces the
policies rely on transfer from simulation to a real robot.

## Overview

Policies are trained with a SAC agent (built on [skrl](https://skrl.readthedocs.io))
using a "BlockSimba" network, and evaluated on contact-rich Factory/Forge tasks
(peg insertion, nut threading, gear meshing) as well as standard benchmarks
(cartpole, ant, humanoid) and block-picking tasks.

## Layout

| Path | Contents |
|------|----------|
| `learning/` | SAC agent, train/eval runner, and calibration / prediction-quality / rescue metrics |
| `models/` | BlockSimba network and per-agent preprocessor wrappers |
| `wrappers/` | Isaac Lab environment wrappers — Factory/Forge, success detection, reward/force decomposition, state-snapshot, and recording |
| `memory/` | Replay buffers (rescue, trajectory, multi-random) |
| `configs/` | Experiment YAMLs (`exp_cfgs/`) and config managers (`manager/`) |
| `data_analysis/` | Standalone analysis / sanity-check scripts |
| `launchers/` | Shell entry points (e.g. end-to-end train→save→load→eval smoke test) |

## Setup

Create the conda environment from the pinned local spec:

```bash
conda env create -n trans -f environment.yml
conda activate trans
```

## Running

Train and evaluate via the runner or the end-to-end launcher:

```bash
# End-to-end: train -> save -> load -> eval
./launchers/sac_block_e2e.sh configs/exp_cfgs/pick_block_baseline.yaml my_run

# Or drive the runner directly
python learning/runner.py --config configs/exp_cfgs/default.yaml \
    --experiment_name my_run --mode train --headless
```
