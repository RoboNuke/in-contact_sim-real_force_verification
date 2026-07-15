# Real-robot contact-force test (FR3)

Hardware counterpart of the sim contact-force push sweep
(`scripts/param_sweep_run.py` + `envs/contact_force_test_env.py`). It runs the
identical protocol on a real Franka FR3 and logs the same data, so sim and real
force readings can be compared directly.

The FR3 interface (`FrankaInterface`, the hybrid controller, quaternion/frame
utilities, and a mock robot) is **vendored** here from RoboNuke's
`Continuous_Force_RL/real_robot_exps/` and customized — nothing at runtime
depends on that repo.

## Layout

| file | purpose |
|------|---------|
| `contact_force_test.py` | the experiment driver (run this) |
| `config.yaml` | robot connection, frames, F/T filtering, and the experiment protocol/gains |
| `pro_robot_interface.py` | process-based `FrankaInterface` (1 kHz comm + compute in separate processes) |
| `robot_interface.py` | `StateSnapshot`, safety, quaternion/frame helpers, FR3 limits |
| `hybrid_controller.py` | `ControlTargets` + pure-PyTorch wrench→torque control math |
| `mock_pylibfranka.py` | fake robot for off-hardware dry runs (`--mock`) |

## What it does

You place the peg tip at the surface by hand. Then, for each target force and
each repeat, the script:

1. returns to the pose you set (the surface / **force = 0 reference**),
2. re-tares the F/T estimate in free space (`calibrate_ft_bias`),
3. pushes straight down in **position control** to `z = surface_ref_z −
   force/gain`, and
4. holds `hold_seconds`, logging, then retracts to (1) for the next rep.

`gain` is the z proportional gain (default **565 N/m** — the sim
`default_task_prop_gains[2]`). The surface is rigid, so the controller settles at
`force = gain × depth`; the "stiffness" that sets the force is the controller
gain, which is known exactly. Defaults: forces `[1, 2, 5, 10, 15]` N × 10 reps =
50 pushes, 2 s each.

## Positioning procedure (before running)

1. Put the peg in the gripper (the script clamps it with `close_gripper`; use
   `--no_grip` for a fixed tool).
2. Jog/hand-guide the robot so the peg tip is **just touching the surface**,
   pointing straight down (roll ≈ π), at the spot you want to test. This pose
   becomes the force = 0 reference for the whole run — place it carefully: at
   1 N the commanded depth is only ~1.8 mm, so a 0.2 mm reference error is ~11 %
   of the target. A hair of clearance (like the sim's 0.2 mm `tip_gap`) keeps the
   per-rep tare contact-free.
3. Run the script. It reads this pose once at start and returns to it between
   every rep.

## Running

Dry run (no hardware — validates the full loop, process spawn, logging, save):

```bash
python real_robot_scripts/contact_force_test.py \
    --config real_robot_scripts/config.yaml --mock \
    --forces 1,2 --reps 2 --hold 0.5
```

On the robot (set `robot.ip` and `use_mock: false` in `config.yaml` first):

```bash
python real_robot_scripts/contact_force_test.py --config real_robot_scripts/config.yaml
```

Useful overrides: `--forces 1,2,5,10,15`, `--reps N`, `--hold SECONDS`,
`--out_dir DIR`, `--no_grip`. **First on-robot run: start small**
(`--forces 1 --reps 2`) to confirm force levels and stability.

## Output

One compressed `.npz` in `out_dir` (default `data/real_robot/`),
`contact_force_real_<timestamp>.npz`.

**15 Hz snapshot fields**, stacked to shape `(n_forces, n_reps, n_steps, …)` —
these mirror the sim per-step log (`envs/contact_force_test_env.py`):

| field | shape tail | sim counterpart |
|-------|-----------|-----------------|
| `joint_pos`, `joint_vel` | `(7,)` | same (arm joints) |
| `joint_torques_meas` | `(7,)` | `joint_torque_measured` (measured `tau_J`) |
| `mass_matrix` | `(7,7)` | `mass_matrix` |
| `jacobian` | `(6,7)` | `jacobian` |
| `ee_pos`, `ee_quat`, `ee_linvel`, `ee_angvel` | `(3/4,)` | same |
| `ft_ee` | `(6,)` | `contact_force_ee` — real analog (from `O_F_ext_hat_K`; the FR3 has no true contact sensor) |
| `est_force_ee_pinv` | `(3,)` | `est_force_ee_pinv` — `pinv(Jᵀ)·τ`, EE frame |
| `est_force_ee_dyn` | `(3,)` | `est_force_ee_dynconsistent` — `J̄ᵀ·τ`, EE frame |

`est_force_ee_pinv/dyn` are recomputed here with the same formulas as the sim
(`_map_torque_to_ee_force[_dyn]`), from the measured `tau_J`, Jacobian, and mass
matrix.

**1 kHz trajectory:** `traj_1khz`, an object array `(n_forces, n_reps)`; each
cell is a dict of the built-in high-rate buffer: `time_ms`, `ee_pos`, `ee_quat`,
`joint_pos`, `joint_vel`, `ft_raw`, `ft_filtered`, `joint_torques_cmd`
(commanded/applied-torque analog), `task_wrench`. Load with
`np.load(path, allow_pickle=True)`.

**Metadata:** `force_targets`, `depths`, `tare_bias (n_forces,n_reps,6)`,
`meta_gain`, `prop_gains`, `control_rate_hz`, `hold_seconds`, `n_steps`, `reps`,
`start_ee_pos/quat`, `start_joint_q`, `surface_ref_z`, `wall_time_total`,
`use_mock`.

## Safety / notes

- The interface caps force at 50 N (`_SAFETY_MARGIN_FORCE`), slews torque at
  ≤ 0.75 Nm/ms, and `check_safety` runs every 15 Hz step (joint pos/vel/force
  limits, process health). 15 N is well within limits.
- **Gain 565 is stiff for the FR3.** It's the default for sim fidelity, but
  RoboNuke's tuned real gains are ~100–150. If the push feels unstable, lower
  `experiment.prop_gains` / `experiment.gain` in `config.yaml` (note: changing
  `gain` also rescales the depth-per-force, keeping `force = gain × depth`).
- Ctrl-C or any safety violation stops the robot on the next comm timeout;
  `shutdown()` always runs.
