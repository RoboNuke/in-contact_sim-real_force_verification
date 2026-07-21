"""
Analyze a 1 kHz eval trajectory recorded by peg_insert_eval.py --log_trajectory.

Prints per-signal peak-to-peak / std / dominant FFT frequency (to localize shaking),
and saves a multi-panel plot of the overall behavior: force magnitude, the task-space
PD wrench (proportional to pose error), commanded joint torque, joint velocity, and the
EE motion. If the matching 15 Hz step-data pkl (from --with_step_data) is passed via
--step_data, it also overlays the policy setpoints (target_pos) and the position error
(target - ee_pos), which come from the same rows so no clock alignment is needed.

Usage:
    python real_robot_scripts/analyze_trajectory.py traj_000.npz
    python real_robot_scripts/analyze_trajectory.py traj_000.npz --step_data ep_0.pkl
    python real_robot_scripts/analyze_trajectory.py traj_000.npz --out behavior.png
"""

import argparse
import os

import numpy as np


def _mag(sig):
    """1-D magnitude: pass-through if already 1-D, else row-wise L2 norm."""
    return sig if sig.ndim == 1 else np.linalg.norm(sig, axis=1)


def dominant_freq(sig, fs):
    """Peak FFT frequency (Hz) of the (DC-removed) signal magnitude."""
    x = _mag(sig).astype(float)
    x = x - x.mean()
    n = len(x)
    if n < 8 or fs <= 0:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec[0] = 0.0  # kill DC
    return float(freqs[int(np.argmax(spec))])


def stats_line(name, sig, fs):
    x = _mag(sig).astype(float)
    return (f"  {name:<24} p2p={x.max() - x.min():9.3f}  std={x.std():8.3f}  "
            f"mean={x.mean():9.3f}  peakfreq={dominant_freq(sig, fs):6.1f} Hz")


def main():
    ap = argparse.ArgumentParser(description="Analyze a 1 kHz eval trajectory .npz")
    ap.add_argument("npz", help="traj_<i>.npz from peg_insert_eval --log_trajectory")
    ap.add_argument("--step_data", default=None,
                    help="Optional 15 Hz ep_<i>.pkl (from --with_step_data) for setpoint/error overlay")
    ap.add_argument("--out", default=None, help="Output PNG (default: <npz>.png)")
    ap.add_argument("--tstart", type=float, default=None, help="Plot window start (s)")
    ap.add_argument("--tend", type=float, default=None, help="Plot window end (s)")
    args = ap.parse_args()

    d = np.load(args.npz)
    t_ms = d["time_ms"].astype(float)
    t = (t_ms - t_ms[0]) / 1000.0
    dt = np.median(np.diff(t_ms)) / 1000.0 if len(t_ms) > 1 else 1e-3
    fs = 1.0 / dt if dt > 0 else 1000.0

    ft_filt = d["ft_filtered"]
    ft_raw = d["ft_raw"]
    fmag = np.linalg.norm(ft_filt[:, :3], axis=1)
    fmag_raw = np.linalg.norm(ft_raw[:, :3], axis=1)
    wrench = d["task_wrench"]          # [n,6] task-space PD wrench (kp*err - kd*vel)
    torque = d["joint_torques_cmd"]    # [n,7]
    jvel = d["joint_vel"]              # [n,7]
    ee = d["ee_pos"]                   # [n,3]

    # ---- Console stats ----
    print(f"\n=== {os.path.basename(args.npz)}  ({len(t)} samples, fs~{fs:.0f} Hz, {t[-1]:.2f} s) ===")
    print(stats_line("force |F| (N)", fmag, fs))
    print(stats_line("task_wrench lin (N)", wrench[:, :3], fs))
    print(stats_line("task_wrench rot (Nm)", wrench[:, 3:], fs))
    print(stats_line("joint_torque_cmd (Nm)", torque, fs))
    print(stats_line("joint_vel (rad/s)", jvel, fs))
    print(stats_line("ee_pos (m)", ee, fs))
    print("  (peakfreq = dominant oscillation frequency; compare wrench vs torque vs "
          "joint_vel to localize the shake)\n")

    # ---- Optional 15 Hz setpoints / error ----
    sp = None
    if args.step_data:
        try:
            import pandas as pd
            df = pd.read_pickle(args.step_data)
            ts = (df["t_mono"].values - df["t_mono"].values[0]) if "t_mono" in df else \
                 (df["step"].values / 15.0)
            tgt = np.stack([df[f"target_pos_{a}"].values for a in "xyz"], axis=1)
            act = np.stack([df[f"ee_pos_{a}"].values for a in "xyz"], axis=1)
            fm15 = df["force_mag"].values if "force_mag" in df else np.linalg.norm(
                np.stack([df[f"ft_f{a}"].values for a in ("x", "y", "z")], axis=1), axis=1)
            sp = {"t": ts, "target": tgt, "ee": act, "err": tgt - act, "fmag": fm15}
        except Exception as e:
            print(f"  [step_data] could not load {args.step_data}: {e}\n")

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Optional time window
    m = np.ones_like(t, dtype=bool)
    if args.tstart is not None:
        m &= t >= args.tstart
    if args.tend is not None:
        m &= t <= args.tend

    axes_specs = ["force", "wrench_lin", "wrench_rot", "torque", "jvel"]
    if sp is not None:
        axes_specs = ["setpoint", "error"] + axes_specs
    n = len(axes_specs)
    fig, axs = plt.subplots(n, 1, figsize=(13, 2.1 * n), sharex=False)
    axs = np.atleast_1d(axs)
    ai = {name: axs[i] for i, name in enumerate(axes_specs)}
    cxyz = ["tab:red", "tab:green", "tab:blue"]

    if sp is not None:
        a = ai["setpoint"]
        for j, c in enumerate(cxyz):
            a.plot(sp["t"], sp["target"][:, j], c, ls="--", lw=1.2, label=f"target {'xyz'[j]}")
            a.plot(sp["t"], sp["ee"][:, j], c, lw=1.0, alpha=0.7, label=f"ee {'xyz'[j]}")
        a.set_ylabel("setpoint\nvs ee (m)")
        a.legend(ncol=3, fontsize=7, loc="upper right")
        a.set_title(f"{os.path.basename(args.npz)} — 15 Hz setpoints (dashed) vs actual (solid)")

        a = ai["error"]
        for j, c in enumerate(cxyz):
            a.plot(sp["t"], sp["err"][:, j] * 1000.0, c, lw=1.2, label=f"err {'xyz'[j]}")
        a.axhline(0, color="k", lw=0.5)
        a.set_ylabel("pos error\n(mm)")
        a.legend(ncol=3, fontsize=7, loc="upper right")

    a = ai["force"]
    a.plot(t[m], fmag[m], "k", lw=0.8, label="|F| filtered")
    a.plot(t[m], fmag_raw[m], "0.6", lw=0.5, alpha=0.6, label="|F| raw")
    a.set_ylabel("force |F|\n(N)")
    a.legend(fontsize=7, loc="upper right")
    if sp is None:
        a.set_title(f"{os.path.basename(args.npz)} — 1 kHz trajectory")

    a = ai["wrench_lin"]
    for j, c in enumerate(cxyz):
        a.plot(t[m], wrench[m, j], c, lw=0.7, label=f"w_{'xyz'[j]}")
    a.set_ylabel("task wrench\nlin (N)\n∝ pose err")
    a.legend(ncol=3, fontsize=7, loc="upper right")

    a = ai["wrench_rot"]
    for j, c in enumerate(cxyz):
        a.plot(t[m], wrench[m, 3 + j], c, lw=0.7, label=f"tau_{'xyz'[j]}")
    a.set_ylabel("task wrench\nrot (Nm)")
    a.legend(ncol=3, fontsize=7, loc="upper right")

    a = ai["torque"]
    for j in range(torque.shape[1]):
        a.plot(t[m], torque[m, j], lw=0.6, label=f"j{j}")
    a.set_ylabel("joint torque\ncmd (Nm)")
    a.legend(ncol=7, fontsize=6, loc="upper right")

    a = ai["jvel"]
    for j in range(jvel.shape[1]):
        a.plot(t[m], jvel[m, j], lw=0.6, label=f"j{j}")
    a.set_ylabel("joint vel\n(rad/s)")
    a.legend(ncol=7, fontsize=6, loc="upper right")
    a.set_xlabel("time (s)")

    for a in axs:
        a.grid(True, alpha=0.3)
    fig.tight_layout()

    out = args.out or (os.path.splitext(args.npz)[0] + ".png")
    fig.savefig(out, dpi=110)
    print(f"saved plot -> {out}\n")


if __name__ == "__main__":
    main()
