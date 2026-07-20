"""
Side-by-side sync check for a recorded trial.

Given a folder that holds a per-step DataFrame (`ep_<i>.pkl`) and its video
(`ep_<i>.mp4`) from `peg_insert_eval.py --with_step_data`, render a new mp4 with the
trial video on the LEFT and a real-time plot of the z-axis contact force (`ft_fz`)
on the RIGHT, growing from t=0 to the current video time. Use it to eyeball whether
the video and the logged data are actually aligned.

Sync: video frames carry no timestamps, but the DataFrame stores, per step, both
`t_mono` (step time) and `frame_time` (the monotonic time of its matched frame) plus
`frame_idx`. We interpolate every video frame's monotonic time from the
(frame_idx -> frame_time) pairs, subtract the trial start, and draw the force curve
up to that time.

Usage:
    python real_robot_scripts/plot_trial_sync.py --folder data/real_robot_eval/<run> --episode 0
    python real_robot_scripts/plot_trial_sync.py --pkl <ep.pkl> --mp4 <ep.mp4> --out synced.mp4
    # optionally draw the break threshold:
    python real_robot_scripts/plot_trial_sync.py --folder <dir> --episode 0 --break_force 10
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio
import cv2


def frame_times_from_df(df: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Monotonic time for every video frame index, interpolated from step matches."""
    m = df["frame_idx"].to_numpy() >= 0
    fi = df["frame_idx"].to_numpy()[m].astype(float)
    ft = df["frame_time"].to_numpy()[m].astype(float)
    if len(fi) < 2 or not np.isfinite(ft).any():
        # No usable frame timestamps: fall back to a uniform span over the trial.
        t0, t1 = df["t_mono"].iloc[0], df["t_mono"].iloc[-1]
        return np.linspace(t0, t1, n_frames)
    order = np.argsort(fi)
    fi, ft = fi[order], ft[order]
    # collapse duplicate frame_idx (keep mean time)
    uniq, inv = np.unique(fi, return_inverse=True)
    ft_u = np.array([ft[inv == k].mean() for k in range(len(uniq))])
    return np.interp(np.arange(n_frames), uniq, ft_u)


def render(folder, episode, pkl, mp4, out, break_force, plot_width):
    if folder is not None:
        pkl = pkl or os.path.join(folder, f"ep_{episode}.pkl")
        mp4 = mp4 or os.path.join(folder, f"ep_{episode}.mp4")
        out = out or os.path.join(folder, f"ep_{episode}_synced.mp4")
    if not (pkl and mp4):
        raise SystemExit("Provide --folder (+--episode) or both --pkl and --mp4")
    out = out or os.path.splitext(mp4)[0] + "_synced.mp4"

    df = pd.read_pickle(pkl)
    if "ft_fz" not in df.columns:
        raise SystemExit(f"'ft_fz' not in {pkl}; columns={list(df.columns)[:10]}...")

    t0 = float(df["t_mono"].iloc[0])
    step_t = df["t_mono"].to_numpy() - t0
    fz = df["ft_fz"].to_numpy()
    total_t = float(step_t[-1]) if len(step_t) else 1.0

    reader = imageio.get_reader(mp4)
    fps = float(reader.get_meta_data().get("fps", 30.0))
    frames = [f for f in reader]
    reader.close()
    n = len(frames)
    if n == 0:
        raise SystemExit(f"No frames in {mp4}")
    vh, vw = frames[0].shape[:2]

    fr_times = frame_times_from_df(df, n) - t0

    # Fixed-axes plot we update per frame (fast: reuse the figure).
    dpi = 100
    fig, ax = plt.subplots(figsize=(plot_width / dpi, vh / dpi), dpi=dpi)
    ymin, ymax = float(np.min(fz)), float(np.max(fz))
    pad = max(0.5, 0.1 * (ymax - ymin))
    ax.set_xlim(0, max(total_t, 1e-3))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force z  (N)")
    ax.set_title("z-axis contact force")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    if break_force:
        ax.axhline(break_force, color="crimson", lw=1.0, ls="--", alpha=0.8, label=f"break {break_force:g} N")
        ax.axhline(-break_force, color="crimson", lw=1.0, ls="--", alpha=0.8)
        ax.legend(loc="upper left", fontsize=8)
    (line,) = ax.plot([], [], color="tab:blue", lw=1.5)
    (dot,) = ax.plot([], [], "o", color="tab:red", ms=5)
    (vline,) = ax.plot([], [], color="tab:red", lw=0.8, alpha=0.5)
    fig.tight_layout()

    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    try:
        for fi in range(n):
            tcur = float(fr_times[fi])
            mask = step_t <= tcur
            line.set_data(step_t[mask], fz[mask])
            if mask.any():
                dot.set_data([step_t[mask][-1]], [fz[mask][-1]])
                vline.set_data([tcur, tcur], [ymin - pad, ymax + pad])
            fig.canvas.draw()
            plot_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            if plot_rgb.shape[0] != vh:
                plot_rgb = cv2.resize(plot_rgb, (int(plot_rgb.shape[1] * vh / plot_rgb.shape[0]), vh))
            combined = np.hstack([frames[fi], plot_rgb])
            writer.append_data(combined)
    finally:
        writer.close()
        plt.close(fig)

    print(f"wrote {out}  ({n} frames @ {fps:.1f} fps, {vw}x{vh} video + {plot_rgb.shape[1]}px plot)")


def main():
    p = argparse.ArgumentParser(description="Render trial video + real-time force-z plot side by side")
    p.add_argument("--folder", default=None, help="Trial folder containing ep_<i>.pkl and ep_<i>.mp4")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--pkl", default=None)
    p.add_argument("--mp4", default=None)
    p.add_argument("--out", default=None, help="Output mp4 (default: <folder>/ep_<i>_synced.mp4)")
    p.add_argument("--break_force", type=float, default=None, help="Draw +/- break threshold lines")
    p.add_argument("--plot_width", type=int, default=640, help="Plot panel width in px")
    args = p.parse_args()
    render(args.folder, args.episode, args.pkl, args.mp4, args.out, args.break_force, args.plot_width)


if __name__ == "__main__":
    main()
