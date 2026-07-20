"""
Per-trial data + video persistence for real-robot eval (`--with_step_data`).

Everything is saved LOCALLY (never uploaded to wandb):
  <data_dir>/<run>_a<agent>_s<step>/
      ep_<i>.pkl        full per-step DataFrame (all signals) + matched frame_idx
      ep_<i>.csv        scalar columns for quick inspection
      ep_<i>.mp4        the trial video (imageio/ffmpeg), optional telemetry overlay
      summary.pkl/.csv  one row per episode (aggregate metrics)

Frames are matched to policy steps by nearest `time.monotonic()` timestamp after
the trial, so the video and the DataFrame are aligned both ways.
"""

import os

import numpy as np
import pandas as pd

try:
    import cv2  # overlay drawing only (not used as a video writer)
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


def _nearest_indices(query_times, ref_times):
    """For each query time, the index of the nearest ref time (ref assumed sorted)."""
    ref = np.asarray(ref_times, dtype=float)
    out = np.searchsorted(ref, np.asarray(query_times, dtype=float))
    out = np.clip(out, 1, len(ref) - 1)
    left = ref[out - 1]
    right = ref[out]
    choose_left = (np.asarray(query_times, dtype=float) - left) <= (right - np.asarray(query_times, dtype=float))
    return np.where(choose_left, out - 1, out)


def _overlay(frame, text_lines, danger=False):
    if not _HAVE_CV2:
        return frame
    img = np.ascontiguousarray(frame)
    y = 18
    for line in text_lines:
        color = (255, 60, 60) if danger else (0, 255, 0)  # RGB
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 20
    return img


def _write_video(path, frames, df, break_force, outcome, overlay, fallback_fps):
    """Encode frames to mp4 via imageio (ffmpeg). Returns (path, fps) or (None, 0)."""
    if not frames:
        return None, 0.0
    import imageio

    times = [t for t, _ in frames]
    if len(times) > 1 and (times[-1] - times[0]) > 1e-6:
        fps = (len(times) - 1) / (times[-1] - times[0])
    else:
        fps = float(fallback_fps)
    fps = float(np.clip(fps, 1.0, 120.0))

    # frame -> nearest step (for the overlay values)
    step_times = df["t_mono"].to_numpy()
    frame_step = _nearest_indices(times, step_times) if len(step_times) else np.zeros(len(times), int)

    writer = imageio.get_writer(path, fps=fps, macro_block_size=None)
    try:
        for fi, (_, img) in enumerate(frames):
            if overlay and len(step_times):
                s = int(frame_step[fi])
                fmag = float(df["force_mag"].iloc[s])
                contact = bool(df["in_contact"].iloc[s])
                danger = fmag >= 0.8 * break_force
                lines = [
                    f"step {s}  {outcome}",
                    f"F={fmag:5.2f}N / brk {break_force:.0f}N" + ("  CONTACT" if contact else ""),
                ]
                img = _overlay(img, lines, danger=danger)
            writer.append_data(img)
    finally:
        writer.close()
    return path, fps


def save_trial(step_records, frames, out_dir, ep_idx, break_force, outcome,
               overlay=True, fallback_fps=30):
    """Persist one episode's per-step DataFrame + video, with frame<->step matching.

    Args:
        step_records: list of per-step flat dicts (must include 't_mono').
        frames: list of (t_monotonic, rgb_uint8) from the recorder.
        out_dir: episode output directory (created if needed).
        ep_idx: episode index.
        break_force: N, for the overlay danger colouring.
        outcome: "SUCCESS" | "BREAK" | "TIMEOUT".
    Returns dict of written paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(step_records)

    # Attach the matched video frame to each step (nearest by time).
    if frames:
        frame_times = [t for t, _ in frames]
        step_frame = _nearest_indices(df["t_mono"].to_numpy(), frame_times)
        df["frame_idx"] = step_frame
        df["frame_time"] = [frame_times[i] for i in step_frame]
    else:
        df["frame_idx"] = -1
        df["frame_time"] = np.nan

    pkl = os.path.join(out_dir, f"ep_{ep_idx}.pkl")
    csv = os.path.join(out_dir, f"ep_{ep_idx}.csv")
    df.to_pickle(pkl)
    # CSV: only scalar columns (drop any list/array-valued ones).
    scalar_cols = [c for c in df.columns if np.isscalar(df[c].iloc[0]) or isinstance(df[c].iloc[0], (int, float, bool, np.number))]
    df[scalar_cols].to_csv(csv, index=False)

    mp4 = os.path.join(out_dir, f"ep_{ep_idx}.mp4")
    video_path, fps = _write_video(mp4, frames, df, break_force, outcome, overlay, fallback_fps)

    print(f"    [step-data] {pkl}  ({len(df)} steps)"
          + (f"  +  {video_path}  ({len(frames)} frames @ {fps:.1f} fps)" if video_path else "  (no frames)"))
    return {"pkl": pkl, "csv": csv, "mp4": video_path, "n_frames": len(frames)}


def save_summary(rows, out_dir):
    """One row per episode -> summary.pkl/.csv."""
    if not rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    pkl = os.path.join(out_dir, "summary.pkl")
    df.to_pickle(pkl)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"[step-data] summary: {pkl}  ({len(df)} episodes)")
    return pkl
