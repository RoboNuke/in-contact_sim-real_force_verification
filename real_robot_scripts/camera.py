"""
RealSense color capture for real-robot trial recording.

A background thread captures color frames with `time.monotonic()` timestamps into
a per-episode buffer so it never blocks the 15 Hz control loop. Each episode is
bounded by `begin_segment()` / `end_segment()`; the pipeline stays alive between
episodes (no re-warmup). Frames are RGB uint8 `(H, W, 3)` (imageio convention).

`make_recorder(...)` returns a `MockRecorder` when the robot is in mock mode (or
pyrealsense2 / a camera is unavailable) so the full record→save path is testable
off-hardware.
"""

import threading
import time

import numpy as np


class _BaseRecorder:
    def __init__(self, width, height, fps):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._buf = []
        self._active = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # ---- lifecycle ----
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def begin_segment(self):
        with self._lock:
            self._buf = []
            self._active = True

    def end_segment(self):
        with self._lock:
            self._active = False
            frames = self._buf
            self._buf = []
        return frames

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._teardown()

    # ---- to implement ----
    def _grab(self):
        """Return an RGB uint8 (H,W,3) frame, or None if none available."""
        raise NotImplementedError

    def _teardown(self):
        pass

    def _loop(self):
        period = 1.0 / max(self.fps, 1)
        while not self._stop.is_set():
            img = self._grab()
            if img is not None:
                with self._lock:
                    if self._active:
                        self._buf.append((time.monotonic(), img))
            else:
                time.sleep(period * 0.5)


class RealSenseRecorder(_BaseRecorder):
    """Intel RealSense color capture via pyrealsense2 (non-blocking poll)."""

    def __init__(self, width=640, height=480, fps=30, serial=None):
        super().__init__(width, height, fps)
        self.serial = serial
        self._pipeline = None

    def start(self):
        import pyrealsense2 as rs
        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(str(self.serial))
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self._pipeline.start(config)
        # Warm up: discard the first few frames (auto-exposure settling).
        for _ in range(5):
            try:
                self._pipeline.wait_for_frames(timeout_ms=2000)
            except Exception:
                break
        super().start()
        print(f"[RealSenseRecorder] streaming color {self.width}x{self.height} @ {self.fps} fps")

    def _grab(self):
        frames = self._pipeline.poll_for_frames()
        if not frames:
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data()).copy()  # RGB uint8

    def _teardown(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


class MockRecorder(_BaseRecorder):
    """Synthetic frames for off-hardware testing (no camera needed)."""

    def __init__(self, width=320, height=240, fps=30):
        super().__init__(width, height, fps)
        self._n = 0
        self._last = None

    def start(self):
        super().start()
        print(f"[MockRecorder] synthetic {self.width}x{self.height} @ {self.fps} fps")

    def _grab(self):
        # Pace synthetic frames to the target fps.
        now = time.monotonic()
        if self._last is not None and (now - self._last) < 1.0 / max(self.fps, 1):
            return None
        self._last = now
        self._n += 1
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :, 0] = (self._n * 3) % 256   # drifting red so frames differ
        img[:, :, 2] = 64
        return img


def make_recorder(camera_cfg: dict, use_mock: bool):
    """Return a recorder. MockRecorder when use_mock or pyrealsense2/camera absent."""
    cfg = camera_cfg or {}
    width = cfg.get("width", 640)
    height = cfg.get("height", 480)
    fps = cfg.get("fps", 30)
    if use_mock:
        return MockRecorder(width=min(width, 320), height=min(height, 240), fps=fps)
    try:
        import pyrealsense2  # noqa: F401
    except Exception:
        print("[camera] pyrealsense2 not installed — falling back to MockRecorder")
        return MockRecorder(width=min(width, 320), height=min(height, 240), fps=fps)
    return RealSenseRecorder(width=width, height=height, fps=fps, serial=cfg.get("serial"))
