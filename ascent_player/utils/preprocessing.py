from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ascent_player.config import ObservationConfig


def preprocess_frame(frame_rgb: np.ndarray, config: ObservationConfig) -> np.ndarray:
    if frame_rgb.ndim != 3:
        raise ValueError(f"Expected RGB frame, got shape {frame_rgb.shape}")
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(
        gray,
        (config.width, config.height),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float32) / 255.0


class FrameStack:
    def __init__(self, size: int) -> None:
        self.size = size
        self.frames: deque[np.ndarray] = deque(maxlen=size)

    def clear(self) -> None:
        self.frames.clear()

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.size):
            self.frames.append(frame.copy())
        return self.array()

    def append(self, frame: np.ndarray) -> np.ndarray:
        if not self.frames:
            return self.reset(frame)
        self.frames.append(frame)
        return self.array()

    def array(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("Frame stack is empty.")
        return np.stack(tuple(self.frames), axis=-1).astype(np.float32)


def qimage_bytes_from_frame(frame_rgb: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise RuntimeError("Failed to encode preview frame.")
    return encoded.tobytes()
