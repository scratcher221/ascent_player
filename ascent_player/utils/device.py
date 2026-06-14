from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from ascent_player.config import DeviceMode
from ascent_player.utils.gpu_env import bootstrap_gpu_environment, nvidia_library_dirs


class GpuNotAvailableError(RuntimeError):
    """Raised when GPU training was requested but TensorFlow cannot use a GPU."""


@dataclass(slots=True)
class DeviceInfo:
    mode: DeviceMode
    training_device: str
    inference_device: str
    gpu_available: bool
    gpu_names: list[str]
    message: str


def import_tensorflow(device_mode: DeviceMode):
    if device_mode == DeviceMode.CPU:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    else:
        bootstrap_gpu_environment()

    import tensorflow as tf

    return tf


def resolve_device(device_mode: DeviceMode) -> DeviceInfo:
    tf = import_tensorflow(device_mode)
    gpus = tf.config.list_physical_devices("GPU")

    gpu_names: list[str] = []
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
        gpu_names.append(gpu.name)

    if device_mode != DeviceMode.CPU and not gpus:
        lib_dirs = nvidia_library_dirs()
        preloaded = bootstrap_gpu_environment()
        hint = (
            "TensorFlow could not load the GPU. On Linux with pip CUDA wheels, "
            "NVIDIA libraries must be on LD_LIBRARY_PATH before TensorFlow imports. "
            "Reinstall with: pip install 'tensorflow[and-cuda]>=2.15' "
            "and ensure the NVIDIA driver works (nvidia-smi)."
        )
        if lib_dirs:
            hint += f" Found {len(lib_dirs)} NVIDIA lib dirs, preloaded {preloaded} libraries."
        else:
            hint += " No nvidia-* pip packages found in the active environment."
        raise GpuNotAvailableError(hint)

    if device_mode == DeviceMode.CPU or not gpus:
        return DeviceInfo(
            mode=device_mode,
            training_device="/CPU:0",
            inference_device="/CPU:0",
            gpu_available=bool(gpus),
            gpu_names=gpu_names,
            message="CPU",
        )

    return DeviceInfo(
        mode=device_mode,
        training_device="/GPU:0",
        inference_device="/GPU:0",
        gpu_available=True,
        gpu_names=gpu_names,
        message=f"GPU: {gpu_names[0]}",
    )


def benchmark_inference_device(model, sample: np.ndarray, info: DeviceInfo) -> str:
    if info.mode == DeviceMode.CPU or not info.gpu_available:
        return info.inference_device
    return "/GPU:0"


def _time_forward(tf, model, sample: np.ndarray, device: str) -> float:
    tensor = tf.convert_to_tensor(sample[None, ...], dtype=tf.float32)
    with tf.device(device):
        for _ in range(3):
            _ = model(tensor, training=False)
        start = time.perf_counter()
        for _ in range(10):
            _ = model(tensor, training=False)
        _ = model(tensor, training=False).numpy()
    return (time.perf_counter() - start) * 1000 / 10
