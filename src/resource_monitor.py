from __future__ import annotations

import os
import resource
import subprocess
import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResourceMonitor:
    device: str
    interval_seconds: float = 10.0
    gpu_utilization: list[float] = field(default_factory=list)
    gpu_memory_mb: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.device.startswith("cuda"):
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        logical_index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
        visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        identifier = visible[logical_index] if visible and logical_index < len(visible) else str(logical_index)
        while not self._stop.wait(0 if not self.gpu_utilization else self.interval_seconds):
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi", f"--id={identifier}",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True, timeout=3, stderr=subprocess.DEVNULL,
                )
                utilization, memory = output.strip().splitlines()[0].split(",", 1)
                self.gpu_utilization.append(float(utilization.strip()))
                self.gpu_memory_mb.append(float(memory.strip()))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                return

    def finish(self) -> dict[str, float | int | None]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        torch_peak_gpu_mb = None
        try:
            import torch

            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch_peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
        except (ImportError, RuntimeError, ValueError, AssertionError):
            pass
        values = np.asarray(self.gpu_utilization, dtype=float)
        memory_values = np.asarray(self.gpu_memory_mb, dtype=float)
        nvidia_peak_gpu_mb = float(memory_values.max()) if len(memory_values) else None
        available_peaks = [value for value in (torch_peak_gpu_mb, nvidia_peak_gpu_mb) if value is not None]
        return {
            "peak_gpu_memory_mb": max(available_peaks) if available_peaks else None,
            "torch_peak_gpu_memory_mb": torch_peak_gpu_mb,
            "nvidia_smi_peak_gpu_memory_mb": nvidia_peak_gpu_mb,
            "average_gpu_utilization_percent": float(values.mean()) if len(values) else None,
            "max_gpu_utilization_percent": float(values.max()) if len(values) else None,
            "gpu_utilization_samples": int(len(values)),
            "peak_cpu_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
