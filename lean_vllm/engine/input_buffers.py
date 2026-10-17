"""A step's input tensors on the device with no per-step allocation, as vLLM's InputBatch keeps them.

On CUDA, each input uses a reusable pinned host buffer and a device buffer. When more space is needed, both buffers
grow to at least twice their current capacity. Each step writes the NumPy data into the host buffer, then copies it
asynchronously to the device. A CUDA event ensures that the copy finishes before the next step overwrites the host
buffer. On other devices, each step creates tensors directly from its input arrays without reusable staging buffers.
"""

import numpy as np
import torch
from numpy.typing import ArrayLike

NUMPY_DTYPES = {torch.int32: np.int32, torch.int64: np.int64, torch.float32: np.float32}
MIN_CAPACITY = 256


class InputBuffers:
    def __init__(self, device: torch.device):
        self.device = device
        self.staged = device.type == "cuda"
        self._host: dict[str, np.ndarray] = {}  # numpy views of the pinned tensors below
        self._pinned: dict[str, torch.Tensor] = {}
        self._device: dict[str, torch.Tensor] = {}
        self._copied: torch.cuda.Event | None = None

    def begin(self):
        """Wait for the previous step's copies to finish before reusing the host buffers."""
        if self._copied is not None:
            self._copied.synchronize()  # copies may still be reading the host buffers; wait before overwriting them
            self._copied = None

    def end(self):
        """Call after the step's final put() to record a CUDA event marking completion of its host-to-device copies."""
        if self.staged:
            self._copied = torch.cuda.Event()
            self._copied.record()

    def put(self, name: str, values: ArrayLike, dtype: torch.dtype) -> torch.Tensor:
        """Return the values as a tensor on the configured device.

        On CUDA, the tensor shares a reusable buffer for this name, so a later put() with the same name
        may overwrite its contents.
        """
        values = np.asarray(values, NUMPY_DTYPES[dtype])
        if not self.staged:
            return torch.from_numpy(np.ascontiguousarray(values)).to(self.device)
        n = values.size
        if name not in self._device or self._device[name].numel() < n:
            capacity = max(n, 2 * (self._device[name].numel() if name in self._device else 0), MIN_CAPACITY)
            self._pinned[name] = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=True)
            self._host[name] = self._pinned[name].numpy()
            self._device[name] = torch.empty(capacity, dtype=dtype, device=self.device)
        self._host[name][:n] = values.reshape(-1)
        out = self._device[name][:n]
        out.copy_(self._pinned[name][:n], non_blocking=True)
        return out.view(values.shape)
