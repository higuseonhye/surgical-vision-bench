"""Single-instance dynamic batching stand-in for a Triton client.

The clock is real PyTorch execution time. NVIDIA Triton and TensorRT are not
started. Client latency includes queue wait plus the batch that served the
request.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future

import torch
from torch import nn


class TinySurgicalNet(nn.Module):
    """Small conv net used as an ONNX-exportable serving stand-in.

    Ops are Conv, ReLU, MaxPool, AdaptiveAvgPool, Dropout, and Linear.
    """

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.features(x).flatten(1)
        hidden = self.dropout(hidden)
        return self.head(hidden)


def build_model(num_classes: int = 4, device: torch.device | None = None) -> TinySurgicalNet:
    model = TinySurgicalNet(num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


class TritonClientMock:
    """Queue requests and execute them with a Triton-style batching policy."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_batch_size: int = 8,
        max_queue_delay_ms: float = 1.0,
        target_sla_ms: float = 30.0,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_queue_delay_s = max_queue_delay_ms / 1000.0
        self.target_sla_ms = target_sla_ms
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("client already started")
        self.model.eval()
        self._thread = threading.Thread(target=self._serve, name="triton-mock", daemon=True)
        self._thread.start()

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._queue.put(None)
        thread.join(timeout=60)
        self._thread = None

    def infer(self, tensor: torch.Tensor, timeout_s: float = 120.0) -> tuple[float, float, int]:
        """Return client latency (ms), batch compute (ms), and batch size."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("inference server is not running")
        submitted = time.perf_counter()
        payload = tensor.detach().contiguous()
        future: Future = Future()
        self._queue.put((payload, future, submitted))
        return future.result(timeout=timeout_s)

    def _serve(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            batch = [item]
            deadline = time.perf_counter() + self.max_queue_delay_s
            while len(batch) < self.max_batch_size:
                timeout = deadline - time.perf_counter()
                try:
                    if timeout <= 0:
                        nxt = self._queue.get_nowait()
                    else:
                        nxt = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if nxt is None:
                    self._execute(batch)
                    return
                batch.append(nxt)
            self._execute(batch)

    def _execute(self, batch: list) -> None:
        try:
            tensors = torch.stack([item[0] for item in batch], dim=0)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            compute_started = time.perf_counter()
            with torch.inference_mode():
                self.model(tensors)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            compute_ms = (time.perf_counter() - compute_started) * 1e3
            done = time.perf_counter()
            batch_size = len(batch)
            for _payload, future, submitted in batch:
                if not future.done():
                    latency_ms = (done - submitted) * 1e3
                    future.set_result((latency_ms, compute_ms, batch_size))
        except Exception as exc:
            for _payload, future, _submitted in batch:
                if not future.done():
                    future.set_exception(exc)


def measure_inline(model: nn.Module, inputs: torch.Tensor, warmup: int = 5) -> list[float]:
    """Time one-request forwards on the calling thread."""
    model.eval()
    latencies: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs[:1])
            if inputs.device.type == "cuda":
                torch.cuda.synchronize()
        for index in range(inputs.shape[0]):
            batch = inputs[index : index + 1]
            if inputs.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            model(batch)
            if inputs.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1e3)
    return latencies
