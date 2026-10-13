"""Latency profiles for sequential inference and dynamic batching."""

from __future__ import annotations

import threading
import time

import numpy as np
import torch
from torch import nn

from .triton_client_mock import TritonClientMock, measure_inline


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _load_clients(
    client: TritonClientMock,
    inputs: torch.Tensor,
    concurrency: int,
) -> tuple[list[float], list[float], list[int], float]:
    count = inputs.shape[0]
    latencies = [0.0] * count
    compute = [0.0] * count
    batch_sizes = [0] * count
    next_index = 0
    lock = threading.Lock()
    errors: list[BaseException] = []

    def worker() -> None:
        nonlocal next_index
        while True:
            with lock:
                if next_index >= count:
                    return
                index = next_index
                next_index += 1
            try:
                latency_ms, compute_ms, batch_size = client.infer(inputs[index])
            except BaseException as exc:
                errors.append(exc)
                return
            latencies[index] = latency_ms
            compute[index] = compute_ms
            batch_sizes[index] = batch_size

    workers = concurrency if concurrency < count else count
    threads = [threading.Thread(target=worker) for _ in range(workers)]
    wall_started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_s = time.perf_counter() - wall_started
    if errors:
        raise errors[0]
    if any(size == 0 for size in batch_sizes):
        raise RuntimeError("a request finished without a batch size")
    return latencies, compute, batch_sizes, count / elapsed_s


def _row(
    *,
    mode: str,
    concurrency: int,
    max_batch_size: int,
    max_queue_delay_ms: float,
    latencies: list[float],
    compute: list[float] | None,
    batch_sizes: list[int] | None,
    throughput: float,
    target_sla_ms: float,
    num_requests: int,
) -> dict:
    stats = _percentiles(latencies)
    mean_batch = 1.0 if batch_sizes is None else float(np.mean(batch_sizes))
    mean_compute = stats["mean_ms"] if compute is None else float(np.mean(compute))
    stats.update(
        {
            "mode": mode,
            "concurrency": concurrency,
            "max_batch_size": max_batch_size,
            "max_queue_delay_ms": max_queue_delay_ms,
            "num_requests": num_requests,
            "mean_batch_size": mean_batch,
            "mean_compute_ms": mean_compute,
            "throughput_requests_per_sec": float(throughput),
            "target_sla_ms": target_sla_ms,
            "sla_met": bool(stats["p99_ms"] < target_sla_ms),
        }
    )
    return stats


def run_serving_benchmark(
    model: nn.Module,
    device: torch.device,
    num_requests: int = 64,
    input_size: int = 128,
    target_sla_ms: float = 30.0,
    seed: int = 0,
) -> dict:
    """Profile inline sequential latency and closed-loop client concurrency.

    Concurrency is the number of client workers, each with one request in
    flight, served by a single model instance.
    """
    if num_requests < 8:
        raise ValueError("num_requests must be at least 8 so percentiles are defined")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = torch.rand(num_requests, 3, input_size, input_size, generator=generator)
    inputs = inputs.to(device)

    inline = measure_inline(model, inputs)
    inline_throughput = float(num_requests / (sum(inline) / 1000.0))
    rows = [
        _row(
            mode="sequential",
            concurrency=1,
            max_batch_size=1,
            max_queue_delay_ms=0.0,
            latencies=inline,
            compute=None,
            batch_sizes=None,
            throughput=inline_throughput,
            target_sla_ms=target_sla_ms,
            num_requests=num_requests,
        )
    ]

    schedules = (
        {"mode": "queued", "concurrency": 4, "max_batch_size": 1, "max_queue_delay_ms": 0.0},
        {"mode": "queued", "concurrency": 8, "max_batch_size": 1, "max_queue_delay_ms": 0.0},
        {"mode": "dynamic_batch", "concurrency": 1, "max_batch_size": 8, "max_queue_delay_ms": 1.0},
        {"mode": "dynamic_batch", "concurrency": 4, "max_batch_size": 8, "max_queue_delay_ms": 1.0},
        {"mode": "dynamic_batch", "concurrency": 8, "max_batch_size": 8, "max_queue_delay_ms": 1.0},
    )
    for schedule in schedules:
        client = TritonClientMock(
            model,
            device,
            max_batch_size=schedule["max_batch_size"],
            max_queue_delay_ms=schedule["max_queue_delay_ms"],
            target_sla_ms=target_sla_ms,
        )
        client.start()
        try:
            for _ in range(4):
                client.infer(inputs[0])
            latencies, compute, batch_sizes, throughput = _load_clients(
                client,
                inputs,
                schedule["concurrency"],
            )
        finally:
            client.close()
        rows.append(
            _row(
                mode=schedule["mode"],
                concurrency=schedule["concurrency"],
                max_batch_size=schedule["max_batch_size"],
                max_queue_delay_ms=schedule["max_queue_delay_ms"],
                latencies=latencies,
                compute=compute,
                batch_sizes=batch_sizes,
                throughput=throughput,
                target_sla_ms=target_sla_ms,
                num_requests=num_requests,
            )
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    meeting = sum(1 for row in rows if row["sla_met"])
    return {
        "backend": "pytorch_triton_dynamic_batch_mock",
        "backend_note": (
            "Local PyTorch model with a single serving thread, a max batch size, "
            "and a short queue delay. NVIDIA Triton and TensorRT were not executed."
        ),
        "device": str(device),
        "input_shape": [3, input_size, input_size],
        "num_requests": num_requests,
        "target_sla_ms": target_sla_ms,
        "sla_rule": "p99_ms < target_sla_ms",
        "load_model": "closed_loop_clients",
        "configurations_meeting_sla": meeting,
        "configuration_count": len(rows),
        "model": {
            "name": "TinySurgicalNet",
            "num_classes": int(model.head.out_features),
            "parameters": int(parameter_count),
            "onnx_compatible_ops": [
                "Conv",
                "Relu",
                "MaxPool",
                "AdaptiveAvgPool",
                "Dropout",
                "Gemm",
            ],
        },
        "configurations": rows,
    }
