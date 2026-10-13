"""Patch extraction from a synthetic high-resolution frame."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch


def write_synthetic_frame(path: Path, height: int, width: int, channels: int = 3) -> None:
    """Write a deterministic uint8 raster used as an 8K / WSI-scale proxy."""
    if channels != 3:
        raise ValueError("synthetic frames are RGB (3 channels)")
    if height < 1 or width < 1:
        raise ValueError("frame dimensions must be positive")

    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(height, width, channels))
    try:
        xs = np.arange(width, dtype=np.uint32)
        for y0 in range(0, height, 256):
            y1 = min(height, y0 + 256)
            rows = np.arange(y0, y1, dtype=np.uint32)[:, None]
            base = ((rows * np.uint32(131) + xs[None, :]) & np.uint32(0xFF)).astype(np.uint8)
            mm[y0:y1, :, 0] = base
            mm[y0:y1, :, 1] = base + np.uint8(40)
            mm[y0:y1, :, 2] = base // np.uint8(2)
        mm.flush()
    finally:
        _close_memmap(mm)


def iter_origins(height: int, width: int, patch_size: int, limit: int):
    produced = 0
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            yield y, x
            produced += 1
            if produced >= limit:
                return


def _close_memmap(mm: np.memmap) -> None:
    mmap_obj = getattr(mm, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _summarize_latencies(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    throughput = float(len(arr) / (arr.sum() / 1000.0))
    return {
        "io_latency_ms": mean_ms,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "throughput_patches_per_sec": throughput,
    }


class PatchStreamer:
    """Stream fixed patches from disk, a memory map, or an in-RAM copy."""

    def __init__(
        self,
        path: Path,
        height: int,
        width: int,
        patch_size: int,
        num_patches: int,
        channels: int = 3,
    ) -> None:
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        if patch_size > height or patch_size > width:
            raise ValueError("patch_size exceeds the frame")
        if num_patches < 1:
            raise ValueError("num_patches must be positive")
        self.path = Path(path)
        self.height = height
        self.width = width
        self.channels = channels
        self.patch_size = patch_size
        self.origins = list(iter_origins(height, width, patch_size, num_patches))
        if len(self.origins) < num_patches:
            raise ValueError(
                f"frame yields {len(self.origins)} patches, fewer than requested {num_patches}"
            )

    def benchmark(self, mode: str) -> dict:
        loaders = {
            "disk": self._benchmark_disk,
            "memmap": self._benchmark_memmap,
            "ram": self._benchmark_ram,
        }
        try:
            loader = loaders[mode]
        except KeyError as exc:
            raise ValueError(f"unknown cache mode: {mode}") from exc
        started = time.perf_counter()
        samples = loader()
        elapsed_s = time.perf_counter() - started
        summary = _summarize_latencies(samples)
        bytes_per_patch = self.patch_size * self.patch_size * self.channels
        summary.update(
            {
                "mode": mode,
                "num_patches": len(samples),
                "patch_size": self.patch_size,
                "elapsed_s": elapsed_s,
                "throughput_mb_per_sec": summary["throughput_patches_per_sec"]
                * bytes_per_patch
                / (1024.0 * 1024.0),
            }
        )
        expected = 1000.0 / summary["io_latency_ms"]
        if abs(summary["throughput_patches_per_sec"] - expected) / expected > 1e-6:
            raise RuntimeError("throughput does not match mean I/O latency")
        return summary

    def _assert_patch(self, tensor: torch.Tensor, y: int, x: int) -> None:
        expected_shape = (self.patch_size, self.patch_size, self.channels)
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.uint8:
            raise RuntimeError(f"unexpected patch tensor {tuple(tensor.shape)} {tensor.dtype}")
        expected = int((np.uint32(y) * np.uint32(131) + np.uint32(x)) & np.uint32(0xFF))
        got = int(tensor[0, 0, 0].item())
        if got != expected:
            raise RuntimeError(f"patch byte mismatch at {(y, x)}: saw {got}, expected {expected}")

    def _time_loads(self, load) -> list[float]:
        for y, x in (self.origins[0], self.origins[-1]):
            self._assert_patch(load(y, x), y, x)
        samples: list[float] = []
        for y, x in self.origins:
            t0 = time.perf_counter()
            load(y, x)
            samples.append((time.perf_counter() - t0) * 1e3)
        return samples

    def _benchmark_disk(self) -> list[float]:
        row_stride = self.width * self.channels
        with self.path.open("rb", buffering=0) as handle:
            def load(y: int, x: int) -> torch.Tensor:
                patch = np.empty((self.patch_size, self.patch_size, self.channels), dtype=np.uint8)
                x_off = x * self.channels
                for row in range(self.patch_size):
                    handle.seek((y + row) * row_stride + x_off)
                    view = patch[row]
                    nread = handle.readinto(view)
                    if nread != view.nbytes:
                        raise OSError(f"short read at {(y, x)} row {row}")
                return torch.from_numpy(patch)

            return self._time_loads(load)

    def _benchmark_memmap(self) -> list[float]:
        mm = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            shape=(self.height, self.width, self.channels),
        )
        try:
            def load(y: int, x: int) -> torch.Tensor:
                window = np.array(
                    mm[y : y + self.patch_size, x : x + self.patch_size, :],
                    copy=True,
                )
                return torch.from_numpy(window)

            return self._time_loads(load)
        finally:
            _close_memmap(mm)

    def _benchmark_ram(self) -> list[float]:
        mm = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            shape=(self.height, self.width, self.channels),
        )
        try:
            ram = np.array(mm, copy=True)
        finally:
            _close_memmap(mm)
        ps = self.patch_size

        def load(y: int, x: int) -> torch.Tensor:
            window = np.ascontiguousarray(ram[y : y + ps, x : x + ps, :])
            return torch.from_numpy(window)

        return self._time_loads(load)
