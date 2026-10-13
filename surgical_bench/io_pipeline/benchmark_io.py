"""Compare disk, memory-map, and in-RAM patch streaming."""

from __future__ import annotations

import gc
import shutil
import tempfile
import time
from pathlib import Path

from .patch_streamer import PatchStreamer, write_synthetic_frame


def _remove_tree(path: Path) -> None:
    gc.collect()
    for _ in range(5):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            gc.collect()
            time.sleep(0.2)
    shutil.rmtree(path, ignore_errors=True)


def run_io_benchmark(
    image_size: int = 8192,
    patch_size: int = 256,
    num_patches: int = 64,
    channels: int = 3,
) -> dict:
    """Time patch streaming from a synthetic high-resolution frame.

    Disk reads use unbuffered Python I/O. The file has just been written, so
    the OS page cache may satisfy those reads. The three modes compare access
    paths on that warm file.
    """
    tmp = Path(tempfile.mkdtemp(prefix="surgical_bench_io_"))
    frame_path = tmp / "frame.raw"
    try:
        write_started = time.perf_counter()
        write_synthetic_frame(frame_path, image_size, image_size, channels)
        write_s = time.perf_counter() - write_started
        expected_bytes = image_size * image_size * channels
        actual_bytes = frame_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(f"frame size {actual_bytes} != {expected_bytes}")

        streamer = PatchStreamer(
            frame_path,
            height=image_size,
            width=image_size,
            patch_size=patch_size,
            num_patches=num_patches,
            channels=channels,
        )
        modes = [streamer.benchmark(mode) for mode in ("disk", "memmap", "ram")]
        return {
            "image_height": image_size,
            "image_width": image_size,
            "channels": channels,
            "patch_size": patch_size,
            "num_patches": num_patches,
            "frame_bytes": actual_bytes,
            "frame_write_s": write_s,
            "access_note": (
                "Warm-cache access-path comparison of unbuffered disk reads, "
                "memory-map slices, and in-memory slices. Not a cold-disk measurement."
            ),
            "modes": modes,
        }
    finally:
        _remove_tree(tmp)
