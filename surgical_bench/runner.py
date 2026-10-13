"""Single entry point for the three benchmark tracks."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .closed_loop import run_closed_loop_benchmark
from .io_pipeline import run_io_benchmark
from .report_generator import write_report
from .serving_pipeline import build_model, run_serving_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark synthetic surgical-vision I/O, serving latency, and uncertainty triage."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Required. Use synthetic frames and requests (no clinical data).",
    )
    parser.add_argument(
        "--output",
        default="results/benchmark_report.json",
        help="Path for the JSON metrics file.",
    )
    parser.add_argument(
        "--report",
        default="BENCHMARK_RESULTS.md",
        help="Path for the Markdown summary.",
    )
    parser.add_argument("--image-size", type=int, default=8192)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-patches", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--top-k", type=float, default=0.10)
    parser.add_argument("--sla-ms", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any serving P99 exceeds the SLA. The report is still written.",
    )
    return parser


def collect_environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.synthetic:
        print("Pass --synthetic. This suite does not read clinical images.", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"surgical_bench {__version__}: I/O on {args.image_size}px frames, "
        f"serving and triage on {device}.",
        flush=True,
    )
    print("Track 1/3: patch streaming...", flush=True)
    io_report = run_io_benchmark(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
    )
    print("Track 2/3: serving latency...", flush=True)
    model = build_model(device=device)
    serving_report = run_serving_benchmark(
        model,
        device,
        num_requests=args.num_requests,
        input_size=args.input_size,
        target_sla_ms=args.sla_ms,
        seed=args.seed,
    )
    print("Track 3/3: uncertainty triage...", flush=True)
    closed_report = run_closed_loop_benchmark(
        model,
        device,
        num_frames=args.frames,
        top_k_ratio=args.top_k,
        mc_samples=args.mc_samples,
        input_size=args.input_size,
        seed=args.seed,
    )

    report = {
        "suite": "surgical_bench",
        "version": __version__,
        "schema_version": 1,
        "synthetic": True,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "argv": sys.argv if argv is None else ["surgical_bench.runner", *argv],
        "environment": collect_environment(device),
        "io": io_report,
        "serving": serving_report,
        "closed_loop": closed_report,
    }
    json_path = Path(args.output)
    markdown_path = Path(args.report)
    write_report(report, json_path, markdown_path)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {markdown_path}", flush=True)

    best = max(io_report["modes"], key=lambda row: row["throughput_patches_per_sec"])
    print(
        f"I/O peak: {best['mode']} {best['throughput_patches_per_sec']:.1f} patches/s.",
        flush=True,
    )
    print(
        f"Serving SLA: {serving_report['configurations_meeting_sla']}/"
        f"{serving_report['configuration_count']} configurations under {args.sla_ms:.1f} ms P99.",
        flush=True,
    )
    print(
        f"Triage: ratio {closed_report['triaged_ratio']:.4f}, "
        f"mean uncertainty {closed_report['avg_uncertainty_score']:.4f} nats.",
        flush=True,
    )
    if args.strict and serving_report["configurations_meeting_sla"] != serving_report["configuration_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
