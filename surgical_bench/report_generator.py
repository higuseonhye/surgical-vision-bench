"""Write JSON metrics and a Markdown summary."""

from __future__ import annotations

import json
from pathlib import Path


NOTES = [
    "Frames, requests, and logits are synthetic. This is a software benchmark, not a clinical validation and not a surgical device.",
    "I/O timings compare unbuffered disk reads, memory-map slices, and in-RAM slices on a frame this process just wrote. The OS page cache may serve those reads.",
    "Serving latency is measured on a local PyTorch model with a Triton-style dynamic batching policy. NVIDIA Triton and TensorRT were not executed.",
    "The SLA check is P99 client latency against the configured budget (default 30 ms).",
    "The review queue ranks a synthetic logit mix (peaked vs near-uniform) by predictive entropy. Monte Carlo dropout on the untrained network is reported separately and sits near the uniform ceiling.",
]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{float(value):.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def render_markdown(report: dict) -> str:
    io_report = report["io"]
    serving = report["serving"]
    closed = report["closed_loop"]
    env = report["environment"]

    best_io = max(io_report["modes"], key=lambda row: row["throughput_patches_per_sec"])
    sla = serving["target_sla_ms"]
    meeting = serving["configurations_meeting_sla"]
    total = serving["configuration_count"]

    io_rows = [
        [
            row["mode"],
            str(row["num_patches"]),
            _fmt(row["throughput_patches_per_sec"], 1),
            _fmt(row["throughput_mb_per_sec"], 1),
            _fmt(row["io_latency_ms"]),
            _fmt(row["p50_ms"]),
            _fmt(row["p95_ms"]),
            _fmt(row["p99_ms"]),
        ]
        for row in io_report["modes"]
    ]
    serving_rows = [
        [
            row["mode"],
            str(row["concurrency"]),
            str(row["max_batch_size"]),
            _fmt(row["mean_batch_size"], 2),
            _fmt(row["p50_ms"]),
            _fmt(row["p95_ms"]),
            _fmt(row["p99_ms"]),
            _fmt(row["throughput_requests_per_sec"], 1),
            "pass" if row["sla_met"] else "fail",
        ]
        for row in serving["configurations"]
    ]
    review_rows = [
        [str(item["frame_id"]), _fmt(item["uncertainty"], 4), "yes" if item.get("labeled_hard") else "no"]
        for item in closed["review_order"]
    ]
    model_scores = closed["model_mc_dropout"]

    lines = [
        "# Surgical vision benchmark results",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Synthetic proxy for high-throughput frame I/O, low-latency model serving, and uncertainty triage. "
        f"Highest measured patch throughput was **{best_io['mode']}** at "
        f"{best_io['throughput_patches_per_sec']:.1f} patches/s. "
        f"{meeting} of {total} serving configurations had P99 latency under {sla:.1f} ms.",
        "",
        "## Environment",
        "",
        _table(
            ["python", "torch", "numpy", "device", "torch threads", "platform"],
            [[
                env["python"],
                env["torch"],
                env["numpy"],
                env["device"],
                str(env["torch_num_threads"]),
                env["platform"],
            ]],
        ),
        "",
        "## Track 1 — I/O",
        "",
        f"Frame {io_report['image_height']}×{io_report['image_width']}×{io_report['channels']} "
        f"uint8 ({io_report['frame_bytes']} bytes), patch {io_report['patch_size']}, "
        f"{io_report['num_patches']} patches per mode. "
        f"Frame creation took {io_report['frame_write_s']:.2f} s and is excluded from the timings.",
        "",
        _table(
            [
                "mode",
                "patches",
                "throughput_patches_per_sec",
                "throughput_MB_per_sec",
                "io_latency_ms",
                "p50_ms",
                "p95_ms",
                "p99_ms",
            ],
            io_rows,
        ),
        "",
        "## Track 2 — Serving",
        "",
        f"Backend `{serving['backend']}` on `{serving['device']}`, model "
        f"{serving['model']['name']} ({serving['model']['parameters']} parameters), "
        f"input {serving['input_shape']}, {serving['num_requests']} requests. "
        f"SLA rule: `{serving['sla_rule']}` with target {sla:.1f} ms. "
        "Sequential mode is an in-process batch-1 loop. Queued and dynamic-batch rows are client latency "
        "against one model instance; concurrency is the number of workers with one request in flight.",
        "",
        _table(
            [
                "mode",
                "concurrency",
                "max_batch",
                "mean_batch",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "requests_per_sec",
                "sla",
            ],
            serving_rows,
        ),
        "",
        "## Track 3 — Closed loop",
        "",
        f"Score `{closed['formula']}` ({closed['entropy_units']}) on synthetic logits: "
        f"{closed['hard_frames']} near-uniform frames and {closed['easy_frames']} peaked frames. "
        f"Top {closed['top_k_ratio']:.0%} were dispatched "
        f"({closed['triaged_count']} frames, triaged_ratio {closed['triaged_ratio']:.4f}; "
        f"{closed['triaged_hard_count']} of those were labeled near-uniform). "
        f"Mean entropy was {closed['avg_uncertainty_score']:.4f} nats "
        f"(peaked frames {closed['easy_mean_uncertainty']:.4f}, "
        f"near-uniform frames {closed['hard_mean_uncertainty']:.4f}).",
        "",
        _table(
            [
                "frames",
                "hard_frames",
                "top_k_ratio",
                "triaged_count",
                "triaged_ratio",
                "avg_uncertainty_score",
                "avg_triaged_uncertainty",
                "queue_depth",
            ],
            [[
                str(closed["num_frames"]),
                str(closed["hard_frames"]),
                _fmt(closed["top_k_ratio"], 2),
                str(closed["triaged_count"]),
                _fmt(closed["triaged_ratio"], 4),
                _fmt(closed["avg_uncertainty_score"], 4),
                _fmt(closed["avg_triaged_uncertainty"], 4),
                str(closed["queue_depth_after_dispatch"]),
            ]],
        ),
        "",
        "Review dispatch order (highest uncertainty first):",
        "",
        _table(["frame_id", "uncertainty", "labeled_hard"], review_rows) if review_rows else "_No frames dispatched._",
        "",
        f"Untrained `{serving['model']['name']}` Monte Carlo dropout "
        f"({model_scores['mc_samples']} samples, {model_scores['num_frames']} random inputs) "
        f"averaged {model_scores['avg_uncertainty_score']:.4f} nats "
        f"(range {model_scores['uncertainty_min']:.4f}–{model_scores['uncertainty_max']:.4f}; "
        f"uniform ceiling {model_scores['uniform_ceiling_nats']:.4f}).",
        "",
        _table(
            ["mc_samples", "avg_uncertainty_score", "min", "max", "uniform_ceiling"],
            [[
                str(model_scores["mc_samples"]),
                _fmt(model_scores["avg_uncertainty_score"], 4),
                _fmt(model_scores["uncertainty_min"], 4),
                _fmt(model_scores["uncertainty_max"], 4),
                _fmt(model_scores["uniform_ceiling_nats"], 4),
            ]],
        ),
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in report.get("notes", NOTES))
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict, json_path: Path, markdown_path: Path) -> None:
    document = dict(report)
    document["notes"] = list(NOTES)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(document), encoding="utf-8")
