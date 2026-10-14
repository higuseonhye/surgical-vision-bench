"""SurgFlow: local demo for patch ingestion, queue scheduling, and entropy triage."""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "hard_negative_registry.json"
MODEL_CANDIDATES = [
    ROOT.parent / "triton_serving_lab" / "model_repository" / "surgical_feature_net" / "1" / "model.onnx",
    ROOT / "model_repository" / "surgical_feature_net" / "1" / "model.onnx",
]
SLA_MS = 30.0
MAX_BATCH_SIZE = 16
PREFERRED_BATCH_SIZES = (4, 8, 16)
QUEUE_BUDGET_MS = 3.0
LABELS = ("Tumor Margin", "Healthy Tissue", "Vessel", "Instrument")
PUBLISHED_IO = {
    "Disk (Unbuffered)": 686.4,
    "Memory Map (mmap)": 12132.0,
    "RAM Cache": 29277.2,
}
PUBLISHED_IO_ORDER = list(PUBLISHED_IO)
IO_MODE_KEYS = {
    "Disk (Unbuffered)": "disk",
    "Memory Map (mmap)": "memmap",
    "RAM Cache": "ram",
}

st.set_page_config(
    page_title="SurgFlow",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _theme_css(dark: bool) -> str:
    if dark:
        bg, card, text, muted, line, accent, danger = (
            "#101418",
            "#1b242c",
            "#e8eef4",
            "#9aabba",
            "#2c3a46",
            "#3dbea0",
            "#e15b64",
        )
    else:
        bg, card, text, muted, line, accent, danger = (
            "#f3f6f8",
            "#ffffff",
            "#1c2830",
            "#5d6e7a",
            "#d5dee6",
            "#0e6e6a",
            "#b4232c",
        )
    return f"""
    <style>
      .stApp {{ background: {bg}; color: {text}; }}
      h1, h2, h3, p, label, .stMarkdown {{ color: {text}; }}
      div[data-testid="stMetric"] {{
        background: {card};
        border: 1px solid {line};
        border-radius: 12px;
        padding: 12px 16px;
      }}
      .kpi-note {{ color: {muted}; font-size: 0.92rem; }}
      .pass-banner {{
        background: #e5f6f1; color: #0d5c45; border: 1px solid #9ed9c8;
        border-radius: 10px; padding: 12px 14px; font-weight: 650;
      }}
      .fail-banner {{
        background: #fdecee; color: #8d1d28; border: 1px solid #f3b4ba;
        border-radius: 10px; padding: 12px 14px; font-weight: 650;
      }}
      .frame-cell {{
        border-radius: 8px; padding: 8px 6px; text-align: center;
        font-size: 12px; margin-bottom: 4px; border: 1px solid {line};
      }}
      .review-badge {{ color: {danger}; font-weight: 700; font-size: 11px; }}
    </style>
    """


def _init_state() -> None:
    st.session_state.setdefault("dark", False)
    st.session_state.setdefault("serving", None)
    st.session_state.setdefault("io_live", None)
    st.session_state.setdefault("selected_frame", None)
    if "cohort" not in st.session_state:
        st.session_state["cohort"] = _make_cohort(64, seed=0)


def _largest_preferred(count: int) -> int | None:
    fit = [size for size in PREFERRED_BATCH_SIZES if size <= count <= MAX_BATCH_SIZE]
    return max(fit) if fit else None


class _BatchServer:
    def __init__(self, session, input_name: str, mode: str) -> None:
        self.session = session
        self.input_name = input_name
        self.mode = mode
        self.request_queue: queue.Queue = queue.Queue()
        self.running = True
        self.batch_sizes: list[int] = []
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def _collect_baseline(self):
        start_wait = time.perf_counter()
        try:
            first = self.request_queue.get(timeout=0.05)
        except queue.Empty:
            return None
        batch = [first]
        while len(batch) < MAX_BATCH_SIZE:
            elapsed_ms = (time.perf_counter() - start_wait) * 1000.0
            remaining = (QUEUE_BUDGET_MS - elapsed_ms) / 1000.0
            if remaining <= 0:
                break
            try:
                batch.append(self.request_queue.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _collect_early(self):
        try:
            first = self.request_queue.get(timeout=0.05)
        except queue.Empty:
            return None
        batch = [first]
        deadline = time.perf_counter() + QUEUE_BUDGET_MS / 1000.0
        while len(batch) < MAX_BATCH_SIZE:
            available = min(MAX_BATCH_SIZE, len(batch) + self.request_queue.qsize())
            target = _largest_preferred(available)
            if target is not None and len(batch) >= target:
                break
            if target is not None:
                try:
                    batch.append(self.request_queue.get_nowait())
                    continue
                except queue.Empty:
                    pass
            if time.perf_counter() >= deadline:
                break
            try:
                batch.append(self.request_queue.get_nowait())
            except queue.Empty:
                if _largest_preferred(len(batch)) is not None:
                    break
                time.sleep(0)
        return batch

    def _loop(self) -> None:
        collect = self._collect_early if self.mode == "early" else self._collect_baseline
        while self.running:
            batch = collect()
            if not batch:
                continue
            self.batch_sizes.append(len(batch))
            stacked = np.concatenate([item["frame"] for item in batch], axis=0)
            try:
                self.session.run(None, {self.input_name: stacked})
            except Exception as exc:  # noqa: BLE001
                for item in batch:
                    item["error"] = exc
                    item["event"].set()
                continue
            for item in batch:
                item["event"].set()

    def predict(self, frame: np.ndarray) -> float:
        started = time.perf_counter()
        item = {"frame": frame, "event": threading.Event(), "error": None}
        self.request_queue.put(item)
        if not item["event"].wait(timeout=30.0):
            raise TimeoutError("scheduler did not answer within 30 s")
        if item["error"] is not None:
            raise item["error"]
        return (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        self.running = False
        self.worker.join(timeout=1.0)


def _resolve_model() -> Path:
    for path in MODEL_CANDIDATES:
        if path.is_file():
            return path
    destination = ROOT / "model_repository" / "surgical_feature_net" / "1" / "model.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    import torch
    from torch import nn

    class SurgicalFeatureNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(64, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(torch.flatten(self.features(x), 1))

    model = SurgicalFeatureNet().eval()
    torch.onnx.export(
        model,
        torch.randn(1, 3, 128, 128),
        str(destination),
        opset_version=18,
        input_names=["input_frame"],
        output_names=["logits"],
        dynamic_axes={"input_frame": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    return destination


@st.cache_resource(show_spinner=False)
def _session():
    import onnxruntime as ort

    path = _resolve_model()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name, path


def _run_serving(mode: str, concurrency: int, num_requests: int) -> dict:
    session, input_name, model_path = _session()
    server = _BatchServer(session, input_name, mode)
    frame = np.random.randn(1, 3, 128, 128).astype(np.float32)
    latencies = [0.0] * num_requests
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(server.predict, frame): index for index in range(num_requests)
            }
            for future in as_completed(futures):
                latencies[futures[future]] = float(future.result())
    finally:
        server.close()
    samples = np.asarray(latencies, dtype=np.float64)
    mean_batch = float(np.mean(server.batch_sizes)) if server.batch_sizes else 0.0
    p99 = float(np.percentile(samples, 99))
    return {
        "mode": mode,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": p99,
        "mean_batch": mean_batch,
        "sla_met": p99 < SLA_MS,
        "latencies": samples,
        "model_path": str(model_path),
    }


def _entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _make_cohort(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    hard = np.zeros(n, dtype=bool)
    hard[: n // 2] = True
    rng.shuffle(hard)
    logits = np.zeros((n, 4), dtype=np.float64)
    classes = rng.integers(0, 4, size=n)
    logits[np.arange(n), classes] = 6.0
    logits[hard] = rng.normal(0.0, 0.05, size=(int(hard.sum()), 4))
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    scores = _entropy(probs)
    return {"hard": hard, "probs": probs, "scores": scores}


def _triage_ids(scores: np.ndarray, ratio: float) -> np.ndarray:
    k = min(len(scores), int(np.ceil(len(scores) * ratio)))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    return np.argsort(scores)[::-1][:k]


def _load_registry() -> list[dict]:
    if not REGISTRY_PATH.is_file():
        return []
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _append_registry(entry: dict) -> int:
    rows = _load_registry()
    rows.append(entry)
    REGISTRY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def _style_axes(ax, dark: bool) -> None:
    text = "#e8eef4" if dark else "#1c2830"
    ax.set_facecolor("#1b242c" if dark else "#ffffff")
    ax.tick_params(colors=text)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    ax.title.set_color(text)
    for spine in ax.spines.values():
        spine.set_color("#5d6e7a")


def _io_chart(values: dict[str, float], title: str, dark: bool):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = PUBLISHED_IO_ORDER
    heights = [values[label] for label in labels]
    colors = ["#8aa0b2", "#1f6f8b", "#0e6e6a"]
    bars = ax.bar(labels, heights, color=colors, width=0.62)
    ax.set_ylabel("patches/s")
    ax.set_title(title)
    ax.set_ylim(0, max(heights) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, height in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _style_axes(ax, dark)
    fig.patch.set_facecolor("#101418" if dark else "#f3f6f8")
    fig.tight_layout()
    return fig


def _latency_chart(samples: np.ndarray, p99: float, dark: bool):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hist(samples, bins=24, color="#1f6f8b", edgecolor="white")
    ax.axvline(SLA_MS, color="#b4232c", linestyle="--", label="30 ms budget")
    ax.axvline(p99, color="#0e6e6a", linestyle="-", label=f"P99 {p99:.2f} ms")
    ax.set_xlabel("end-to-end latency (ms)")
    ax.set_ylabel("requests")
    ax.set_title("Latency distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    _style_axes(ax, dark)
    fig.patch.set_facecolor("#101418" if dark else "#f3f6f8")
    fig.tight_layout()
    return fig


def _gauge(p99: float, dark: bool):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.15, 1.15)
    theta = np.linspace(np.pi, 0, 80)
    ax.plot(np.cos(theta), np.sin(theta), color="#d5dee6", linewidth=10, solid_capstyle="round")
    span = min(p99 / 60.0, 1.0)
    mark = np.linspace(np.pi, np.pi - span * np.pi, 40)
    color = "#0e6e6a" if p99 < SLA_MS else "#b4232c"
    ax.plot(np.cos(mark), np.sin(mark), color=color, linewidth=10, solid_capstyle="round")
    ax.text(0, 0.15, f"{p99:.2f} ms", ha="center", va="center", fontsize=14, color=color)
    ax.text(0, -0.05, "P99", ha="center", va="center", fontsize=9, color="#5d6e7a")
    ax.axis("off")
    fig.patch.set_facecolor("#101418" if dark else "#f3f6f8")
    fig.tight_layout()
    return fig


def _render_header() -> None:
    dark = st.toggle("Dark theme", value=st.session_state["dark"])
    st.session_state["dark"] = dark
    st.markdown(_theme_css(dark), unsafe_allow_html=True)
    st.markdown("# SurgFlow: Low-Latency Intraoperative Perception & HITL Triage Platform")
    st.caption(
        "Local CPU demo. The 30 ms line is a lab budget. Triton Server is not running."
    )
    left, mid, right = st.columns(3)
    left.metric("Target SLA budget", "< 30.0 ms", "Intraoperative CPU lab budget")
    mid.metric("Peak I/O throughput", "29,277 patches/s (RAM)", "12,132 patches/s (mmap)")
    right.metric("Triage efficiency", "7 / 64 hard frames", "H(p) ≈ 1.39 nats")


def _tab_serving(dark: bool) -> None:
    st.subheader("Dynamic batching and serving latency")
    concurrency = st.select_slider("Concurrency level", options=[1, 4, 8, 16], value=8)
    requests = st.slider("Number of requests", min_value=30, max_value=200, value=120, step=10)
    mode_label = st.radio(
        "Scheduler mode",
        [
            "Baseline (Fixed Queue Delay)",
            "Proposed (Early Trigger on Preferred Batches [4, 8, 16] + perf_counter Deadline)",
        ],
    )
    mode = "baseline" if mode_label.startswith("Baseline") else "early"
    if st.button("Run Serving Benchmark", type="primary"):
        with st.spinner("Running the local ONNX Runtime queue..."):
            st.session_state["serving"] = _run_serving(mode, int(concurrency), int(requests))

    result = st.session_state["serving"]
    if result:
        c1, c2, c3, c4, gauge = st.columns([1, 1, 1, 1, 1.3])
        c1.metric("P50", f"{result['p50_ms']:.2f} ms")
        c2.metric("P95", f"{result['p95_ms']:.2f} ms")
        c3.metric("P99", f"{result['p99_ms']:.2f} ms")
        c4.metric("Mean batch", f"{result['mean_batch']:.2f}")
        banner = "pass-banner" if result["sla_met"] else "fail-banner"
        text = (
            "[PASS - Within 30ms SLA]"
            if result["sla_met"]
            else "[FAIL - Exceeded 30ms SLA]"
        )
        st.markdown(f'<div class="{banner}">{text}</div>', unsafe_allow_html=True)
        chart_col, gauge_col = st.columns([2, 1])
        chart_col.pyplot(_latency_chart(result["latencies"], result["p99_ms"], dark))
        gauge_col.pyplot(_gauge(result["p99_ms"], dark))
        st.caption(f"Model: `{result['model_path']}`")

    st.info(
        "On the published baseline run, concurrency 8 reached P99 47.59 ms because the "
        "scheduler waited out the delay unless 16 requests were queued. Mean batch was "
        "7.5, so one request missed the batch and paid that cycle plus its own wait. "
        "The Windows queue timeout also slept longer than 3 ms. Early dispatch at "
        "preferred sizes 4, 8, and 16, with a perf_counter deadline, brought that "
        "concurrency-8 P99 to 7.01 ms (mean batch 3.64). A new button press measures "
        "this machine again. It will not reprint those exact figures."
    )


def _tab_io(dark: bool) -> None:
    st.subheader("High-resolution frame ingestion")
    mode = st.selectbox("Ingestion mode", PUBLISHED_IO_ORDER)
    patch_size = st.number_input("Patch size", min_value=32, max_value=1024, value=256, step=32)
    st.pyplot(
        _io_chart(
            PUBLISHED_IO,
            "Published warm-cache run (64 patches, 8192 frame)",
            dark,
        )
    )
    st.caption(
        "mmap was 17.7× the unbuffered disk path (12,132 vs 686 patches/s). "
        "The mapping avoids a separate kernel-to-user read for each row. "
        "This benchmark still copies the window into a contiguous tensor before "
        "returning it. RAM was 29,277 patches/s. The file had just been written, "
        "so the page cache may have served the reads."
    )
    if st.button("Benchmark Ingestion Throughput", type="primary"):
        from surgical_bench.io_pipeline.benchmark_io import run_io_benchmark

        with st.spinner("Streaming patches from a synthetic 8192 frame..."):
            try:
                report = run_io_benchmark(
                    image_size=8192,
                    patch_size=int(patch_size),
                    num_patches=32,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
        st.session_state["io_live"] = report

    live = st.session_state["io_live"]
    if live:
        live_values = {
            "Disk (Unbuffered)": next(row["throughput_patches_per_sec"] for row in live["modes"] if row["mode"] == "disk"),
            "Memory Map (mmap)": next(row["throughput_patches_per_sec"] for row in live["modes"] if row["mode"] == "memmap"),
            "RAM Cache": next(row["throughput_patches_per_sec"] for row in live["modes"] if row["mode"] == "ram"),
        }
        st.pyplot(
            _io_chart(
                live_values,
                f"This session (32 patches, patch {int(patch_size)})",
                dark,
            )
        )
        selected = next(
            row for row in live["modes"] if row["mode"] == IO_MODE_KEYS[mode]
        )
        st.metric(
            f"{mode} this session",
            f"{selected['throughput_patches_per_sec']:,.0f} patches/s",
            f"mean {selected['io_latency_ms']:.3f} ms",
        )


def _tab_triage() -> None:
    st.subheader("Closed-loop uncertainty triage")
    ratio = st.slider("Triage threshold ratio (top K%)", min_value=5, max_value=50, value=10, step=5) / 100.0
    cohort = st.session_state["cohort"]
    scores = cohort["scores"]
    chosen = set(int(i) for i in _triage_ids(scores, ratio))
    st.caption(
        "H(p) = -sum p_i ln(p_i). A flat 4-class distribution has ln(4) = 1.386 nats. "
        f"This board has {int(cohort['hard'].sum())} near-uniform frames and "
        f"{int((~cohort['hard']).sum())} peaked frames."
    )

    for row in range(8):
        columns = st.columns(8)
        for col in range(8):
            index = row * 8 + col
            score = float(scores[index])
            triaged = index in chosen
            tone = "#f8d0d3" if triaged else ("#d9efe8" if score < 0.5 else "#f4ead2")
            badge = '<div class="review-badge">Review Required</div>' if triaged else ""
            with columns[col]:
                st.markdown(
                    f'<div class="frame-cell" style="background:{tone}">'
                    f"#{index}<br>{score:.3f}{badge}</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Open", key=f"open_{index}"):
                    st.session_state["selected_frame"] = index

    selected = st.session_state.get("selected_frame")
    st.markdown("### Reviewer action")
    if selected is None:
        st.write("Open a frame from the grid.")
        return
    score = float(scores[selected])
    st.write(
        f"Frame {selected}: H(p) = {score:.4f} nats. "
        f"{'Near-uniform control.' if cohort['hard'][selected] else 'Peaked control.'}"
    )
    if selected not in chosen:
        st.warning("This frame is outside the current top-K review set.")
        return
    label = st.radio("Verified ground truth", LABELS, horizontal=True)
    if st.button("Commit to Retraining Pool", type="primary"):
        count = _append_registry(
            {
                "frame_id": int(selected),
                "entropy_nats": score,
                "label": label,
                "labeled_hard": bool(cohort["hard"][selected]),
                "committed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        st.success(
            f"Frame {selected} labeled {label} and appended to "
            f"`hard_negative_registry.json` ({count} rows)."
        )


def main() -> None:
    _init_state()
    _render_header()
    serving, ingestion, triage = st.tabs(
        [
            "Dynamic Batching & Serving Latency",
            "High-Resolution Frame Ingestion",
            "Closed-Loop Uncertainty Triage",
        ]
    )
    dark = st.session_state["dark"]
    with serving:
        _tab_serving(dark)
    with ingestion:
        _tab_io(dark)
    with triage:
        _tab_triage()


main()
