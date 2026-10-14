"""Write the two README charts from the published CPU benchmark run."""

import os

import matplotlib.pyplot as plt

# Rounded values from BENCHMARK_RESULTS.md (2026-10-13 CPU run).
os.makedirs("docs/figures", exist_ok=True)

io_modes = ["Disk (Unbuffered)", "Memory Map", "RAM Cache"]
throughput = [686, 12132, 29277]
colors = ["#888888", "#2b5c8f", "#1b8a5a"]

plt.figure(figsize=(7, 4.2))
bars = plt.bar(io_modes, throughput, color=colors, width=0.55)
plt.title("Surgical Vision Frame Ingestion Throughput (Patches/sec)", fontsize=12, fontweight="bold", pad=12)
plt.ylabel("Throughput (patches/sec)", fontsize=10)
plt.ylim(0, max(throughput) * 1.18)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 600,
        f"{int(yval):,} p/s",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("docs/figures/io_throughput_benchmark.png", dpi=300)
plt.close()

workloads = ["Seq (Batch 1)", "8 Clients (No Batching)", "8 Clients (Dynamic Batching)"]
p99_latency = [9.7, 33.3, 24.4]
bar_colors = ["#2b5c8f", "#d9534f", "#1b8a5a"]

plt.figure(figsize=(7.5, 4.2))
bars2 = plt.bar(workloads, p99_latency, color=bar_colors, width=0.55)
plt.axhline(y=30.0, color="r", linestyle="--", linewidth=1.5, label="Intraoperative SLA Budget (30 ms)")
plt.title("Serving Latency Verification (P99 under Simulated Load)", fontsize=12, fontweight="bold", pad=12)
plt.ylabel("P99 Latency (ms)", fontsize=10)
plt.ylim(0, 48)
plt.legend(loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars2:
    yval = bar.get_height()
    status = " (FAIL)" if yval > 30 else " (PASS)"
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 1.0,
        f"{yval:.1f} ms{status}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("docs/figures/serving_latency_sla.png", dpi=300)
plt.close()

print("Plots successfully saved in docs/figures/")
