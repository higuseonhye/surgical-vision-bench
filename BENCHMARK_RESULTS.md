# Surgical vision benchmark results

Generated: 2026-10-13T16:04:13+09:00

Synthetic proxy for high-throughput frame I/O, low-latency model serving, and uncertainty triage. Highest measured patch throughput was **ram** at 29277.2 patches/s. 5 of 6 serving configurations had P99 latency under 30.0 ms.

## Environment

| python | torch | numpy | device | torch threads | platform |
| --- | --- | --- | --- | --- | --- |
| 3.14.3 | 2.13.0+cpu | 2.4.3 | cpu | 6 | Windows-11-10.0.26200-SP0 |

## Track 1 — I/O

Frame 8192×8192×3 uint8 (201326592 bytes), patch 256, 64 patches per mode. Frame creation took 1.64 s and is excluded from the timings.

| mode | patches | throughput_patches_per_sec | throughput_MB_per_sec | io_latency_ms | p50_ms | p95_ms | p99_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| disk | 64 | 686.4 | 128.7 | 1.457 | 1.390 | 1.776 | 2.156 |
| memmap | 64 | 12132.0 | 2274.8 | 0.082 | 0.021 | 0.412 | 0.417 |
| ram | 64 | 29277.2 | 5489.5 | 0.034 | 0.033 | 0.060 | 0.077 |

## Track 2 — Serving

Backend `pytorch_triton_dynamic_batch_mock` on `cpu`, model TinySurgicalNet (5220 parameters), input [3, 128, 128], 64 requests. SLA rule: `p99_ms < target_sla_ms` with target 30.0 ms. Sequential mode is an in-process batch-1 loop. Queued and dynamic-batch rows are client latency against one model instance; concurrency is the number of workers with one request in flight.

| mode | concurrency | max_batch | mean_batch | p50_ms | p95_ms | p99_ms | requests_per_sec | sla |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequential | 1 | 1 | 1.00 | 1.923 | 2.627 | 9.725 | 452.2 | pass |
| queued | 4 | 1 | 1.00 | 11.029 | 13.907 | 14.571 | 341.3 | pass |
| queued | 8 | 1 | 1.00 | 24.757 | 32.487 | 33.329 | 308.6 | fail |
| dynamic_batch | 1 | 8 | 1.00 | 14.798 | 16.470 | 20.136 | 75.2 | pass |
| dynamic_batch | 4 | 8 | 4.00 | 14.762 | 21.628 | 21.957 | 262.0 | pass |
| dynamic_batch | 8 | 8 | 7.50 | 16.461 | 22.996 | 24.384 | 410.8 | pass |

## Track 3 — Closed loop

Score `H(p) = -sum_i p_i log(p_i)` (nats) on synthetic logits: 32 near-uniform frames and 32 peaked frames. Top 10% were dispatched (7 frames, triaged_ratio 0.1094; 7 of those were labeled near-uniform). Mean entropy was 0.7186 nats (peaked frames 0.0517, near-uniform frames 1.3854).

| frames | hard_frames | top_k_ratio | triaged_count | triaged_ratio | avg_uncertainty_score | avg_triaged_uncertainty | queue_depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 32 | 0.10 | 7 | 0.1094 | 0.7186 | 1.3861 | 7 |

Review dispatch order (highest uncertainty first):

| frame_id | uncertainty | labeled_hard |
| --- | --- | --- |
| 19 | 1.3863 | yes |
| 14 | 1.3862 | yes |
| 26 | 1.3862 | yes |
| 30 | 1.3862 | yes |
| 40 | 1.3861 | yes |
| 48 | 1.3860 | yes |
| 57 | 1.3860 | yes |

Untrained `TinySurgicalNet` Monte Carlo dropout (8 samples, 64 random inputs) averaged 1.3761 nats (range 1.3734–1.3782; uniform ceiling 1.3863).

| mc_samples | avg_uncertainty_score | min | max | uniform_ceiling |
| --- | --- | --- | --- | --- |
| 8 | 1.3761 | 1.3734 | 1.3782 | 1.3863 |

## Notes

- Frames, requests, and logits are synthetic. This is a software benchmark, not a clinical validation and not a surgical device.
- I/O timings compare unbuffered disk reads, memory-map slices, and in-RAM slices on a frame this process just wrote. The OS page cache may serve those reads.
- Serving latency is measured on a local PyTorch model with a Triton-style dynamic batching policy. NVIDIA Triton and TensorRT were not executed.
- The SLA check is P99 client latency against the configured budget (default 30 ms).
- The review queue ranks a synthetic logit mix (peaked vs near-uniform) by predictive entropy. Monte Carlo dropout on the untrained network is reported separately and sits near the uniform ceiling.
