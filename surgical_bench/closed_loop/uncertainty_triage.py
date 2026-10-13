"""Predictive-entropy triage with Monte Carlo dropout scores."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .hitl_queue import ClinicianReviewQueue


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """H(p) = -sum_i p_i log(p_i), in nats, for each row of ``probs``."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("probs must have shape (N, C)")
    clipped = np.clip(arr, eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def probabilities_from_model_output(outputs: np.ndarray) -> np.ndarray:
    """Accept a probability matrix or a logit matrix and return probabilities."""
    arr = np.asarray(outputs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("model outputs must have shape (N, C)")
    row_sums = arr.sum(axis=1, keepdims=True)
    looks_like_probs = bool(np.all(arr >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-3))
    if looks_like_probs:
        return arr / row_sums
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _check_entropy_identity() -> None:
    uniform = predictive_entropy(np.full((1, 4), 0.25))[0]
    if not np.isclose(uniform, np.log(4.0)):
        raise RuntimeError(f"uniform entropy {uniform} != log(4)")
    one_hot = predictive_entropy(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64))[0]
    if one_hot > 1e-6:
        raise RuntimeError(f"one-hot entropy {one_hot} is not near zero")


class UncertaintyTriager:
    """Score frames and keep the highest-entropy fraction."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def score(self, outputs: np.ndarray) -> np.ndarray:
        probs = probabilities_from_model_output(outputs)
        return predictive_entropy(probs, eps=self.eps)

    def select_top_fraction(self, scores: np.ndarray, top_k_ratio: float) -> np.ndarray:
        if not 0.0 <= top_k_ratio <= 1.0:
            raise ValueError("top_k_ratio must be between 0 and 1")
        values = np.asarray(scores, dtype=np.float64)
        count = int(values.shape[0])
        k = min(count, int(np.ceil(count * top_k_ratio)))
        if k == 0:
            return np.empty(0, dtype=np.int64)
        order = np.argsort(values, kind="mergesort")[::-1]
        chosen = np.ascontiguousarray(order[:k])
        if k < count and np.any(values[order[k:]] > values[chosen].min() + 1e-9):
            raise RuntimeError("top-k selection is not the highest scores")
        return chosen


@torch.no_grad()
def monte_carlo_dropout_entropy(
    model: nn.Module,
    images: torch.Tensor,
    n_samples: int = 8,
    chunk_size: int = 16,
) -> np.ndarray:
    """Entropy of the mean softmax across stochastic dropout passes."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    device = next(model.parameters()).device
    images = images.to(device)
    was_training = model.training
    model.train()
    try:
        parts: list[np.ndarray] = []
        for start in range(0, images.shape[0], chunk_size):
            chunk = images[start : start + chunk_size]
            total = None
            for _ in range(n_samples):
                probs = torch.softmax(model(chunk), dim=-1)
                total = probs if total is None else total + probs
            mean = (total / float(n_samples)).detach().cpu().numpy()
            parts.append(predictive_entropy(mean))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    finally:
        model.train(was_training)


def controlled_logits(
    num_frames: int,
    num_classes: int,
    hard_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Peaked logits for easy frames and near-zero logits for hard frames."""
    if num_frames < 1 or num_classes < 2:
        raise ValueError("need at least one frame and two classes")
    if not 0.0 < hard_fraction < 1.0:
        raise ValueError("hard_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    hard_count = int(round(num_frames * hard_fraction))
    hard_count = min(max(hard_count, 1), num_frames - 1)
    hard = np.zeros(num_frames, dtype=bool)
    hard[:hard_count] = True
    rng.shuffle(hard)

    logits = np.zeros((num_frames, num_classes), dtype=np.float64)
    classes = rng.integers(0, num_classes, size=num_frames)
    logits[np.arange(num_frames), classes] = 6.0
    logits[hard] = rng.normal(0.0, 0.05, size=(hard_count, num_classes))
    return logits, hard


def run_closed_loop_benchmark(
    model: nn.Module,
    device: torch.device,
    num_frames: int = 64,
    top_k_ratio: float = 0.10,
    mc_samples: int = 8,
    input_size: int = 128,
    seed: int = 0,
    hard_fraction: float = 0.50,
) -> dict:
    """Dispatch high-entropy synthetic logits and score the live model separately.

    The queue is fed a labeled mix of peaked and near-uniform logits so the
    top fraction is a real separation, not a tie. Monte Carlo dropout on
    ``model`` is measured and reported on its own.
    """
    _check_entropy_identity()
    if num_frames < 1:
        raise ValueError("num_frames must be positive")

    num_classes = int(model.head.out_features)
    logits, hard = controlled_logits(num_frames, num_classes, hard_fraction, seed)
    scores = UncertaintyTriager().score(logits)
    chosen = UncertaintyTriager().select_top_fraction(scores, top_k_ratio)
    hard_count = int(hard.sum())
    if len(chosen) <= hard_count:
        if not np.all(hard[chosen]):
            raise RuntimeError("top-k entropy did not select the near-uniform controls")
    elif not np.all(np.isin(np.flatnonzero(hard), chosen)):
        raise RuntimeError("triage omitted a near-uniform control")

    queue = ClinicianReviewQueue()
    for frame_id in chosen.tolist():
        queue.push(int(frame_id), float(scores[frame_id]))

    review_order = []
    previous = float("inf")
    while len(queue):
        item = queue.pop()
        if item.uncertainty > previous + 1e-9:
            raise RuntimeError("review queue did not pop in priority order")
        previous = item.uncertainty
        review_order.append(
            {
                "frame_id": item.frame_id,
                "uncertainty": item.uncertainty,
                "labeled_hard": bool(hard[item.frame_id]),
            }
        )

    triaged_count = len(review_order)
    triaged_scores = scores[chosen] if triaged_count else np.empty(0, dtype=np.float64)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    images = torch.rand(num_frames, 3, input_size, input_size, generator=generator)
    images = images.to(device)
    model_scores = monte_carlo_dropout_entropy(model, images, n_samples=mc_samples)

    return {
        "method": "predictive_entropy",
        "score_source": "synthetic_logits",
        "entropy_units": "nats",
        "formula": "H(p) = -sum_i p_i log(p_i)",
        "num_frames": num_frames,
        "num_classes": num_classes,
        "hard_fraction": hard_fraction,
        "hard_frames": hard_count,
        "easy_frames": int(num_frames - hard_count),
        "top_k_ratio": top_k_ratio,
        "triaged_count": triaged_count,
        "triaged_ratio": float(triaged_count / num_frames),
        "triaged_hard_count": int(hard[chosen].sum()) if triaged_count else 0,
        "avg_uncertainty_score": float(scores.mean()),
        "avg_triaged_uncertainty": float(triaged_scores.mean()) if triaged_count else None,
        "easy_mean_uncertainty": float(scores[~hard].mean()),
        "hard_mean_uncertainty": float(scores[hard].mean()),
        "uncertainty_min": float(scores.min()),
        "uncertainty_max": float(scores.max()),
        "queue_depth_after_dispatch": triaged_count,
        "entropy_self_check_passed": True,
        "priority_order_verified": True,
        "review_order": review_order,
        "model_mc_dropout": {
            "mc_samples": mc_samples,
            "num_frames": num_frames,
            "avg_uncertainty_score": float(model_scores.mean()),
            "uncertainty_min": float(model_scores.min()),
            "uncertainty_max": float(model_scores.max()),
            "uniform_ceiling_nats": float(np.log(num_classes)),
            "note": (
                "Monte Carlo dropout entropy of the untrained TinySurgicalNet on "
                "random inputs. Scores sit near the uniform ceiling because the "
                "network has not been fit to a task."
            ),
        },
    }
