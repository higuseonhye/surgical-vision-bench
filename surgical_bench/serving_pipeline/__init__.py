"""Local PyTorch stand-in for Triton-style dynamic batching."""

from .benchmark_serving import run_serving_benchmark
from .triton_client_mock import TinySurgicalNet, TritonClientMock, build_model

__all__ = ["TinySurgicalNet", "TritonClientMock", "build_model", "run_serving_benchmark"]
