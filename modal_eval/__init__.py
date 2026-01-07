"""
Modal Eval - Unified LLM evaluation on Modal with vLLM serving.

A standalone framework for running language model benchmarks on Modal's GPU
infrastructure. Supports LiveCodeBench, GSM8K, Math500, AIME2025, and IFEval.

Usage (CLI):
    # Run evaluation
    modal-eval run --model allenai/Olmo-3-7B-Think --benchmark livecodebench

    # Run with overrides
    modal-eval run --model allenai/Olmo-3-7B-Think --benchmark livecodebench \\
        --vllm.max_model_len=81920 \\
        --sampling.temperature=0.5

    # Deploy a model (persistent)
    modal-eval deploy --model allenai/Olmo-3-7B-Think --benchmark livecodebench

    # Resume from checkpoint
    modal-eval resume --model allenai/Olmo-3-7B-Think --benchmark livecodebench

    # Check status
    modal-eval status

    # List available benchmarks
    modal-eval benchmarks

Direct deployment (alternative):
    MODAL_EVAL_MODEL=allenai/Olmo-3-7B-Think MODAL_EVAL_BENCHMARK=livecodebench \\
        modal deploy modal_eval/serve.py

Environment Variables:
    MODAL_EVAL_OUTPUT_DIR: Override default output directory (default: outputs/evals)
"""

__version__ = "0.1.0"

from pathlib import Path

# Package root directory
PACKAGE_DIR = Path(__file__).parent
CONFIG_DIR = PACKAGE_DIR / "config"

# Re-export key classes for convenience
from .models import (
    BenchmarkConfig,
    EvalParams,
    InfraConfig,
    RunConfig,
    SamplingConfig,
    VLLMConfig,
    model_id_to_slug,
)
from .registry import Registry, get_registry

__all__ = [
    "__version__",
    "PACKAGE_DIR",
    "CONFIG_DIR",
    "InfraConfig",
    "VLLMConfig",
    "SamplingConfig",
    "EvalParams",
    "BenchmarkConfig",
    "RunConfig",
    "Registry",
    "get_registry",
    "model_id_to_slug",
]
