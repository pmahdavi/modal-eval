"""Registry for loading per-benchmark YAML configs."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from .models import (
    BenchmarkConfig,
    EvalParams,
    InfraConfig,
    RunConfig,
    SamplingConfig,
    VLLMConfig,
    model_id_to_slug,
)
from .overrides import (
    ParsedOverrides,
    compute_config_hash,
    merge_eval_overrides,
    merge_infra_overrides,
    merge_sampling_overrides,
    merge_vllm_overrides,
)


class Registry:
    """Loads per-benchmark YAML configs from config/ directory."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.benchmarks: dict[str, BenchmarkConfig] = {}

        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        # Load all YAML files in config/
        for yaml_file in config_dir.glob("*.yaml"):
            name = yaml_file.stem  # filename without extension
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {yaml_file}: {e}") from e

            if data is None:
                warnings.warn(f"Skipping empty YAML file: {yaml_file}")
                continue

            # Parse nested configs
            if "infra" in data and isinstance(data["infra"], dict):
                data["infra"] = InfraConfig(**data["infra"])
            if "vllm" in data and isinstance(data["vllm"], dict):
                data["vllm"] = VLLMConfig(**data["vllm"])
            if "sampling" in data and isinstance(data["sampling"], dict):
                data["sampling"] = SamplingConfig(**data["sampling"])
            if "eval" in data and isinstance(data["eval"], dict):
                data["eval"] = EvalParams(**data["eval"])

            data["name"] = name
            benchmark = BenchmarkConfig(**data)
            self.benchmarks[name] = benchmark

        # Templates directory (sibling to config/)
        self.templates_dir = config_dir.parent / "templates"

    def list_benchmarks(self) -> list[str]:
        """List all available benchmark names."""
        return sorted(self.benchmarks.keys())

    def get_benchmark(self, name: str) -> BenchmarkConfig:
        """Get a benchmark by name."""
        if name not in self.benchmarks:
            raise KeyError(
                f"Benchmark '{name}' not found. Available: {self.list_benchmarks()}"
            )
        return self.benchmarks[name]

    def get_chat_template(self, name: str) -> str:
        """Load chat template from templates/ directory."""
        path = self.templates_dir / f"{name}.jinja"
        if not path.exists():
            raise FileNotFoundError(f"Template '{name}' not found at {path}")
        return path.read_text()

    def build_run_config(
        self,
        model_id: str,
        benchmark_name: str,
        checkpoint_dir: str,
        overrides: Optional[ParsedOverrides] = None,
    ) -> RunConfig:
        """Build complete RunConfig by merging benchmark + CLI overrides.

        Args:
            model_id: HuggingFace model ID
            benchmark_name: Name of benchmark (filename without .yaml)
            checkpoint_dir: Directory to store checkpoints
            overrides: Parsed CLI overrides (--infra.*, --vllm.*, --sampling.*, --eval.*, --env.*)

        Returns:
            Complete RunConfig with all settings merged
        """
        benchmark = self.get_benchmark(benchmark_name)

        # Apply overrides to each config section
        if overrides and overrides.has_infra_overrides():
            infra = merge_infra_overrides(benchmark.infra, overrides.infra)
        else:
            infra = benchmark.infra

        if overrides and overrides.has_vllm_overrides():
            vllm = merge_vllm_overrides(benchmark.vllm, overrides.vllm)
        else:
            vllm = benchmark.vllm

        if overrides and overrides.has_sampling_overrides():
            sampling = merge_sampling_overrides(benchmark.sampling, overrides.sampling)
        else:
            sampling = benchmark.sampling

        if overrides and overrides.has_eval_overrides():
            eval_params = merge_eval_overrides(benchmark.eval, overrides.eval)
        else:
            eval_params = benchmark.eval

        # Merge env_args
        env_args = dict(benchmark.env_args)
        if overrides and overrides.env:
            env_args.update(overrides.env)

        # Compute hash if any overrides were applied
        config_hash = None
        if overrides and overrides.has_any_overrides():
            config_hash = compute_config_hash(infra, vllm, sampling, eval_params, env_args)

        return RunConfig(
            model_id=model_id,
            model_slug=model_id_to_slug(model_id),
            benchmark_name=benchmark_name,
            infra=infra,
            vllm=vllm,
            sampling=sampling,
            eval=eval_params,
            env_args=env_args,
            chat_template=benchmark.chat_template,
            env_id=benchmark.env_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            checkpoint_dir=checkpoint_dir,
            config_hash=config_hash,
        )


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    """Get cached registry instance."""
    config_dir = Path(__file__).parent / "config"
    return Registry(config_dir)
