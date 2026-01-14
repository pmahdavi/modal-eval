"""Pydantic models for modal_eval configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field


def model_id_to_slug(model_id: str, max_len: int = 22) -> str:
    """Convert HuggingFace model ID to URL-safe slug.

    Example: 'allenai/Olmo-3-7B-Think' -> 'allenai-olmo-3-7b-think'

    The slug is truncated to max_len to ensure DNS hostname stays under 63 chars.
    Modal hostname format: ota-merge--mk-{slug}-{benchmark}-serve.modal.run

    Raises:
        ValueError: If model_id is empty or whitespace only.
    """
    if not model_id or not model_id.strip():
        raise ValueError("model_id cannot be empty")
    slug = re.sub(r"[^a-z0-9-]", "-", model_id.lower())
    # Truncate if needed, removing trailing dashes
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug


class InfraConfig(BaseModel):
    """Modal infrastructure settings."""

    gpu: str = "A100-80GB"
    timeout: int = 7200
    scaledown_window: int = 900
    max_concurrent_inputs: int = 999


class VLLMConfig(BaseModel):
    """vLLM server settings."""

    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    prefix_caching: bool = True
    trust_remote_code: bool = True
    attention_backend: str = "FLASHINFER"
    dtype: Optional[str] = None
    quantization: Optional[str] = None


class SamplingConfig(BaseModel):
    """Sampling parameters for generation."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 8192
    min_tokens: int = 0


class EvalParams(BaseModel):
    """vf-eval parameters."""

    num_examples: Optional[int] = None  # None = all
    rollouts: int = 4
    max_concurrent: int = 32
    checkpoint_every: int = 1


class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration loaded from YAML."""

    name: str = ""  # Set from filename
    env_id: str  # vf-eval environment ID

    # Nested configs
    infra: InfraConfig = Field(default_factory=InfraConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    eval: EvalParams = Field(default_factory=EvalParams)

    # Optional
    chat_template: Optional[str] = None
    env_args: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def vf_eval_env_id(self) -> str:
        """Environment ID for vf-eval."""
        return self.env_id


class RunConfig(BaseModel):
    """Complete runtime config with overrides applied.

    Stores only resolved values (with CLI overrides applied).
    Used for resume - just uses the exact values that were used.
    """

    model_id: str
    model_slug: str
    benchmark_name: str  # Just the name, not full BenchmarkConfig

    # Resolved configs (with CLI overrides applied)
    infra: InfraConfig
    vllm: VLLMConfig
    sampling: SamplingConfig
    eval: EvalParams
    env_args: dict[str, Any]

    # Optional from benchmark
    chat_template: Optional[str] = None
    env_id: str  # vf-eval environment ID

    # Metadata
    started_at: str
    checkpoint_dir: str
    config_hash: Optional[str] = None  # For unique Modal app names

    @computed_field
    @property
    def app_name(self) -> str:
        """Modal app name: mk-{model_slug}-{benchmark}[-{hash}]

        Hash is appended when config has CLI overrides to ensure unique deployments.
        Prefix 'mk-' keeps DNS hostname under 63 chars.
        Full hostname: {modal_workspace}--mk-{slug}-{benchmark}[-{hash}]-serve.modal.run
        """
        base = f"mk-{self.model_slug}-{self.benchmark_name}"
        if self.config_hash:
            return f"{base}-{self.config_hash}"
        return base

    @computed_field
    @property
    def modal_url(self) -> str:
        """Modal endpoint URL.

        Format: ota-merge--{app_name}-serve.modal.run
        The subdomain must be ≤63 chars (DNS label limit).
        """
        return f"https://ota-merge--{self.app_name}-serve.modal.run"

    @computed_field
    @property
    def vf_eval_env_id(self) -> str:
        """Environment ID for vf-eval."""
        return self.env_id

    def save(self, path: Path) -> None:
        """Save config to JSON for resume."""
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        """Load config from JSON."""
        return cls.model_validate_json(path.read_text())
