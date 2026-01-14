"""Override parsing utilities for namespaced dotted notation CLI options.

Supports syntax like:
    --infra.gpu="H100:2"
    --vllm.max_model_len=81920
    --sampling.temperature=0.5
    --eval.rollouts=8
    --env.sandbox_backend=modal
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .models import EvalParams, InfraConfig, SamplingConfig, VLLMConfig


# Field type definitions for each namespace
FIELD_TYPES: dict[str, dict[str, type]] = {
    "infra": {
        "gpu": str,
        "timeout": int,
        "scaledown_window": int,
        "max_concurrent_inputs": int,
    },
    "vllm": {
        "max_model_len": int,
        "gpu_memory_utilization": float,
        "tensor_parallel_size": int,
        "data_parallel_size": int,
        "prefix_caching": bool,
        "trust_remote_code": bool,
        "attention_backend": str,
        "dtype": str,
        "quantization": str,
    },
    "sampling": {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
        "min_tokens": int,
    },
    "eval": {
        "num_examples": int,
        "rollouts": int,
        "max_concurrent": int,
        "checkpoint_every": int,
    },
    "env": {},  # Dynamic - any key allowed
}


@dataclass
class ParsedOverrides:
    """Container for parsed CLI overrides by namespace."""

    infra: dict[str, Any] = field(default_factory=dict)
    vllm: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)

    def has_infra_overrides(self) -> bool:
        """Check if any infra overrides were specified."""
        return bool(self.infra)

    def has_vllm_overrides(self) -> bool:
        """Check if any vllm overrides were specified."""
        return bool(self.vllm)

    def has_sampling_overrides(self) -> bool:
        """Check if any sampling overrides were specified."""
        return bool(self.sampling)

    def has_eval_overrides(self) -> bool:
        """Check if any eval overrides were specified."""
        return bool(self.eval)

    def has_env_overrides(self) -> bool:
        """Check if any env overrides were specified."""
        return bool(self.env)

    def has_any_overrides(self) -> bool:
        """Check if any overrides were specified."""
        return bool(self.infra or self.vllm or self.sampling or self.eval or self.env)

    def is_empty(self) -> bool:
        """Check if no overrides were specified."""
        return not self.has_any_overrides()


def parse_dotted_options(args: list[str]) -> ParsedOverrides:
    """Parse dotted notation options from Click's extra args.

    Handles formats:
        --namespace.field=value
        --namespace.field value

    Args:
        args: List of extra arguments from Click context

    Returns:
        ParsedOverrides with separated namespace dictionaries

    Raises:
        ValueError: If namespace or field is unknown, or type coercion fails
    """
    overrides = ParsedOverrides()

    i = 0
    while i < len(args):
        arg = args[i]

        if not arg.startswith("--"):
            raise ValueError(f"Expected option starting with '--', got: {arg}")

        # Handle --namespace.field=value format
        if "=" in arg:
            key, value = arg[2:].split("=", 1)
            namespace, field_name = _parse_key(key)
            typed_value = _coerce_type(namespace, field_name, value)
            _store_override(overrides, namespace, field_name, typed_value)
            i += 1

        # Handle --namespace.field value format
        elif "." in arg[2:]:
            key = arg[2:]
            namespace, field_name = _parse_key(key)
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for {arg}")
            value = args[i + 1]
            typed_value = _coerce_type(namespace, field_name, value)
            _store_override(overrides, namespace, field_name, typed_value)
            i += 2

        else:
            raise ValueError(
                f"Unknown option: {arg}. "
                f"Use --namespace.field=value format (e.g., --vllm.max_model_len=81920)"
            )

    return overrides


def _parse_key(key: str) -> tuple[str, str]:
    """Parse 'namespace.field' into (namespace, field).

    Args:
        key: Key in format 'namespace.field'

    Returns:
        Tuple of (namespace, field_name)

    Raises:
        ValueError: If key format is invalid or namespace is unknown
    """
    if "." not in key:
        raise ValueError(
            f"Invalid key format: {key}. Expected namespace.field "
            f"(e.g., vllm.max_model_len)"
        )

    parts = key.split(".", 1)
    namespace, field_name = parts[0], parts[1]

    if namespace not in FIELD_TYPES:
        valid = ", ".join(FIELD_TYPES.keys())
        raise ValueError(f"Unknown namespace: '{namespace}'. Valid namespaces: {valid}")

    return namespace, field_name


def _coerce_type(namespace: str, field_name: str, value: str) -> Any:
    """Coerce string value to appropriate type based on field.

    Args:
        namespace: The namespace (infra, vllm, sampling, eval, env)
        field_name: The field name within the namespace
        value: String value to coerce

    Returns:
        Value coerced to the correct type

    Raises:
        ValueError: If field is unknown (for non-env namespaces) or type coercion fails
    """
    # env namespace: auto-coerce using JSON parsing (handles int, float, bool, null)
    if namespace == "env":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value  # Keep as string if not valid JSON literal

    if field_name not in FIELD_TYPES.get(namespace, {}):
        valid = ", ".join(FIELD_TYPES[namespace].keys())
        raise ValueError(
            f"Unknown field: '{namespace}.{field_name}'. "
            f"Valid fields for {namespace}: {valid}"
        )

    target_type = FIELD_TYPES[namespace][field_name]

    # Special handling for bool
    if target_type is bool:
        if value.lower() in ("true", "1", "yes"):
            return True
        elif value.lower() in ("false", "0", "no"):
            return False
        else:
            raise ValueError(
                f"Cannot convert '{value}' to bool for {namespace}.{field_name}. "
                f"Use true/false, 1/0, or yes/no."
            )

    try:
        return target_type(value)
    except ValueError as e:
        raise ValueError(
            f"Cannot convert '{value}' to {target_type.__name__} "
            f"for {namespace}.{field_name}: {e}"
        ) from e


def _store_override(
    overrides: ParsedOverrides, namespace: str, field_name: str, value: Any
) -> None:
    """Store override in the appropriate namespace dictionary."""
    if namespace == "infra":
        overrides.infra[field_name] = value
    elif namespace == "vllm":
        overrides.vllm[field_name] = value
    elif namespace == "sampling":
        overrides.sampling[field_name] = value
    elif namespace == "eval":
        overrides.eval[field_name] = value
    elif namespace == "env":
        overrides.env[field_name] = value


def compute_config_hash(
    infra: InfraConfig,
    vllm: VLLMConfig,
    sampling: SamplingConfig,
    eval_params: EvalParams,
    env_args: dict[str, Any],
) -> str:
    """Compute deterministic hash from resolved configuration.

    The hash is computed from all configuration values to ensure
    different configurations produce different hashes.

    Args:
        infra: InfraConfig
        vllm: VLLMConfig
        sampling: SamplingConfig
        eval_params: EvalParams
        env_args: Environment arguments

    Returns:
        6-character hex string. Must be short to keep Modal URL under
        DNS 63-char label limit: ota-merge--mk-{slug}-{benchmark}-{hash}
    """
    data = {
        "infra": infra.model_dump(),
        "vllm": vllm.model_dump(),
        "sampling": sampling.model_dump(),
        "eval": eval_params.model_dump(),
        "env_args": env_args,
    }
    # Sort keys for deterministic serialization
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:6]


def merge_infra_overrides(base: InfraConfig, overrides: dict[str, Any]) -> InfraConfig:
    """Create new InfraConfig with overrides applied.

    Args:
        base: Base InfraConfig to start from
        overrides: Dictionary of field overrides

    Returns:
        New InfraConfig with overrides applied (original unchanged)
    """
    data = base.model_dump()
    data.update(overrides)
    return InfraConfig(**data)


def merge_vllm_overrides(base: VLLMConfig, overrides: dict[str, Any]) -> VLLMConfig:
    """Create new VLLMConfig with overrides applied.

    Args:
        base: Base VLLMConfig to start from
        overrides: Dictionary of field overrides

    Returns:
        New VLLMConfig with overrides applied (original unchanged)
    """
    data = base.model_dump()
    data.update(overrides)
    return VLLMConfig(**data)


def merge_sampling_overrides(
    base: SamplingConfig, overrides: dict[str, Any]
) -> SamplingConfig:
    """Create new SamplingConfig with overrides applied.

    Args:
        base: Base SamplingConfig to start from
        overrides: Dictionary of field overrides

    Returns:
        New SamplingConfig with overrides applied (original unchanged)
    """
    data = base.model_dump()
    data.update(overrides)
    return SamplingConfig(**data)


def merge_eval_overrides(base: EvalParams, overrides: dict[str, Any]) -> EvalParams:
    """Create new EvalParams with overrides applied.

    Args:
        base: Base EvalParams to start from
        overrides: Dictionary of field overrides

    Returns:
        New EvalParams with overrides applied (original unchanged)
    """
    data = base.model_dump()
    data.update(overrides)
    return EvalParams(**data)
