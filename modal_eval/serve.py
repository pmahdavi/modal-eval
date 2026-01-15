"""
Unified vLLM Server for Modal.

Deploy a model:
    MODAL_EVAL_MODEL=allenai/Olmo-3-7B-Think MODAL_EVAL_BENCHMARK=livecodebench modal deploy modal_eval/serve.py

Stop:
    modal app stop mk-{model_slug}-{benchmark}
"""

import base64
import json
import os
import subprocess
from pathlib import Path

import modal


def resolve_chat_template(
    model_id: str,
    benchmark_template: str | None = None,
    force: bool = False,
) -> tuple[str | None, str]:
    """
    Resolve chat template with priority:
    1. tokenizer_config.json chat_template (return None - vLLM handles automatically)
    2. HF chat_template.jinja file (fetch and return content)
    3. Benchmark's local template (load from templates/)
    4. Error if none found

    If force=True, skip steps 1-2 and use the benchmark's local template directly.

    Returns: (template_content, source_description)
    """
    from huggingface_hub import hf_hub_download, file_exists

    # Force mode: skip HF lookup, use local template directly
    if force and benchmark_template:
        local_path = Path(__file__).parent / "templates" / f"{benchmark_template}.jinja"
        if local_path.exists():
            print(f"Force using local template: {local_path}")
            return local_path.read_text(), f"templates/{benchmark_template}.jinja (forced)"
        raise ValueError(f"Forced template '{benchmark_template}' not found at {local_path}")

    # 1. Check tokenizer_config.json for chat_template
    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer_config.json"
        )
        with open(config_path) as f:
            tokenizer_config = json.load(f)
        if tokenizer_config.get("chat_template"):
            print(f"Using chat_template from {model_id}/tokenizer_config.json")
            return None, "tokenizer_config.json"  # vLLM uses automatically
    except Exception as e:
        print(f"Could not check tokenizer_config.json: {e}")

    # 2. Check for chat_template.jinja in HF repo
    try:
        if file_exists(repo_id=model_id, filename="chat_template.jinja"):
            jinja_path = hf_hub_download(
                repo_id=model_id,
                filename="chat_template.jinja"
            )
            print(f"Using chat_template.jinja from {model_id}")
            with open(jinja_path) as f:
                return f.read(), "HF chat_template.jinja"
    except Exception as e:
        print(f"Could not check chat_template.jinja: {e}")

    # Warn: HF has no template
    print(f"WARNING: Model {model_id} has no chat_template in HuggingFace repo")

    # 3. Use benchmark's local template
    if benchmark_template:
        local_path = Path(__file__).parent / "templates" / f"{benchmark_template}.jinja"
        if local_path.exists():
            print(f"Using local template: {local_path}")
            return local_path.read_text(), f"templates/{benchmark_template}.jinja"
        raise ValueError(
            f"Benchmark template '{benchmark_template}' not found at {local_path}"
        )

    # 4. Error - no template anywhere
    raise ValueError(
        f"No chat template found for {model_id}.\n"
        f"  - No chat_template in tokenizer_config.json\n"
        f"  - No chat_template.jinja in HF repo\n"
        f"  - No template specified in benchmark config\n"
        f"Add a template to the HF repo or specify one in the benchmark."
    )


def get_config() -> dict:
    """
    Get configuration. Handles two contexts:

    1. DEPLOY TIME (local machine):
       - MODAL_EVAL_CONFIG is NOT set yet
       - Reads MODAL_EVAL_MODEL + MODAL_EVAL_BENCHMARK from env
       - Optionally reads MODAL_EVAL_BENCHMARK_CONFIG for custom config overrides
       - Loads config via Registry, resolves chat template
       - Stores result in MODAL_EVAL_CONFIG for the container (base64 encoded)

    2. RUNTIME (Modal container):
       - MODAL_EVAL_CONFIG IS set (embedded in image as base64)
       - Decode and parse JSON
    """
    # Check if we're in the Modal container (MODAL_EVAL_CONFIG already set)
    config_b64 = os.environ.get("MODAL_EVAL_CONFIG")
    if config_b64:
        config_json = base64.b64decode(config_b64).decode("utf-8")
        return json.loads(config_json)

    # We're deploying locally - need to build config from env vars + Registry
    from modal_eval.models import BenchmarkConfig, model_id_to_slug
    from modal_eval.overrides import compute_config_hash
    from modal_eval.registry import get_registry

    model_id = os.environ.get("MODAL_EVAL_MODEL")
    benchmark_name = os.environ.get("MODAL_EVAL_BENCHMARK")
    benchmark_config_b64 = os.environ.get("MODAL_EVAL_BENCHMARK_CONFIG")

    if not model_id:
        raise ValueError(
            "Missing MODAL_EVAL_MODEL. Set to HuggingFace model ID, e.g.:\n"
            "  MODAL_EVAL_MODEL=allenai/Olmo-3-7B-Think MODAL_EVAL_BENCHMARK=livecodebench modal deploy modal_eval/serve.py"
        )

    if not benchmark_name:
        raise ValueError(
            "Missing MODAL_EVAL_BENCHMARK. Set to benchmark name, e.g.:\n"
            "  MODAL_EVAL_MODEL=allenai/Olmo-3-7B-Think MODAL_EVAL_BENCHMARK=livecodebench modal deploy modal_eval/serve.py"
        )

    registry = get_registry()

    # Get benchmark config - priority: MODAL_EVAL_BENCHMARK_CONFIG > registry lookup
    config_hash = None
    if benchmark_config_b64:
        # Full config with CLI overrides passed from modal_lifecycle.deploy()
        config_json = base64.b64decode(benchmark_config_b64).decode("utf-8")
        benchmark = BenchmarkConfig.model_validate_json(config_json)
        config_hash = compute_config_hash(
            benchmark.infra, benchmark.vllm, benchmark.sampling,
            benchmark.eval, benchmark.env_args
        )
        print(f"Using custom benchmark config (hash={config_hash})")
    else:
        benchmark = registry.get_benchmark(benchmark_name)

    # Resolve chat template using priority chain:
    # 1. tokenizer_config.json chat_template (vLLM auto)
    # 2. HF chat_template.jinja
    # 3. Benchmark's local template
    # If force_chat_template=True, skip 1-2 and use local template directly
    chat_template_content, template_source = resolve_chat_template(
        model_id, benchmark.chat_template, force=benchmark.force_chat_template
    )
    print(f"Template source: {template_source}")

    # Generate slug from model_id
    slug = model_id_to_slug(model_id)

    # Build config dict for Modal container
    model_config = {
        "model_id": model_id,
        "slug": slug,
        "benchmark": benchmark_name,
        "config_hash": config_hash,  # None if no overrides, 6-char hash otherwise
        # Infra config values
        "gpu": benchmark.infra.gpu,
        "timeout": benchmark.infra.timeout,
        "scaledown_window": benchmark.infra.scaledown_window,
        "max_concurrent_inputs": benchmark.infra.max_concurrent_inputs,
        # vLLM config values
        "max_model_len": benchmark.vllm.max_model_len,
        "gpu_memory_utilization": benchmark.vllm.gpu_memory_utilization,
        "tensor_parallel_size": benchmark.vllm.tensor_parallel_size,
        "data_parallel_size": benchmark.vllm.data_parallel_size,
        "prefix_caching": benchmark.vllm.prefix_caching,
        "trust_remote_code": benchmark.vllm.trust_remote_code,
        "attention_backend": benchmark.vllm.attention_backend,
        "dtype": benchmark.vllm.dtype,
        "quantization": benchmark.vllm.quantization,
        # Optional chat template (None = use tokenizer default)
        "chat_template": chat_template_content,
        "template_source": template_source,
    }

    # Store as base64-encoded JSON for the Modal container
    config_json = json.dumps(model_config)
    config_b64 = base64.b64encode(config_json.encode("utf-8")).decode("ascii")
    os.environ["MODAL_EVAL_CONFIG"] = config_b64

    return model_config


# Load config at module import time
MODEL_CONFIG = get_config()

# Determine app name from config: mk-{slug}-{benchmark}[-{hash}]
# Prefix 'mk-' keeps DNS hostname under 63 chars
# Hash is appended when config has CLI overrides
_base_name = f"mk-{MODEL_CONFIG['slug']}-{MODEL_CONFIG['benchmark']}"
_config_hash = MODEL_CONFIG.get("config_hash")
APP_NAME = f"{_base_name}-{_config_hash}" if _config_hash else _base_name

app = modal.App(APP_NAME)

# Persistent volumes for caching
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/vllm": vllm_cache,
}

# Image with vLLM and fast HF downloads
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .pip_install(
        "vllm>=0.13.0",
        "transformers>=4.56.0,<5",
        "huggingface-hub>=0.27",
        "hf-transfer",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        # Pass config to container as JSON
        "MODAL_EVAL_CONFIG": os.environ.get("MODAL_EVAL_CONFIG", "{}"),
    })
)


def run_vllm_server(
    model_id: str,
    max_len: int = 16384,
    gpu_util: float = 0.9,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    prefix_caching: bool = True,
    trust_remote_code: bool = True,
    dtype: str | None = None,
    quantization: str | None = None,
    chat_template: str | None = None,
):
    """Start the vLLM OpenAI-compatible server."""
    cmd = [
        "vllm", "serve", model_id,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", str(max_len),
    ]

    if prefix_caching:
        cmd.append("--enable-prefix-caching")

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

    if data_parallel_size > 1:
        cmd.extend(["--data-parallel-size", str(data_parallel_size)])

    if dtype:
        cmd.extend(["--dtype", dtype])

    if quantization:
        cmd.extend(["--quantization", quantization])

    if chat_template:
        # Write template to file and pass to vLLM
        template_path = "/tmp/chat_template.jinja"
        with open(template_path, "w") as f:
            f.write(chat_template)
        cmd.extend(["--chat-template", template_path])

    print(f"Starting vLLM: {model_id} (ctx={max_len}, gpu_util={gpu_util}, tp={tensor_parallel_size}, dp={data_parallel_size})")
    print(f"Command: {' '.join(cmd)}")
    subprocess.Popen(cmd)


# Dynamic GPU configuration from config
GPU_TYPE = MODEL_CONFIG.get("gpu", "A100-80GB")
TIMEOUT = MODEL_CONFIG.get("timeout", 7200)
SCALEDOWN = MODEL_CONFIG.get("scaledown_window", 900)
MAX_INPUTS = MODEL_CONFIG.get("max_concurrent_inputs", 999)


@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=SCALEDOWN,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve():
    """Serve the configured model."""
    # In container, get_config() reads from MODAL_EVAL_CONFIG env var
    config = get_config()
    run_vllm_server(
        model_id=config["model_id"],
        max_len=config.get("max_model_len", 16384),
        gpu_util=config.get("gpu_memory_utilization", 0.9),
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        data_parallel_size=config.get("data_parallel_size", 1),
        prefix_caching=config.get("prefix_caching", True),
        trust_remote_code=config.get("trust_remote_code", True),
        dtype=config.get("dtype"),
        quantization=config.get("quantization"),
        chat_template=config.get("chat_template"),
    )


@app.local_entrypoint()
def main():
    """Show deployment info."""
    config = MODEL_CONFIG
    print(f"\n{'=' * 60}")
    print("Modal Eval - vLLM Server")
    print(f"{'=' * 60}")
    print(f"Model:    {config['model_id']}")
    print(f"App:      {APP_NAME}")
    print(f"GPU:      {config.get('gpu', 'A100-80GB')}")
    print(f"Context:  {config.get('max_model_len', 16384)} tokens")
    print(f"\nEndpoint: https://ota-merge--{APP_NAME}-serve.modal.run/v1")
    print(f"{'=' * 60}\n")
