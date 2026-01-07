# modal-eval

Unified LLM evaluation framework on [Modal](https://modal.com) with [vLLM](https://github.com/vllm-project/vllm) serving.

A standalone tool for running language model benchmarks on Modal's GPU infrastructure using the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) evaluation framework from PrimeIntellect.

## Overview

modal-eval provides a streamlined workflow for:
1. **Deploying models** to Modal's GPU infrastructure with vLLM
2. **Running evaluations** using verifiers (vf-eval) from PrimeIntellect
3. **Managing checkpoints** for long-running evaluations

It integrates with the [Prime evaluation environments](https://app.primeintellect.ai/dashboard/environments?ex_sort=by_sections) - you can use any environment listed there by specifying its `env_id` in your benchmark configuration.

## Features

- **One-command evaluation**: Deploy model + run benchmark in a single command
- **Automatic proxy**: Handles Modal's 150s web endpoint timeout via local proxy
- **Checkpoint/resume**: Resume interrupted evaluations seamlessly
- **Config overrides**: Override any setting via CLI dotted notation
- **Extensible**: Add new benchmarks by creating YAML configs

## Installation

```bash
# Basic installation
pip install modal-eval

# With HuggingFace Hub support (for pushing results)
pip install modal-eval[hub]

# With verifiers integration (recommended)
pip install modal-eval[vf-eval]

# Development
pip install modal-eval[dev]
```

### Prerequisites

1. **Modal account**: Sign up at [modal.com](https://modal.com) and authenticate:
   ```bash
   modal token new
   ```

2. **Modal secrets**: Create a HuggingFace secret in Modal:
   ```bash
   modal secret create huggingface HF_TOKEN=your_token_here
   ```

3. **verifiers** (the evaluation framework):
   ```bash
   pip install "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git"
   ```

## Quick Start

```bash
# Run evaluation
modal-eval run --model allenai/Olmo-3-7B-Think --benchmark livecodebench

# Deploy model (persistent)
modal-eval deploy --model allenai/Olmo-3-7B-Think --benchmark livecodebench

# Check status
modal-eval status

# List available benchmarks
modal-eval benchmarks
```

## How It Works

modal-eval orchestrates:

1. **vLLM Server on Modal**: Deploys your model with optimal GPU configuration
2. **Local Proxy**: Routes requests to Modal, handling timeout limitations
3. **verifiers (vf-eval)**: Runs the actual evaluation against the model

```
┌─────────────────────────────────────────────────────────────┐
│  Local Machine                                              │
│  ┌─────────────┐     ┌─────────────┐                        │
│  │   vf-eval   │────▶│    Proxy    │────────────────────┐   │
│  │ (verifiers) │     │ :ephemeral  │                    │   │
│  └─────────────┘     └─────────────┘                    │   │
└─────────────────────────────────────────────────────────│───┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Modal (Cloud)                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  vLLM Server (A100/H100)                            │    │
│  │  - OpenAI-compatible API                            │    │
│  │  - Model: your-model-id                             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## CLI Commands

### `modal-eval run`

Run evaluation with automatic deployment and proxy management.

```bash
modal-eval run -m allenai/Olmo-3-7B-Think -b livecodebench

# With config overrides
modal-eval run -m allenai/Olmo-3-7B-Think -b livecodebench \
    --vllm.max_model_len=81920 \
    --sampling.temperature=0.5 \
    --eval.rollouts=8
```

**Override namespaces:**

| Namespace | Examples |
|-----------|----------|
| `--infra.*` | `--infra.gpu="H100:2"`, `--infra.timeout=7200` |
| `--vllm.*` | `--vllm.max_model_len=81920`, `--vllm.tensor_parallel_size=2` |
| `--sampling.*` | `--sampling.temperature=0.5`, `--sampling.max_tokens=32768` |
| `--eval.*` | `--eval.rollouts=8`, `--eval.num_examples=100` |
| `--env.*` | `--env.sandbox_backend=modal` |

### `modal-eval resume`

Resume an interrupted evaluation.

```bash
# Auto-find latest incomplete run
modal-eval resume -m allenai/Olmo-3-7B-Think -b livecodebench

# Resume specific checkpoint
modal-eval resume ./outputs/evals/livecodebench--allenai--Olmo-3-7B-Think/abc123
```

### `modal-eval deploy`

Deploy a model persistently (without running evaluation).

```bash
modal-eval deploy -m allenai/Olmo-3-7B-Think -b livecodebench
```

### `modal-eval stop`

Stop deployed Modal apps.

```bash
# Stop specific model
modal-eval stop -m allenai/Olmo-3-7B-Think -b livecodebench

# Stop all modal-eval apps
modal-eval stop --all
```

### `modal-eval status`

Show currently deployed models.

### `modal-eval runs`

List recent evaluation runs.

```bash
modal-eval runs
modal-eval runs -b livecodebench -l 20
```

### `modal-eval benchmarks`

List available benchmarks with their configurations.

## Available Benchmarks

Pre-configured benchmarks (using [Prime evaluation environments](https://app.primeintellect.ai/dashboard/environments)):

| Benchmark | env_id | GPUs | Parallelism | Rollouts | Notes |
|-----------|--------|------|-------------|----------|-------|
| livecodebench | livecodebench-modal | 2x A100 | DP=2 | 2 | Code generation with Modal sandbox |
| gsm8k | gsm8k | 1x A100 | - | 4 | Grade school math |
| math500 | math500 | 1x A100 | - | 4 | Math reasoning |
| aime2025 | aime2025 | 2x A100 | TP=2 | 32 | Competition math |
| ifeval | ifeval | 2x A100 | DP=2 | 4 | Instruction following |

### Adding New Benchmarks

You can use any environment from the [Prime environments dashboard](https://app.primeintellect.ai/dashboard/environments?ex_sort=by_sections). Create a YAML file in `modal_eval/config/`:

```yaml
# my_benchmark.yaml
env_id: your-prime-env-id  # From Prime dashboard

infra:
  gpu: "A100-80GB"
  timeout: 7200

vllm:
  max_model_len: 16384

sampling:
  temperature: 0.7
  max_tokens: 8192

eval:
  rollouts: 4
  max_concurrent: 32
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODAL_EVAL_OUTPUT_DIR` | Output directory for results | `outputs/evals` |
| `MODAL_EVAL_MODEL` | Model ID for direct deployment | - |
| `MODAL_EVAL_BENCHMARK` | Benchmark for direct deployment | - |

### Benchmark Configuration Reference

```yaml
# Full configuration example
env_id: livecodebench-modal  # verifiers environment ID

chat_template: code  # Optional: base, code, math, think

infra:
  gpu: "A100-80GB:2"       # GPU type and count
  timeout: 7200             # Function timeout (seconds)
  scaledown_window: 900     # Idle timeout before scale-down
  max_concurrent_inputs: 999

vllm:
  max_model_len: 40960
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1   # Shard across GPUs
  data_parallel_size: 2     # Replicate across GPUs
  prefix_caching: true
  trust_remote_code: true

sampling:
  temperature: 1.0
  top_p: 1.0
  max_tokens: 32768

eval:
  num_examples: null        # null = run all
  rollouts: 2               # Samples per example
  max_concurrent: 32        # vf-eval workers
  checkpoint_every: 1

env_args:                   # Passed to verifiers
  sandbox_backend: modal
```

## Architecture

```
modal-eval/
├── modal_eval/
│   ├── cli.py              # CLI entry point (Click)
│   ├── runner.py           # EvalRunner + ProxyManager
│   ├── serve.py            # Modal vLLM server definition
│   ├── proxy.py            # FastAPI proxy (timeout workaround)
│   ├── models.py           # Pydantic config models
│   ├── registry.py         # Benchmark config loading
│   ├── checkpoint.py       # Checkpoint management
│   ├── modal_lifecycle.py  # Modal CLI wrapper
│   ├── push_to_hub.py      # Push results to HF Hub
│   ├── config/             # Benchmark YAML configs
│   └── templates/          # Jinja2 chat templates
```

## Direct Deployment

For advanced use cases, deploy directly with environment variables:

```bash
MODAL_EVAL_MODEL=allenai/Olmo-3-7B-Think \
MODAL_EVAL_BENCHMARK=livecodebench \
    modal deploy modal_eval/serve.py
```

## Development

```bash
# Clone and install
git clone https://github.com/your-org/modal-eval.git
cd modal-eval
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check modal_eval
ruff format modal_eval

# Type check
mypy modal_eval
```

## Related Projects

- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) - The evaluation framework powering modal-eval
- [Prime Environments](https://app.primeintellect.ai/dashboard/environments) - Available evaluation environments
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [Modal](https://modal.com) - Serverless GPU infrastructure

## License

MIT
