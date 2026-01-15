#!/usr/bin/env python3
"""
Unified CLI for modal-eval.

Usage:
    modal-eval run -m allenai/Olmo-3-7B-Think -b livecodebench
    modal-eval run -m allenai/Olmo-3-7B-Think -b livecodebench \\
        --vllm.max_model_len=81920 \\
        --sampling.temperature=0.5 \\
        --eval.rollouts=8
    modal-eval resume outputs/evals/livecodebench-modal--allenai--Olmo-3-7B-Think/...
    modal-eval deploy -m allenai/Olmo-3-7B-Think -b livecodebench
    modal-eval stop -m allenai/Olmo-3-7B-Think -b livecodebench
    modal-eval status
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import click

from .checkpoint import CheckpointStore

# Default output directory, configurable via environment variable
DEFAULT_OUTPUT_DIR = "outputs/evals"


def get_output_dir() -> Path:
    """Get output directory from environment or use default."""
    return Path(os.environ.get("MODAL_EVAL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))


from .modal_lifecycle import ModalLifecycle
from .models import RunConfig, benchmark_name_to_slug, model_id_to_slug
from .overrides import parse_dotted_options
from .registry import get_registry
from .runner import EvalRunner


@click.group()
@click.version_option()
def cli():
    """Modal Eval - Unified model serving and evaluation on Modal."""
    pass


@cli.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option("--model", "-m", required=True, help="HuggingFace model ID (e.g., allenai/Olmo-3-7B-Think)")
@click.option("--benchmark", "-b", required=True, help="Benchmark name (e.g., livecodebench)")
@click.option(
    "--auto-deploy/--no-auto-deploy", default=True, help="Auto-deploy if not running"
)
@click.option(
    "--auto-cleanup/--no-auto-cleanup",
    default=False,
    help="Stop Modal app on exit",
)
@click.pass_context
def run(
    ctx,
    model,
    benchmark,
    auto_deploy,
    auto_cleanup,
):
    """Run evaluation with automatic proxy management.

    Override config with namespaced dotted notation:

    \b
    Infra (Modal):
        --infra.gpu="H100:2"
        --infra.timeout=7200
        --infra.scaledown_window=900

    \b
    vLLM (model serving):
        --vllm.max_model_len=81920
        --vllm.tensor_parallel_size=2
        --vllm.gpu_memory_utilization=0.95

    \b
    Sampling (generation):
        --sampling.temperature=0.5
        --sampling.max_tokens=73728
        --sampling.top_p=0.95

    \b
    Eval (vf-eval):
        --eval.rollouts=8
        --eval.num_examples=100
        --eval.max_concurrent=64

    \b
    Env (environment args):
        --env.sandbox_backend=modal
    """
    registry = get_registry()

    # Parse dotted notation overrides from extra args
    try:
        overrides = parse_dotted_options(ctx.args) if ctx.args else None
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)

    # Generate checkpoint directory using vf-eval compatible path convention
    # Path format: {env_id}--{model_id_with_dashes}/{uuid}
    benchmark_config = registry.get_benchmark(benchmark)
    env_id = benchmark_config.vf_eval_env_id
    model_str = model.replace("/", "--")  # model is HuggingFace model ID
    run_id = uuid.uuid4().hex[:8]  # 8-char UUID like vf-eval
    checkpoint_dir = get_output_dir() / f"{env_id}--{model_str}" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build RunConfig (merges benchmark + CLI overrides)
    run_config = registry.build_run_config(
        model_id=model,
        benchmark_name=benchmark,
        checkpoint_dir=str(checkpoint_dir),
        overrides=overrides,
    )

    # Save config for resume
    run_config.save(checkpoint_dir / "run_config.json")
    click.echo(f"Run config saved to: {checkpoint_dir / 'run_config.json'}")

    # Run evaluation
    runner = EvalRunner(
        run_config, auto_deploy=auto_deploy, auto_cleanup=auto_cleanup
    )
    exit_code = runner.run()
    sys.exit(exit_code)


@cli.command()
@click.argument("checkpoint_path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--model", "-m", help="HuggingFace model ID (for auto-find)")
@click.option("--benchmark", "-b", help="Benchmark name (for auto-find)")
def resume(checkpoint_path, model, benchmark):
    """Resume evaluation from checkpoint directory."""
    registry = get_registry()

    if checkpoint_path:
        # Resume specific checkpoint
        run_config_path = checkpoint_path / "run_config.json"
    elif model and benchmark:
        # Auto-find latest incomplete checkpoint
        store = CheckpointStore(get_output_dir())
        benchmark_config = registry.get_benchmark(benchmark)
        checkpoint_path = store.find_latest_incomplete(
            benchmark_config.vf_eval_env_id, model
        )
        if not checkpoint_path:
            click.echo(
                f"No incomplete checkpoint found for {model}+{benchmark}", err=True
            )
            sys.exit(1)
        run_config_path = checkpoint_path / "run_config.json"
    else:
        click.echo("Specify checkpoint path or --model and --benchmark", err=True)
        sys.exit(1)

    if not run_config_path.exists():
        click.echo(f"Error: {run_config_path} not found", err=True)
        sys.exit(1)

    run_config = RunConfig.load(run_config_path)
    click.echo(
        f"Resuming: model={run_config.model_id}, benchmark={run_config.benchmark_name}"
    )
    click.echo(
        f"Config: temperature={run_config.sampling.temperature}, rollouts={run_config.eval.rollouts}"
    )

    runner = EvalRunner(run_config, auto_deploy=True, resume=True)
    exit_code = runner.run()
    sys.exit(exit_code)


@cli.command()
@click.option("--model", "-m", required=True, help="HuggingFace model ID (e.g., allenai/Olmo-3-7B-Think)")
@click.option("--benchmark", "-b", required=True, help="Benchmark name (e.g., livecodebench)")
def deploy(model, benchmark):
    """Deploy a model to Modal (persistent).

    Deploys the model with the benchmark's default configuration.
    """
    registry = get_registry()
    registry.get_benchmark(benchmark)  # Validate benchmark exists
    slug = model_id_to_slug(model)
    benchmark_slug = benchmark_name_to_slug(benchmark)
    app_name = f"mk-{slug}-{benchmark_slug}"

    if ModalLifecycle.is_deployed(app_name):
        click.echo(f"Already deployed: {app_name}")
        return

    click.echo(f"Deploying {model} for benchmark {benchmark}...")
    if ModalLifecycle.deploy(model, benchmark):
        click.echo(f"Deployed: https://ota-merge--{app_name}-serve.modal.run")
    else:
        click.echo("Deployment failed", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", help="HuggingFace model ID")
@click.option("--benchmark", "-b", help="Benchmark name")
@click.option("--all", "stop_all", is_flag=True, help="Stop all Modal apps")
def stop(model, benchmark, stop_all):
    """Stop deployed Modal app(s).

    Use --all to stop all modal-eval deployed apps.
    """
    if stop_all:
        deployed = ModalLifecycle.list_deployed()
        count = 0
        for app_name in deployed:
            if app_name.startswith("mk-"):
                click.echo(f"Stopping {app_name}...")
                ModalLifecycle.stop_by_name(app_name)
                count += 1
        click.echo(f"Stopped {count} apps")
        return

    if not model or not benchmark:
        click.echo("Specify --model and --benchmark, or --all", err=True)
        sys.exit(1)

    slug = model_id_to_slug(model)
    benchmark_slug = benchmark_name_to_slug(benchmark)
    base_name = f"mk-{slug}-{benchmark_slug}"

    # Find all matching apps (including those with config hash suffix)
    # Match exactly or with hash suffix (base_name followed by '-' and hash)
    deployed = ModalLifecycle.list_deployed()
    matching = [a for a in deployed if a == base_name or a.startswith(f"{base_name}-")]

    if not matching:
        click.echo(f"No deployed apps matching {base_name}*", err=True)
        sys.exit(1)

    for app_name in matching:
        click.echo(f"Stopping {app_name}...")
        if ModalLifecycle.stop_by_name(app_name):
            click.echo(f"  Stopped")
        else:
            click.echo(f"  Failed to stop", err=True)


@cli.command()
def status():
    """Show status of all deployed models."""
    deployed = ModalLifecycle.list_deployed()
    eval_apps = [a for a in deployed if a.startswith("mk-")]

    if not eval_apps:
        click.echo("No Modal apps running")
        return

    click.echo("Deployed Modal apps:")
    for app_name in sorted(eval_apps):
        click.echo(f"  {app_name}")


@cli.command()
@click.option("--benchmark", "-b", help="Filter by benchmark")
@click.option("--model", "-m", help="Filter by HuggingFace model ID")
@click.option("--limit", "-l", type=int, default=10, help="Max runs to show")
def runs(benchmark, model, limit):
    """List recent evaluation runs."""
    store = CheckpointStore(get_output_dir())

    # Use model_str format matching directory names: model.replace("/", "--")
    model_str = None
    if model:
        model_str = model.replace("/", "--")

    # Resolve benchmark name to env_id (directories use env_id, not benchmark name)
    env_id = None
    if benchmark:
        registry = get_registry()
        try:
            benchmark_config = registry.get_benchmark(benchmark)
            env_id = benchmark_config.vf_eval_env_id
        except KeyError:
            # If not a known benchmark, use as-is (might be an env_id directly)
            env_id = benchmark

    run_list = store.list_runs(benchmark=env_id, model_str=model_str)

    if not run_list:
        click.echo("No runs found")
        return

    click.echo(f"Recent runs (showing {min(limit, len(run_list))} of {len(run_list)}):\n")
    for run_dir in run_list[:limit]:
        summary = store.get_run_summary(run_dir)
        completed = summary.get("completed")
        expected = summary.get("expected")

        # Determine status icon
        if expected is None:
            # "Run all" mode - can't determine completion by count
            status_icon = "~" if completed else "…"
        elif completed == expected:
            status_icon = "✓"
        else:
            status_icon = "…"

        completed_str = completed if completed is not None else "?"
        expected_str = expected if expected is not None else "all"

        click.echo(f"  {status_icon} {run_dir.parent.name}/{run_dir.name}")
        click.echo(f"    {completed_str}/{expected_str} examples")


@cli.command()
def benchmarks():
    """List available benchmarks."""
    registry = get_registry()
    click.echo("Available benchmarks:")
    for name in registry.list_benchmarks():
        benchmark = registry.get_benchmark(name)
        click.echo(f"  {name}: {benchmark.env_id} (GPU: {benchmark.infra.gpu})")


if __name__ == "__main__":
    cli()
