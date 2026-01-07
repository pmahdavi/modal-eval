"""CheckpointStore for managing eval run state and resumption."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import RunConfig


class CheckpointStore:
    """Manages eval run state and checkpoints."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def find_latest_incomplete(
        self, env_id: str, model_id: str
    ) -> Optional[Path]:
        """Find most recent incomplete checkpoint for this model+benchmark.

        Args:
            env_id: The vf-eval environment ID (e.g., 'livecodebench')
            model_id: HuggingFace model ID (e.g., 'allenai/Olmo-3-7B-Think')
        """
        model_str = model_id.replace("/", "--")
        parent = self.output_dir / f"{env_id}--{model_str}"

        if not parent.exists():
            return None

        # Find runs with run_config.json but incomplete results
        # Sort by directory name (contains timestamp) in reverse order
        try:
            run_dirs = sorted(
                [d for d in parent.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True,
            )
        except Exception:
            return None

        for run_dir in run_dirs:
            config_file = run_dir / "run_config.json"
            results_file = run_dir / "results.jsonl"

            if config_file.exists():
                # Check if incomplete (has config but no final results)
                if not results_file.exists():
                    return run_dir

                # Or has results but fewer than expected
                # Note: When num_examples is None (run all), we can't determine
                # incompleteness by count, so we skip this check and consider
                # any run with results as complete.
                try:
                    config = RunConfig.load(config_file)
                    if config.eval.num_examples:
                        with open(results_file) as f:
                            num_results = sum(1 for _ in f)
                        if num_results < config.eval.num_examples:
                            return run_dir
                except Exception:
                    # Skip malformed configs
                    continue

        return None

    def list_runs(
        self,
        benchmark: Optional[str] = None,
        model_str: Optional[str] = None,
    ) -> list[Path]:
        """List all run directories, optionally filtered.

        Args:
            benchmark: Filter by benchmark/env_id name
            model_str: Filter by model string in directory format (e.g., 'user--model')
        """
        runs = []

        if not self.output_dir.exists():
            return runs

        for dir_path in self.output_dir.iterdir():
            if not dir_path.is_dir():
                continue
            if "--" not in dir_path.name:
                continue

            bench, model = dir_path.name.split("--", 1)
            if benchmark and bench != benchmark:
                continue
            if model_str and model != model_str:
                continue

            for run_dir in dir_path.iterdir():
                if (run_dir / "run_config.json").exists():
                    runs.append(run_dir)

        return sorted(runs, key=lambda x: x.name, reverse=True)

    def get_run_summary(self, run_dir: Path) -> dict:
        """Get summary info about a run."""
        config_file = run_dir / "run_config.json"
        results_file = run_dir / "results.jsonl"

        summary = {
            "path": str(run_dir),
            "config_exists": config_file.exists(),
            "results_exist": results_file.exists(),
        }

        if config_file.exists():
            try:
                config = RunConfig.load(config_file)
                summary.update({
                    "model_id": config.model_id,
                    "benchmark": config.benchmark_name,
                    "started_at": config.started_at,
                    "expected": config.eval.num_examples,
                })
            except Exception as e:
                summary["config_error"] = str(e)

        if results_file.exists():
            try:
                with open(results_file) as f:
                    summary["completed"] = sum(1 for _ in f)
            except Exception as e:
                summary["results_error"] = str(e)

        return summary
