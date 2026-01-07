"""Manage Modal app deployment lifecycle."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from typing import Optional

from .models import BenchmarkConfig


class ModalLifecycle:
    """Manage Modal app deployment lifecycle."""

    @classmethod
    def list_deployed(cls) -> set[str]:
        """Get set of currently deployed app names."""
        result = subprocess.run(
            ["modal", "app", "list", "--json"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return set()
        try:
            apps = json.loads(result.stdout)
            return {
                app.get("Description", "")
                for app in apps
                if app.get("State") == "deployed"
            }
        except json.JSONDecodeError:
            return set()

    @classmethod
    def is_deployed(cls, app_name: str) -> bool:
        """Check if a specific app is deployed."""
        return app_name in cls.list_deployed()

    @classmethod
    def deploy(
        cls,
        model_id: str,
        benchmark: str,
        benchmark_config: Optional[BenchmarkConfig] = None,
    ) -> bool:
        """Deploy a model using serve.py.

        Args:
            model_id: HuggingFace model ID (e.g., 'allenai/Olmo-3-7B-Think')
            benchmark: Benchmark name (e.g., 'livecodebench')
            benchmark_config: Full BenchmarkConfig with any CLI overrides applied.
                           If provided, serialized and passed via MODAL_EVAL_BENCHMARK_CONFIG.
        """
        env = {
            **os.environ,
            "MODAL_EVAL_MODEL": model_id,
            "MODAL_EVAL_BENCHMARK": benchmark,
        }

        # If we have a custom benchmark_config (with CLI overrides), serialize it
        if benchmark_config:
            config_json = benchmark_config.model_dump_json()
            env["MODAL_EVAL_BENCHMARK_CONFIG"] = base64.b64encode(
                config_json.encode()
            ).decode("ascii")

        result = subprocess.run(
            ["modal", "deploy", "modal_eval/serve.py"],
            env=env,
        )
        return result.returncode == 0

    @classmethod
    def stop_by_name(cls, app_name: str) -> bool:
        """Stop a Modal app by its full name."""
        result = subprocess.run(["modal", "app", "stop", app_name])
        return result.returncode == 0
