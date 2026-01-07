"""EvalRunner - orchestrates proxy and vf-eval with guaranteed cleanup."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from .modal_lifecycle import ModalLifecycle
from .models import BenchmarkConfig, RunConfig


class ProxyManager:
    """Manages the proxy subprocess with ephemeral port allocation."""

    def __init__(self, modal_url: str):
        self.modal_url = modal_url
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.log_path: Optional[Path] = None
        self._log_file = None

    def _find_free_port(self) -> int:
        """Find an available port using OS allocation."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start(self, timeout: int = 60) -> int:
        """Start the proxy and return the port."""
        self.port = self._find_free_port()
        self.log_path = Path(f"proxy_{self.port}.log")
        self._log_file = open(self.log_path, "w")

        cmd = [
            sys.executable,
            "-m",
            "modal_eval.proxy",
            "--modal-url",
            self.modal_url,
            "--port",
            str(self.port),
        ]

        print(f"Starting proxy on port {self.port}...")
        print(f"Proxy logs: {self.log_path}")

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Wait for proxy to start
            time.sleep(2)
            if self.proc.poll() is not None:
                raise RuntimeError(f"Proxy failed to start (exit code: {self.proc.returncode})")

            # Wait for proxy to be ready
            if not self._wait_for_ready(timeout):
                raise RuntimeError(f"Proxy not ready after {timeout}s")

            return self.port
        except Exception:
            # Ensure log file is closed on failure
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            # Clean up on failure
            self.stop(timeout=5)
            raise

    def _wait_for_ready(self, timeout: int) -> bool:
        """Wait for proxy to respond to health check."""
        url = f"http://localhost:{self.port}/proxy/status"
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = httpx.get(url, timeout=5)
                if response.status_code == 200:
                    print("Proxy is ready!")
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def stop(self, timeout: int = 30):
        """Stop the proxy process gracefully."""
        if self.proc is None:
            return

        if self.proc.poll() is not None:
            # Already stopped
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            return

        print(f"Stopping proxy (timeout={timeout}s)...")

        # Try graceful termination first
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("Proxy did not stop gracefully, killing...")
            self.proc.kill()
            self.proc.wait(timeout=5)

        if self._log_file:
            self._log_file.close()
            self._log_file = None
            print(f"Proxy logs saved to: {self.log_path}")


class EvalRunner:
    """Orchestrates proxy and vf-eval with guaranteed cleanup."""

    def __init__(
        self,
        config: RunConfig,
        auto_deploy: bool = True,
        auto_cleanup: bool = False,
        resume: bool = False,
    ):
        self.config = config
        self.auto_deploy = auto_deploy
        self.auto_cleanup = auto_cleanup
        self.resume = resume
        self.proxy: Optional[ProxyManager] = None
        self._cleanup_done = False
        self._signal_received = False
        self._original_sigint = None
        self._original_sigterm = None

    def run(self) -> int:
        """Main entry point with guaranteed cleanup."""
        self._register_signal_handlers()

        try:
            # Auto-deploy if needed
            if self.auto_deploy and not ModalLifecycle.is_deployed(self.config.app_name):
                print(f"Deploying {self.config.app_name}...")
                # Build BenchmarkConfig from RunConfig for deployment when overrides present
                benchmark_config = None
                if self.config.config_hash:
                    benchmark_config = BenchmarkConfig(
                        name=self.config.benchmark_name,
                        env_id=self.config.env_id,
                        infra=self.config.infra,
                        vllm=self.config.vllm,
                        sampling=self.config.sampling,
                        eval=self.config.eval,
                        chat_template=self.config.chat_template,
                        env_args=self.config.env_args,
                    )
                if not ModalLifecycle.deploy(
                    self.config.model_id,
                    self.config.benchmark_name,
                    benchmark_config=benchmark_config,
                ):
                    raise RuntimeError("Deployment failed")

            # Start proxy (ephemeral port)
            self.proxy = ProxyManager(self.config.modal_url)
            port = self.proxy.start()

            # Run vf-eval
            return self._run_vf_eval(port)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            return 130
        finally:
            self._cleanup()
            self._restore_signal_handlers()

    def _register_signal_handlers(self):
        """Handle SIGINT/SIGTERM gracefully."""
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

        def handler(signum, frame):
            if self._signal_received:
                # Force exit on second signal
                print("\nForce exit...")
                os._exit(128 + signum)
            self._signal_received = True
            # Raise KeyboardInterrupt to trigger normal cleanup path
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _cleanup(self):
        """Terminate proxy and release resources."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        if self.proxy:
            self.proxy.stop(timeout=30)

        if self.auto_cleanup:
            print(f"Stopping Modal app {self.config.app_name}...")
            ModalLifecycle.stop_by_name(self.config.app_name)

    def _run_vf_eval(self, port: int) -> int:
        """Run vf-eval with all parameters including sampling."""
        base_url = f"http://localhost:{port}/v1"
        env_id = self.config.vf_eval_env_id

        cmd = [
            sys.executable,
            "-m",
            "verifiers.scripts.eval",
            env_id,
            "-m",
            self.config.model_id,
            "-b",
            base_url,
            "-k",
            "EMPTY",
            "-r",
            str(self.config.eval.rollouts),
            "-t",
            str(self.config.sampling.max_tokens),
            "--max-concurrent",
            str(self.config.eval.max_concurrent),
            "--temperature",
            str(self.config.sampling.temperature),
            "--checkpoint-every",
            str(self.config.eval.checkpoint_every),
            "--save-results",
            "--output-dir",
            self.config.checkpoint_dir,
        ]

        # Add top_p via sampling-args if not default
        if self.config.sampling.top_p != 1.0:
            sampling_args = {"top_p": self.config.sampling.top_p}
            cmd.extend(["--sampling-args", json.dumps(sampling_args)])

        # Default to -1 (all examples) if not specified
        num_examples = self.config.eval.num_examples if self.config.eval.num_examples is not None else -1
        cmd.extend(["-n", str(num_examples)])

        if self.resume:
            cmd.append("--resume")
            cmd.extend(["--resume-from", self.config.checkpoint_dir])

        if self.config.env_args:
            # Isolate Modal sandbox pools per run by default.
            # Without this, concurrent evals will each create their own pool under
            # the same Modal app name ("livecodebench-sandboxes"), making container
            # counts confusing and making cleanup/stop operations riskier.
            env_args = dict(self.config.env_args)
            if env_args.get("sandbox_backend") == "modal" and "modal_app_name" not in env_args:
                # Stable per-run identifier: the checkpoint directory suffix is our run_id.
                run_id = Path(self.config.checkpoint_dir).name
                env_args["modal_app_name"] = f"livecodebench-sandboxes-{run_id}"

            cmd.extend(["--env-args", json.dumps(env_args)])

        print(f"\n{'='*60}")
        print(f"Model:      {self.config.model_id}")
        print(f"Benchmark:  {self.config.benchmark_name} (env: {env_id})")
        print(f"Infra:      GPU={self.config.infra.gpu}")
        print(f"Sampling:   temp={self.config.sampling.temperature}, top_p={self.config.sampling.top_p}")
        print(f"Checkpoint: {self.config.checkpoint_dir}")
        print(f"{'='*60}")
        print(f"Running: {' '.join(cmd)}\n")

        return subprocess.call(cmd)
