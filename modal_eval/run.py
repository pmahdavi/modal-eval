"""
Eval orchestrator - runs vf-eval with automatic proxy management.

Usage:
    # Run eval with model from config
    python -m modal_eval.run --model olmo3-math-code --env math500 -n 500

    # Run eval with ad-hoc model
    python -m modal_eval.run \
        --model-id "user/model" \
        --modal-url https://ota-merge--mk-custom-serve.modal.run \
        --proxy-port 8090 \
        --env gsm8k -n 100

    # Multiple environments
    python -m modal_eval.run --model olmo3-math-code --env math500,gsm8k -n 500,1319
"""

import argparse
import atexit
import signal
import subprocess
import sys
import time

import httpx

from modal_eval import load_config


def get_model_config(args) -> dict:
    """Get model configuration from args or config file."""
    config = load_config()
    defaults = config.get("defaults", {})
    eval_defaults = config.get("eval_defaults", {})

    if args.model:
        if args.model not in config.get("models", {}):
            raise ValueError(f"Model '{args.model}' not found in config.yaml")
        model_config = {**defaults, **config["models"][args.model]}
        model_config["name"] = args.model
    elif args.model_id:
        model_config = {
            **defaults,
            "model_id": args.model_id,
            "name": "adhoc",
        }
        if args.modal_url:
            # Extract app suffix from URL
            model_config["modal_url"] = args.modal_url
        if args.proxy_port:
            model_config["proxy_port"] = args.proxy_port
    else:
        raise ValueError("Must specify --model or --model-id")

    # Merge eval defaults
    model_config["eval"] = eval_defaults
    return model_config


def get_modal_url(model_config: dict) -> str:
    """Get Modal endpoint URL for the model."""
    if "modal_url" in model_config:
        return model_config["modal_url"]

    app_suffix = model_config.get("app_suffix", model_config["name"])
    return f"https://ota-merge--mk-{app_suffix}-serve.modal.run"


def wait_for_server(url: str, timeout: int = 300) -> bool:
    """Wait for server to be ready."""
    print(f"Waiting for server at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(f"{url}/v1/models", timeout=5)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_proxy(model_config: dict) -> tuple[subprocess.Popen, str]:
    """Start the proxy server in the background. Returns (proc, log_path)."""
    port = model_config.get("proxy_port", 8080)
    modal_url = get_modal_url(model_config)

    cmd = [
        sys.executable, "-m", "modal_eval.proxy",
        "--modal-url", modal_url,
        "--port", str(port),
    ]

    # Write proxy logs to a file for debugging
    log_path = f"proxy_{port}.log"
    log_file = open(log_path, "w")

    print(f"Starting proxy on port {port}...")
    print(f"Proxy logs: {log_path}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Store log file handle for cleanup
    proc._log_file = log_file
    proc._log_path = log_path

    # Wait for proxy to start
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError("Proxy failed to start")

    return proc, log_path


def run_vf_eval(
    model_id: str,
    base_url: str,
    env: str,
    num_examples: int,
    rollouts: int,
    max_tokens: int,
    max_concurrent: int,
    save_results: bool = True,
    resume: bool = False,
    resume_from: str | None = None,
    checkpoint_every: int = 1,
    env_args: str | None = None,
    hf_hub: bool = False,
    hf_hub_dataset_name: str | None = None,
    hf_hub_private: bool = False,
) -> int:
    """Run vf-eval with the specified parameters."""
    cmd = [
        sys.executable, "-m", "verifiers.scripts.eval", env,
        "-m", model_id,
        "-b", base_url,
        "-k", "EMPTY",
        "-n", str(num_examples),
        "-r", str(rollouts),
        "-t", str(max_tokens),
        "--max-concurrent", str(max_concurrent),
        "--checkpoint-every", str(checkpoint_every),
    ]

    if save_results:
        cmd.append("--save-results")

    # Resume support
    if resume:
        cmd.append("--resume")
        if resume_from:
            cmd.extend(["--resume-from", resume_from])

    if env_args:
        cmd.extend(["--env-args", env_args])

    # HF Hub options
    if hf_hub:
        cmd.append("--save-to-hf-hub")
        if hf_hub_dataset_name:
            cmd.extend(["--hf-hub-dataset-name", hf_hub_dataset_name])
        if hf_hub_private:
            cmd.append("--hf-hub-private")

    print(f"\nRunning: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run evaluations with auto proxy")

    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", "-m", help="Model name from config.yaml")
    model_group.add_argument("--model-id", help="HuggingFace model ID (ad-hoc)")

    # Ad-hoc options
    parser.add_argument("--modal-url", help="Modal endpoint URL (ad-hoc)")
    parser.add_argument("--proxy-port", type=int, help="Local proxy port (ad-hoc)")

    # Eval options
    parser.add_argument("--env", "-e", required=True, help="Environment(s) to eval (comma-separated)")
    parser.add_argument("-n", "--num-examples", type=str, help="Number of examples (comma-separated for multiple envs)")
    parser.add_argument("-r", "--rollouts", type=int, help="Rollouts per example")
    parser.add_argument("-t", "--max-tokens", type=int, help="Max tokens per response")
    parser.add_argument("-c", "--max-concurrent", type=int, help="Max concurrent requests")
    parser.add_argument(
        "--env-args",
        type=str,
        help="JSON string of additional environment kwargs to forward to vf-eval (e.g. '{\"sandbox_backend\":\"modal\"}')",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--no-proxy", action="store_true", help="Don't start proxy (use existing)")

    # Resume support
    parser.add_argument(
        "--resume", "-R",
        action="store_true",
        help="Resume from previous checkpoint",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N example groups (default: 1)",
    )

    # HF Hub options
    parser.add_argument(
        "--hf-hub", "-H",
        action="store_true",
        help="Push results to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-hub-dataset-name", "-D",
        type=str,
        help="Dataset name on HF Hub (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--hf-hub-private", "-P",
        action="store_true",
        help="Make the HF Hub dataset private",
    )

    args = parser.parse_args()

    # Get model config
    model_config = get_model_config(args)
    eval_defaults = model_config.get("eval", {})

    # Parse environments and example counts
    envs = args.env.split(",")
    if args.num_examples:
        num_examples_list = [int(n) for n in args.num_examples.split(",")]
        if len(num_examples_list) == 1:
            num_examples_list = num_examples_list * len(envs)
    else:
        # Default example counts per environment
        default_counts = {"math500": 500, "gsm8k": 1319}
        num_examples_list = [default_counts.get(e, 100) for e in envs]

    if len(num_examples_list) != len(envs):
        raise ValueError("Number of --num-examples must match number of --env")

    # Get eval parameters
    rollouts = args.rollouts or eval_defaults.get("rollouts", 4)
    max_tokens = args.max_tokens or eval_defaults.get("max_tokens", 8192)
    max_concurrent = args.max_concurrent or eval_defaults.get("max_concurrent", 32)

    port = model_config.get("proxy_port", 8080)
    base_url = f"http://localhost:{port}/v1"

    proxy_proc = None
    proxy_log_path = None

    def cleanup():
        """Clean up proxy process on exit."""
        if proxy_proc and proxy_proc.poll() is None:
            print("\nStopping proxy...")
            proxy_proc.terminate()
            try:
                proxy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_proc.kill()
            # Close log file
            if hasattr(proxy_proc, '_log_file'):
                proxy_proc._log_file.close()
                print(f"Proxy logs saved to: {proxy_proc._log_path}")

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    # Start proxy if needed
    if not args.no_proxy:
        proxy_proc, proxy_log_path = start_proxy(model_config)

        # Wait for proxy to be ready
        if not wait_for_server(f"http://localhost:{port}", timeout=60):
            print("Warning: Proxy may not be ready, continuing anyway...")

    # Run evaluations
    print(f"\n{'=' * 60}")
    print("Modal Eval - Running Evaluations")
    print(f"{'=' * 60}")
    print(f"Model:      {model_config['model_id']}")
    print(f"Endpoint:   {base_url}")
    print(f"Envs:       {', '.join(envs)}")
    print(f"Examples:   {', '.join(map(str, num_examples_list))}")
    print(f"Rollouts:   {rollouts}")
    print(f"Max tokens: {max_tokens}")
    print(f"{'=' * 60}\n")

    for env, num_examples in zip(envs, num_examples_list):
        print(f"\n>>> Running {env} ({num_examples} examples, {rollouts} rollouts)...\n")
        exit_code = run_vf_eval(
            model_id=model_config["model_id"],
            base_url=base_url,
            env=env,
            num_examples=num_examples,
            rollouts=rollouts,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            save_results=not args.no_save,
            resume=args.resume,
            resume_from=args.resume_from,
            checkpoint_every=args.checkpoint_every,
            env_args=args.env_args,
            hf_hub=args.hf_hub,
            hf_hub_dataset_name=args.hf_hub_dataset_name,
            hf_hub_private=args.hf_hub_private,
        )
        if exit_code != 0:
            print(f"Warning: {env} eval exited with code {exit_code}")

    print("\n>>> All evaluations complete!")


if __name__ == "__main__":
    main()
