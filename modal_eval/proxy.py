"""
Local proxy that forwards requests to Modal and follows 303 redirects.

This works around Modal's 150s web endpoint timeout by following the
redirect chain that Modal provides for long-running requests.

Usage:
    # From config (new format with HuggingFace model ID)
    python -m modal_eval.proxy --model allenai/Olmo-3-7B-Think --profile default

    # Ad-hoc
    python -m modal_eval.proxy \
        --modal-url https://ota-merge--mk-custom-serve.modal.run \
        --port 8090
"""

import argparse
import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("proxy")

# Constants for retry logic
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # seconds, exponential backoff

# Track active requests for debugging
active_requests = {}


def get_proxy_config(args) -> tuple[str, int]:
    """Get Modal URL and local port from args or config."""
    if args.modal_url and args.port:
        return args.modal_url, args.port

    if args.model:
        from modal_eval.models import model_id_to_slug

        slug = model_id_to_slug(args.model)  # args.model is now HuggingFace model ID
        profile = args.profile or "default"

        # Build app name and URL
        app_name = f"mk-{slug}-{profile}"
        modal_url = f"https://ota-merge--{app_name}-serve.modal.run"

        # Use provided port or find a free one
        port = args.port or 0  # 0 means let OS pick
        return modal_url, port

    raise ValueError("Must specify --model or (--modal-url and --port)")


def create_app(modal_url: str) -> FastAPI:
    """Create FastAPI app with proxy routes using a shared HTTP client."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage shared HTTP client lifecycle."""
        # Timeout tuned for Modal's behavior:
        # - connect: 60s to handle cold start container spin-up
        # - read: 1800s (30 min) for very long generation (32k tokens)
        # - pool: 30s to allow for temporary pool exhaustion
        timeout = httpx.Timeout(
            connect=60.0,
            read=1800.0,  # 30 minutes for long generations
            write=30.0,
            pool=30.0,
        )

        # Connection limits for high concurrency
        # vf-eval uses max_concurrent=32, so we need headroom
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=50,
            keepalive_expiry=30.0,
        )

        # Create shared client at startup
        app.state.client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            follow_redirects=True,
            max_redirects=30,
        )
        app.state.request_counter = 0
        logger.info(f"Initialized HTTP client (max_connections={limits.max_connections})")
        logger.info(f"Timeouts: connect={timeout.connect}s, read={timeout.read}s, pool={timeout.pool}s")
        yield
        # Clean up on shutdown
        await app.state.client.aclose()
        logger.info("Closed shared HTTP client")

    app = FastAPI(lifespan=lifespan)

    @app.get("/proxy/status")
    async def status(request: Request):
        """Return proxy status and active requests."""
        now = time.time()
        active = []
        for req_id, info in active_requests.items():
            elapsed = now - info["start"]
            active.append({
                "id": req_id,
                "path": info["path"],
                "elapsed_seconds": round(elapsed, 1),
            })
        return {
            "total_requests": getattr(request.app.state, "request_counter", 0),
            "active_requests": len(active_requests),
            "requests": sorted(active, key=lambda x: -x["elapsed_seconds"]),
        }

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy(request: Request, path: str):
        """Proxy requests to Modal with retry logic for transient failures."""
        target_url = f"{modal_url}/{path}"

        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)

        client = getattr(request.app.state, "client", None)
        if client is None:
            logger.error("Proxy client not initialized")
            return Response(content="Proxy not ready", status_code=503)

        # Track request
        request.app.state.request_counter += 1
        req_id = request.app.state.request_counter
        start_time = time.time()
        active_requests[req_id] = {"path": path, "start": start_time}

        # Log request start (skip noisy /v1/models health checks)
        if path != "v1/models":
            logger.info(f"[{req_id}] START {request.method} /{path} (active: {len(active_requests)})")

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body,
                    params=request.query_params,
                )

                # Log completion
                elapsed = time.time() - start_time
                active_requests.pop(req_id, None)
                if path != "v1/models":
                    logger.info(f"[{req_id}] DONE {response.status_code} in {elapsed:.1f}s (active: {len(active_requests)})")

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            except (httpx.ConnectError, httpx.PoolTimeout) as e:
                # Retry on connection errors and pool exhaustion
                last_error = e
                delay = RETRY_DELAY_BASE * (2**attempt)
                logger.warning(f"[{req_id}] Attempt {attempt + 1}/{MAX_RETRIES} failed: {type(e).__name__}: {e}, retrying in {delay}s")
                await asyncio.sleep(delay)
            except httpx.TimeoutException as e:
                # Don't retry on read timeouts (request was sent, might be processing)
                elapsed = time.time() - start_time
                active_requests.pop(req_id, None)
                logger.error(f"[{req_id}] TIMEOUT after {elapsed:.1f}s: {type(e).__name__}: {e}")
                return Response(content=f"Proxy timeout: {e}", status_code=504)
            except Exception as e:
                elapsed = time.time() - start_time
                active_requests.pop(req_id, None)
                logger.error(f"[{req_id}] ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
                return Response(content=f"Proxy error: {e}", status_code=502)

        # All retries exhausted
        elapsed = time.time() - start_time
        active_requests.pop(req_id, None)
        logger.error(f"[{req_id}] FAILED after {MAX_RETRIES} retries in {elapsed:.1f}s: {last_error}")
        return Response(content=f"Proxy failed after {MAX_RETRIES} retries: {last_error}", status_code=502)

    return app


def main():
    parser = argparse.ArgumentParser(description="Local proxy for Modal endpoints")
    parser.add_argument("--model", "-m", help="HuggingFace model ID (e.g., allenai/Olmo-3-7B-Think)")
    parser.add_argument("--profile", help="Server profile (default: 'default')")
    parser.add_argument("--modal-url", help="Modal endpoint URL (ad-hoc)")
    parser.add_argument("--port", "-p", type=int, help="Local port")

    args = parser.parse_args()

    modal_url, port = get_proxy_config(args)

    # If port is 0 or not specified, find a free port
    if not port:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]

    print("=" * 60)
    print("Modal Eval - Proxy Server")
    print("=" * 60)
    print(f"Forwarding to: {modal_url}")
    print(f"Local endpoint: http://localhost:{port}/v1")
    print("This proxy follows 303 redirects (Modal's timeout workaround)")
    print("Uses shared connection pool for efficiency")
    print("=" * 60)

    app = create_app(modal_url)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
