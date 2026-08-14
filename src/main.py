"""Container entrypoint: spawn `vllm serve`, wait for it, start RunPod serverless.

The worker never imports vLLM. We build the CLI from environment variables
(see args_builder.py), launch `vllm serve` on the loopback interface, poll
/health until the server (and model) is ready, and only then start the RunPod
serverless job loop so no job is pulled before the backend can serve it.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from args_builder import build_vllm_args
from download_model import LOCAL_MODEL_ARGS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

VLLM_HOST = "127.0.0.1"
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
STARTUP_TIMEOUT = int(os.getenv("VLLM_STARTUP_TIMEOUT", "1200"))
HEALTH_POLL_INTERVAL = 2

CIT_MISTRAL_MODEL_MARKER = "mistral-small-3.1-24b-instruct-2503"

vllm_process: subprocess.Popen | None = None


def _truthy_env(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def configure_cit_tokenizer() -> None:
    """Keep Mistral3 on its HF processor and patch the regex at load time.

    Forcing ``tokenizer_mode=mistral`` with a separate Mistral tokenizer breaks
    PixtralProcessor startup because the multimodal processor expects HF image
    special tokens. ``sitecustomize.py`` patches both AutoTokenizer and
    AutoProcessor with ``fix_mistral_regex=True`` instead, so use auto mode and
    let model+processor come from the same GPTQ repository.
    """
    model_name = os.getenv("MODEL_NAME", "")
    if CIT_MISTRAL_MODEL_MARKER not in model_name.lower():
        return
    if not _truthy_env("CIT_FIX_MISTRAL_REGEX"):
        return

    os.environ["TOKENIZER_MODE"] = "auto"
    os.environ.pop("TOKENIZER_NAME", None)
    os.environ.pop("TOKENIZER_REVISION", None)
    logging.info("CIT tokenizer configured: HF processor + regex patch, tokenizer_mode=auto")


def apply_local_model_args() -> None:
    if not os.path.exists(LOCAL_MODEL_ARGS_PATH):
        return
    try:
        with open(LOCAL_MODEL_ARGS_PATH) as f:
            local_args = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.error("Failed to read %s: %s", LOCAL_MODEL_ARGS_PATH, e)
        return

    logging.info("Using baked-in model args: %s", local_args)
    for key, value in local_args.items():
        if value not in (None, ""):
            os.environ[key] = str(value)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def start_vllm() -> subprocess.Popen:
    argv = ["vllm", "serve", "--host", VLLM_HOST, "--port", VLLM_PORT]
    argv += build_vllm_args()
    logging.info("Starting vLLM: %s", " ".join(argv))
    return subprocess.Popen(argv)


def wait_for_vllm(proc: subprocess.Popen) -> None:
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM serve exited during startup with code {proc.returncode}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as resp:
                if resp.status == 200:
                    logging.info("vLLM is healthy")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    raise RuntimeError(f"vLLM did not become healthy within {STARTUP_TIMEOUT}s")


def _forward_signal(signum, _frame):
    logging.info("Received signal %s, shutting down vLLM", signum)
    if vllm_process and vllm_process.poll() is None:
        vllm_process.send_signal(signum)
    sys.exit(128 + signum)


def main() -> None:
    global vllm_process

    apply_local_model_args()
    configure_cit_tokenizer()

    if not (os.getenv("MODEL_NAME") or os.getenv("VLLM_CONFIG_FILE") or os.getenv("MODEL")):
        logging.warning("MODEL_NAME is not set; `vllm serve` will fail without a --model argument")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _forward_signal)

    vllm_process = start_vllm()
    try:
        wait_for_vllm(vllm_process)
    except RuntimeError as e:
        logging.error("%s", e)
        sys.exit(1)

    import handler as proxy_handler
    import runpod

    proxy_handler.vllm_process = vllm_process

    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "30"))
    runpod.serverless.start(
        {
            "handler": proxy_handler.handler,
            "concurrency_modifier": lambda _current: max_concurrency,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
