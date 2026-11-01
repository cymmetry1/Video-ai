from __future__ import annotations

import os
import secrets
import shutil
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx


PROVIDER_ENV = {
    "KAGGLE": ("KAGGLE_WORKER_URL", "KAGGLE_WORKER_TOKEN"),
    "COLAB": ("COLAB_WORKER_URL", "COLAB_WORKER_TOKEN"),
    "HUGGINGFACE_ZEROGPU": ("HF_WORKER_URL", "HF_WORKER_TOKEN"),
}
UNSUPPORTED_PROVIDERS = {
    "HUGGINGFACE_ZEROGPU": (
        "Unsupported: ZeroGPU supports Gradio Spaces with GPU work inside @spaces.GPU "
        "functions; this FastAPI worker and long-running ONNX Runtime video pipeline "
        "have not been validated within its per-call allocation model."
    )
}
MAX_RESULT_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class WorkerConfig:
    provider: str
    url: str
    token: str = field(repr=False)


@dataclass(frozen=True)
class WorkerHealth:
    provider: str
    available: bool
    message: str
    qualities: tuple[str, ...] = ()
    gpu: dict[str, Any] | None = None


class GPUWorker(Protocol):
    provider: str

    def health(self) -> WorkerHealth: ...

    def submit_job(
        self, external_job_id: str, paths: dict[str, Any], settings: dict[str, Any]
    ) -> str: ...

    def job_status(self, job_id: str) -> dict[str, Any]: ...

    def download_result(self, job_id: str, destination: Path) -> None: ...

    def cancel_job(self, job_id: str) -> None: ...

    def cleanup_job(self, job_id: str) -> None: ...


class HttpGPUWorker:
    def __init__(self, config: WorkerConfig) -> None:
        self.provider = config.provider
        self.base_url = config.url.rstrip("/")
        self._token = config.token
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=httpx.Timeout(connect=4, read=15, write=300, pool=4),
            follow_redirects=False,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> WorkerHealth:
        try:
            response = self._client.get(self._url("/api/worker/health"))
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            return WorkerHealth(self.provider, False, f"Worker health check failed: {error}")
        if not isinstance(payload, dict) or payload.get("provider_ready") is not True:
            return WorkerHealth(self.provider, False, "Worker did not confirm GPU readiness.")
        capabilities = payload.get("capabilities")
        qualities = capabilities.get("qualities") if isinstance(capabilities, dict) else None
        gpu = payload.get("gpu")
        modes = capabilities.get("modes") if isinstance(capabilities, dict) else None
        if (
            not isinstance(modes, list)
            or "face_swap" not in modes
            or not isinstance(qualities, list)
            or not all(isinstance(item, str) for item in qualities)
        ):
            return WorkerHealth(self.provider, False, "Worker returned invalid capabilities.")
        return WorkerHealth(
            self.provider,
            True,
            str(payload.get("message") or "GPU worker is ready."),
            tuple(qualities),
            gpu if isinstance(gpu, dict) else None,
        )

    def submit_job(
        self, external_job_id: str, paths: dict[str, Any], settings: dict[str, Any]
    ) -> str:
        fields = {
            "external_job_id": external_job_id,
            "mode": settings["mode"],
            "quality": settings["quality"],
            "permission_confirmed": str(settings["permission_confirmed"]).lower(),
            "start_seconds": str(settings["start_seconds"]),
            "end_seconds": str(settings["end_seconds"]),
            "face_enhance": str(settings["face_enhance"]).lower(),
            "multiple_faces": str(settings["multiple_faces"]).lower(),
            "mouth_mask": str(settings["mouth_mask"]).lower(),
        }
        with ExitStack() as stack:
            files: dict[str, Any] = {
                "source_video": (
                    paths["video"].name,
                    stack.enter_context(paths["video"].open("rb")),
                    "application/octet-stream",
                )
            }
            if paths.get("face"):
                files["face_image"] = (
                    paths["face"].name,
                    stack.enter_context(paths["face"].open("rb")),
                    "application/octet-stream",
                )
            response = self._client.post(
                self._url("/api/worker/jobs"),
                data=fields,
                files=files,
                timeout=httpx.Timeout(connect=4, read=30, write=900, pool=4),
            )
        response.raise_for_status()
        payload = response.json()
        job_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("GPU worker returned an invalid job receipt.")
        return job_id

    def job_status(self, job_id: str) -> dict[str, Any]:
        response = self._client.get(self._url(f"/api/worker/jobs/{job_id}"))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("GPU worker returned invalid job status.")
        return payload

    def download_result(self, job_id: str, destination: Path) -> None:
        written = 0
        try:
            with self._client.stream(
                "GET", self._url(f"/api/worker/jobs/{job_id}/result"), timeout=300
            ) as response:
                response.raise_for_status()
                declared_size = response.headers.get("content-length")
                if declared_size:
                    try:
                        if int(declared_size) > MAX_RESULT_BYTES:
                            raise RuntimeError("GPU worker result exceeds the 1 GB limit.")
                    except ValueError as error:
                        raise RuntimeError("GPU worker returned an invalid result size.") from error
                with destination.open("wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        written += len(chunk)
                        if written > MAX_RESULT_BYTES:
                            raise RuntimeError("GPU worker result exceeds the 1 GB limit.")
                        output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        if written == 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError("GPU worker returned an empty result.")

    def cancel_job(self, job_id: str) -> None:
        response = self._client.delete(self._url(f"/api/worker/jobs/{job_id}"))
        response.raise_for_status()

    def cleanup_job(self, job_id: str) -> None:
        response = self._client.delete(self._url(f"/api/worker/jobs/{job_id}/result"))
        response.raise_for_status()


class WorkerRouter:
    def __init__(self, workers: list[GPUWorker] | None = None) -> None:
        self.workers = workers if workers is not None else self._configured_workers()

    @staticmethod
    def _configured_workers() -> list[GPUWorker]:
        workers: list[GPUWorker] = []
        for provider in ("KAGGLE", "COLAB", "HUGGINGFACE_ZEROGPU"):
            url_env, token_env = PROVIDER_ENV[provider]
            url, token = os.getenv(url_env, "").strip(), os.getenv(token_env, "")
            if provider in UNSUPPORTED_PROVIDERS or not url or not token:
                continue
            parsed = urlparse(url)
            local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            if (
                not parsed.hostname
                or parsed.path not in {"", "/"}
                or (parsed.scheme != "https" and not (local and parsed.scheme == "http"))
            ):
                continue
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                continue
            workers.append(
                HttpGPUWorker(
                    WorkerConfig(
                        provider=provider,
                        url=url,
                        token=token,
                    )
                )
            )
        return workers

    def health(self) -> list[WorkerHealth]:
        result = []
        for provider, message in UNSUPPORTED_PROVIDERS.items():
            result.append(WorkerHealth(provider, False, message))
        for worker in self.workers:
            result.append(worker.health())
        return result

    def ready_workers(self, mode: str, quality: str) -> list[GPUWorker]:
        if mode != "face_swap":
            return []
        ready = []
        for worker in self.workers:
            health = worker.health()
            if health.available and quality in health.qualities:
                ready.append(worker)
        return ready


def detect_gpu_runtime() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "message": "nvidia-smi is unavailable."}
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "message": f"Could not query NVIDIA GPU: {error}"}
    if result.returncode or not result.stdout.strip():
        return {"available": False, "message": "No NVIDIA GPU was assigned to this session."}
    try:
        name, memory = result.stdout.strip().splitlines()[0].split(",", maxsplit=1)
        memory_mb = int(memory.strip())
    except (ValueError, IndexError):
        return {"available": False, "message": "nvidia-smi returned invalid GPU information."}
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
    except (ImportError, OSError) as error:
        providers = []
        provider_error = str(error)
    else:
        provider_error = ""
    if "CUDAExecutionProvider" not in providers:
        return {
            "available": False,
            "message": (
                f"ONNX Runtime could not initialize: {provider_error}"
                if provider_error
                else "An NVIDIA GPU is present, but ONNX Runtime CUDA is unavailable."
            ),
            "gpu": {"name": name.strip(), "memory_mb": memory_mb},
        }
    return {
        "available": True,
        "message": "NVIDIA GPU and ONNX Runtime CUDA are available.",
        "gpu": {"name": name.strip(), "memory_mb": memory_mb},
        "providers": providers,
    }


def valid_worker_token(configured: str | None, supplied: str | None) -> bool:
    return bool(configured and supplied) and secrets.compare_digest(
        configured, supplied.removeprefix("Bearer ").strip()
    )


def provider_state(router: WorkerRouter) -> list[dict[str, Any]]:
    return [
        {
            "provider": status.provider,
            "available": status.available,
            "message": status.message,
            "qualities": list(status.qualities),
            "gpu": status.gpu,
        }
        for status in router.health()
    ]
