import os
import sys
import threading
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import app
from engines import EngineStatus
from gpu_workers import WorkerHealth, WorkerRouter, detect_gpu_runtime, valid_worker_token


class ReadyEngine:
    name = "Test face engine"
    root = Path.cwd()

    def status(self):
        return EngineStatus(True, "Ready", self.name)

    def command(self, **kwargs):
        raise AssertionError("This worker validation test must not start a job.")


class FakeWorker:
    def __init__(self, provider, available, qualities):
        self.provider = provider
        self.available = available
        self.qualities = qualities

    def health(self):
        return WorkerHealth(
            self.provider,
            self.available,
            "Ready" if self.available else "Offline",
            tuple(self.qualities),
        )

    def submit_job(self, external_job_id, paths, settings):
        raise AssertionError("This test only exercises readiness and routing.")

    def job_status(self, job_id):
        raise AssertionError("This test only exercises readiness and routing.")

    def download_result(self, job_id, destination):
        raise AssertionError("This test only exercises readiness and routing.")

    def cancel_job(self, job_id):
        raise AssertionError("This test only exercises readiness and routing.")

    def cleanup_job(self, job_id):
        raise AssertionError("This test only exercises readiness and routing.")


class GPUWorkerRouterTests(unittest.TestCase):
    def test_selects_first_healthy_compatible_worker_and_skips_unavailable(self):
        offline = FakeWorker("KAGGLE", False, ["fast", "balanced", "best"])
        first = FakeWorker("COLAB", True, ["fast", "balanced"])
        second = FakeWorker("KAGGLE", True, ["fast", "balanced", "best"])
        router = WorkerRouter([offline, first, second])

        self.assertEqual(router.ready_workers("face_swap", "balanced"), [first, second])
        self.assertEqual(router.ready_workers("face_swap", "best"), [second])
        self.assertEqual(router.ready_workers("background_swap", "fast"), [])

    def test_worker_reconnect_is_discovered_on_the_next_health_check(self):
        worker = FakeWorker("COLAB", False, ["fast"])
        router = WorkerRouter([worker])
        self.assertEqual(router.ready_workers("face_swap", "fast"), [])
        worker.available = True
        self.assertEqual(router.ready_workers("face_swap", "fast"), [worker])

    def test_config_ignores_plain_http_remote_workers_and_unsupported_zerogpu(self):
        environment = {
            "KAGGLE_WORKER_URL": "http://worker.example",
            "KAGGLE_WORKER_TOKEN": "secret",
            "COLAB_WORKER_URL": "https://colab.example",
            "COLAB_WORKER_TOKEN": "secret",
            "HF_WORKER_URL": "https://hf.example",
            "HF_WORKER_TOKEN": "secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            router = WorkerRouter()

        self.assertEqual([worker.provider for worker in router.workers], ["COLAB"])

    def test_token_check_rejects_missing_or_incorrect_credentials(self):
        self.assertTrue(valid_worker_token("abc", "Bearer abc"))
        self.assertFalse(valid_worker_token("abc", None))
        self.assertFalse(valid_worker_token("abc", "Bearer other"))
        self.assertFalse(valid_worker_token("", "Bearer abc"))

    def test_gpu_readiness_requires_detected_device_and_cuda_provider(self):
        with patch("gpu_workers.shutil.which", return_value=None):
            self.assertFalse(detect_gpu_runtime()["available"])

        fake_nvidia_smi = SimpleNamespace(
            returncode=0,
            stdout="NVIDIA Test GPU, 8192\n",
        )
        fake_onnxruntime = SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        with (
            patch("gpu_workers.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("gpu_workers.subprocess.run", return_value=fake_nvidia_smi),
            patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}),
        ):
            status = detect_gpu_runtime()
        self.assertTrue(status["available"])
        self.assertEqual(status["gpu"]["name"], "NVIDIA Test GPU")

        fake_onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]
        with (
            patch("gpu_workers.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("gpu_workers.subprocess.run", return_value=fake_nvidia_smi),
            patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}),
        ):
            self.assertFalse(detect_gpu_runtime()["available"])

    def test_worker_submission_rejects_unauthenticated_health(self):
        with TestClient(app.app) as client:
            response = client.get("/api/worker/health")
        self.assertEqual(response.status_code, 401)

    def test_worker_only_mode_hides_the_app_and_non_worker_api(self):
        with (
            patch.dict(os.environ, {"VIDEO_AI_WORKER_ONLY": "1"}),
            TestClient(app.app) as client,
        ):
            self.assertEqual(client.get("/").status_code, 404)
            self.assertEqual(client.get("/api/health").status_code, 404)
            self.assertEqual(client.get("/openapi.json").status_code, 404)
            self.assertEqual(client.get("/api/worker/health").status_code, 401)

    def test_worker_rejects_mismatched_file_content(self):
        with (
            patch.dict(
                os.environ,
                {
                    "VIDEO_AI_WORKER_TOKEN": "test-token",
                    "EXECUTION_PROVIDER": "cuda",
                    "VIDEO_AI_EXECUTION_PROVIDER": "cuda",
                },
            ),
            patch.dict(app.ENGINE_REGISTRY, {"face_swap": ReadyEngine()}),
            patch("app.detect_gpu_runtime", return_value={"available": True, "message": "GPU ready"}),
            TestClient(app.app) as client,
        ):
            response = client.post(
                "/api/worker/jobs",
                headers={"Authorization": "Bearer test-token"},
                data={
                    "external_job_id": "a" * 32,
                    "mode": "face_swap",
                    "quality": "fast",
                    "permission_confirmed": "true",
                },
                files=[
                    ("source_video", ("clip.mp4", b"not a video", "video/mp4")),
                    ("face_image", ("face.jpg", b"not an image", "image/jpeg")),
                ],
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("contents do not match", response.json()["detail"])

    def test_healthy_gpu_worker_is_selected_without_cpu_fallback_or_duplicate_submit(self):
        class AcceptedThenLostWorker(FakeWorker):
            def __init__(self):
                super().__init__("KAGGLE", True, ["fast"])
                self.submissions = 0

            def submit_job(self, external_job_id, paths, settings):
                self.submissions += 1
                self.asserted_settings = settings
                return "remote-id"

            def job_status(self, job_id):
                raise ConnectionError("session ended")

        class UnavailableCpuEngine:
            name = "CPU unavailable"
            root = Path.cwd()

            def status(self):
                return EngineStatus(False, "CPU unavailable", self.name)

            def command(self, **kwargs):
                raise AssertionError("The GPU-ready job must not run on CPU.")

        worker = AcceptedThenLostWorker()
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(app, "JOBS_DIR", Path(temporary)),
                patch.object(app, "GPU_ROUTER", WorkerRouter([worker])),
                patch.dict(app.ENGINE_REGISTRY, {"face_swap": UnavailableCpuEngine()}),
                TestClient(app.app) as client,
            ):
                response = client.post(
                    "/api/jobs",
                    data={"mode": "face_swap", "quality": "fast", "permission_confirmed": "true"},
                    files=[
                        ("source_video", ("clip.mp4", b"\x00\x00\x00\x18ftypisom0000", "video/mp4")),
                        ("face_image", ("face.jpg", b"\xff\xd8\xffimage", "image/jpeg")),
                    ],
                )
                self.assertEqual(response.status_code, 202, response.text)
                job_id = response.json()["id"]
                self.assertEqual(response.json()["worker"], "KAGGLE")
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    job = client.get(f"/api/jobs/{job_id}").json()
                    if job["status"] == "worker_lost":
                        break
                    time.sleep(0.01)
                self.assertEqual(job["status"], "worker_lost")
                app.JOBS.pop(job_id, None)
                app.CANCEL_EVENTS.pop(job_id, None)

        self.assertEqual(worker.submissions, 1)

    def test_remote_failure_is_reported_without_resubmitting(self):
        class LostWorker(FakeWorker):
            def __init__(self):
                super().__init__("KAGGLE", True, ["fast"])
                self.submissions = 0

            def submit_job(self, external_job_id, paths, settings):
                self.submissions += 1
                return "remote-id"

            def job_status(self, job_id):
                raise ConnectionError("session ended")

        worker = LostWorker()
        job_id = "b" * 32
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(app, "JOBS_DIR", Path(temporary)):
                (app.JOBS_DIR / job_id).mkdir()
                app.JOBS[job_id] = app.Job(id=job_id, mode="face_swap")
                app.CANCEL_EVENTS[job_id] = threading.Event()
                app._process_remote_job(
                    job_id,
                    worker,
                    {"video": Path("source.mp4"), "face": Path("face.jpg"), "references": []},
                    {
                        "mode": "face_swap",
                        "quality": "fast",
                        "start_seconds": 0,
                        "end_seconds": 0,
                        "face_enhance": False,
                        "multiple_faces": False,
                        "mouth_mask": False,
                    },
                )
                status = app.JOBS[job_id].status
                app.JOBS.pop(job_id, None)
                app.CANCEL_EVENTS.pop(job_id, None)

        self.assertEqual(worker.submissions, 1)
        self.assertEqual(status, "worker_lost")
