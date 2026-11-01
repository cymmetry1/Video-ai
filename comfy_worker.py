from __future__ import annotations

import argparse
import http.client
import json
import signal
import sys
import mimetypes
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import urlopen


def _request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    from urllib.request import Request

    request = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _cancel_remote_job(base_url: str, prompt_id: str) -> None:
    queue = _request_json(f"{base_url}/queue")
    running = queue.get("queue_running", [])
    pending = queue.get("queue_pending", [])
    is_running = any(len(item) > 1 and str(item[1]) == prompt_id for item in running)
    is_pending = any(len(item) > 1 and str(item[1]) == prompt_id for item in pending)
    if is_running:
        _request_json(f"{base_url}/interrupt", {})
    elif is_pending:
        _request_json(f"{base_url}/queue", {"delete": [prompt_id]})


def _upload(base_url: str, path: Path) -> str:
    parsed = urlsplit(base_url)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=120)
    boundary = f"----video-ai-{uuid.uuid4().hex}"
    filename = path.name.replace('"', "")
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    route = f"{parsed.path.rstrip('/')}/upload/image?type=input&overwrite=true"
    connection.putrequest("POST", route)
    connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
    connection.putheader("Content-Length", str(len(prefix) + path.stat().st_size + len(suffix)))
    connection.endheaders()
    connection.send(prefix)
    with path.open("rb") as media:
        while chunk := media.read(1024 * 1024):
            connection.send(chunk)
    connection.send(suffix)
    response = connection.getresponse()
    body = response.read()
    connection.close()
    if response.status >= 400:
        raise RuntimeError(f"ComfyUI upload failed ({response.status}): {body.decode(errors='replace')[-1000:]}")
    uploaded = json.loads(body)
    name = str(uploaded.get("name", ""))
    if not name:
        raise RuntimeError("ComfyUI did not return an uploaded filename.")
    subfolder = str(uploaded.get("subfolder", "")).strip("/")
    return f"{subfolder}/{name}" if subfolder else name


def _assign(workflow: dict, mapping: dict, value: str) -> None:
    node_id = str(mapping.get("node", ""))
    field = str(mapping.get("field", ""))
    if node_id not in workflow or not field:
        raise RuntimeError("The workflow input mapping refers to a missing node or field.")
    workflow[node_id].setdefault("inputs", {})[field] = value


def _download_result(base_url: str, history: dict, output_path: Path, output_nodes: list[str]) -> None:
    outputs = history.get("outputs", {})
    candidates = []
    selected_nodes = set(output_nodes)
    for node_id, node_output in outputs.items():
        if selected_nodes and node_id not in selected_nodes:
            continue
        if not isinstance(node_output, dict):
            continue
        for key in ("videos", "gifs"):
            candidates.extend(
                item
                for item in node_output.get(key, [])
                if isinstance(item, dict)
                and Path(str(item.get("filename", ""))).suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
            )
    if not candidates:
        raise RuntimeError("The configured ComfyUI workflow finished without a video output.")
    item = candidates[0]
    query = (
        f"filename={quote(str(item.get('filename', '')))}"
        f"&subfolder={quote(str(item.get('subfolder', '')))}"
        f"&type={quote(str(item.get('type', 'output')))}"
    )
    with urlopen(f"{base_url}/view?{query}", timeout=120) as response, output_path.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    if output_path.stat().st_size == 0:
        raise RuntimeError("ComfyUI returned an empty video output.")


def run_job(
    base_url: str,
    workflow_path: Path,
    video_path: Path,
    references: list[Path],
    prompt: str,
    output_path: Path,
) -> None:
    config = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow = config["workflow"]
    inputs = config["inputs"]
    _assign(workflow, inputs["video"], _upload(base_url, video_path))

    reference_mappings = inputs.get("references", [])
    if references and len(reference_mappings) < len(references):
        raise RuntimeError("The configured workflow has fewer reference inputs than the uploaded images.")
    for mapping, reference in zip(reference_mappings, references):
        _assign(workflow, mapping, _upload(base_url, reference))

    prompt_mapping = inputs.get("prompt")
    if prompt_mapping and prompt.strip():
        _assign(workflow, prompt_mapping, prompt.strip())

    prompt_id = _request_json(
        f"{base_url}/prompt",
        {"prompt": workflow, "client_id": str(uuid.uuid4())},
    ).get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI did not return a job identifier.")

    def cancel(signum: int, _frame: object) -> None:
        try:
            _cancel_remote_job(base_url, str(prompt_id))
        except (OSError, ValueError, KeyError) as error:
            print(f"Could not cancel ComfyUI job {prompt_id}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        history = _request_json(f"{base_url}/history/{quote(str(prompt_id))}")
        result = history.get(str(prompt_id))
        if result:
            status = result.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI workflow failed: {status.get('messages', [])}")
            if status.get("completed"):
                _download_result(base_url, result, output_path, [str(node) for node in config.get("outputs", [])])
                return
        time.sleep(2)
    raise TimeoutError("ComfyUI processing exceeded the one-hour limit.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--references", default="[]")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    references = [Path(path) for path in json.loads(arguments.references)]
    run_job(
        arguments.url.rstrip("/"),
        arguments.workflow,
        arguments.video,
        references,
        arguments.prompt,
        arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
