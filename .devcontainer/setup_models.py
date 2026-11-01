from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DLC_ROOT = Path(os.environ.get("DLC_ENGINE", ROOT / "Deep-Live-Cam"))
MODEL_CACHE = ROOT / "work" / "models" / "rembg"
os.environ["U2NET_HOME"] = str(MODEL_CACHE)
sys.path.insert(0, str(DLC_ROOT))

from modules.model_downloader import MODEL_SIZES, ensure_model, local_path


def install_deep_live_cam_models() -> None:
    insightface_cache = Path.home() / ".insightface" / "models" / "buffalo_l"
    model_names = [
        "inswapper_128.onnx",
        "buffalo_l/buffalo_l/1k3d68.onnx",
        "buffalo_l/buffalo_l/2d106det.onnx",
        "buffalo_l/buffalo_l/det_10g.onnx",
        "buffalo_l/buffalo_l/genderage.onnx",
        "buffalo_l/buffalo_l/w600k_r50.onnx",
    ]
    for model_name in model_names:
        destination_dir = str(insightface_cache) if model_name.startswith("buffalo_l/") else None
        path = Path(local_path(model_name, destination_dir))
        expected_size = MODEL_SIZES[model_name]
        if path.is_file() and path.stat().st_size != expected_size:
            path.unlink()
        downloaded = ensure_model(model_name, dest_dir=destination_dir)
        if not downloaded or Path(downloaded).stat().st_size != expected_size:
            raise RuntimeError(f"Could not install and verify the required model: {path}")


def verify_cpu_inference() -> None:
    import modules.globals as globals

    globals.execution_providers = ["CPUExecutionProvider"]
    from modules.face_analyser import get_face_analyser
    from modules.processors.frame.face_swapper import get_face_swapper

    if get_face_analyser() is None or get_face_swapper() is None:
        raise RuntimeError("Deep-Live-Cam models failed to load on CPU.")

    import numpy as np
    from PIL import Image
    from rembg import new_session, remove

    session = new_session("u2netp", providers=["CPUExecutionProvider"])
    result = remove(Image.new("RGB", (64, 64)), session=session)
    if not isinstance(result, Image.Image) or "CPUExecutionProvider" not in session.inner_session.get_providers():
        raise RuntimeError("U2NetP did not complete CPU inference.")
    if not np.asarray(result).size:
        raise RuntimeError("U2NetP returned an empty inference result.")
    print(f"Verified U2NetP CPU inference; model cache: {MODEL_CACHE}")


if __name__ == "__main__":
    install_deep_live_cam_models()
    verify_cpu_inference()
