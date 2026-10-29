from pathlib import Path
import os
import shutil
import subprocess
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent.resolve()
WORK = ROOT / "work"
UPLOADS = WORK / "uploads"
OUTPUTS = WORK / "outputs"
ENGINE = Path(os.getenv("DLC_ENGINE", ROOT / "Deep-Live-Cam"))
for p in (UPLOADS, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Video AI Research")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/api/health")
def health():
    return {"ok": True, "engine_found": (ENGINE / "run.py").exists()}

async def save_upload(file: UploadFile, suffixes: set[str], folder: Path) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in suffixes:
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    path = folder / f"{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return path

@app.post("/api/swap")
async def swap(source: UploadFile = File(...), target: UploadFile = File(...)):
    if not (ENGINE / "run.py").exists():
        raise HTTPException(503, "Deep-Live-Cam engine is not installed on this server yet.")
    source_path = await save_upload(source, {".jpg", ".jpeg", ".png", ".webp"}, UPLOADS)
    target_path = await save_upload(target, {".mp4", ".mov", ".mkv", ".webm"}, UPLOADS)
    output_path = OUTPUTS / f"{uuid.uuid4().hex}.mp4"

    # Deep-Live-Cam supports headless source/target/output arguments.
    cmd = [
        os.getenv("PYTHON", "python"), str(ENGINE / "run.py"),
        "-s", str(source_path), "-t", str(target_path), "-o", str(output_path),
        "--keep-fps", "--keep-audio",
    ]
    provider = os.getenv("EXECUTION_PROVIDER")
    if provider:
        cmd += ["--execution-provider", provider]

    try:
        result = subprocess.run(cmd, cwd=ENGINE, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Processing timed out")
    if result.returncode != 0 or not output_path.exists():
        detail = (result.stderr or result.stdout or "Processing failed")[-2000:]
        raise HTTPException(500, detail)
    return FileResponse(output_path, media_type="video/mp4", filename="swap.mp4")
