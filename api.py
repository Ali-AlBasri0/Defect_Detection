"""
Model server (FastAPI). Deploy this on Render.

Models are NOT stored in the GitHub code repo. At start-up they are
downloaded from either:
  - zip files, one per product, e.g. a GitHub Release (env MODEL_ZIP_URLS,
    comma separated; the zip name is the product name: "PRODUCT.zip"), or
  - a private Hugging Face model repo (env HF_REPO).
They can also already be on disk in MODELS_DIR (e.g. a Render persistent disk).

Expected layout inside MODELS_DIR:

    <product>/thresholds.json
    <product>/yolo_type.pt
    <product>/patchcore_C1.ckpt ... patchcore_C5.ckpt

Environment variables:
    MODEL_ZIP_URLS  https://.../PRODUCT1.zip,https://.../PRODUCT2.zip
    HF_REPO     e.g. "your-name/defect-models"   (optional alternative)
    HF_TOKEN    Hugging Face read token           (needed for a private repo)
    MODELS_DIR  default "models"
    API_KEY     shared secret; the Streamlit app sends it in X-API-Key
"""
import base64
import io
import json
import os
import secrets
import shutil
import tempfile
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from urllib.parse import unquote, urlparse

import matplotlib
import requests

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Patchcore  # noqa: E402
from ultralytics import YOLO  # noqa: E402

VIEWS = ["C1", "C2", "C3", "C4", "C5"]
IMAGE_EXT = (".jpg", ".jpeg", ".png")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
API_KEY = os.environ.get("API_KEY", "")
MAX_FILES = 16
MAX_FILE_MB = 15

_lock = threading.Lock()  # one prediction at a time (keeps RAM low)
_yolo_cache = {}
_threshold_cache = {}


def download_model_zip(url):
    """Download PRODUCT.zip and unpack its model files into MODELS_DIR/PRODUCT/."""
    product = Path(unquote(urlparse(url).path)).stem
    target = MODELS_DIR / product
    wanted = required_files()
    if all((target / name).is_file() for name in wanted):
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = MODELS_DIR / f"{product}.zip"
    try:
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    file.write(chunk)
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                name = Path(info.filename).name  # base name only: no path tricks
                if info.is_dir() or name not in wanted:
                    continue
                with archive.open(info) as source, open(target / name, "wb") as out:
                    shutil.copyfileobj(source, out)
    finally:
        zip_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app):
    for url in os.environ.get("MODEL_ZIP_URLS", "").split(","):
        if url.strip():
            download_model_zip(url.strip())
    repo = os.environ.get("HF_REPO")
    if repo:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo,
            local_dir=str(MODELS_DIR),
            token=os.environ.get("HF_TOKEN") or None,
        )
    yield


app = FastAPI(title="Defect detection model server", lifespan=lifespan)


def check_key(x_api_key: str = Header(default="")):
    if API_KEY and not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Wrong or missing API key")


def required_files():
    return ["thresholds.json", "yolo_type.pt"] + [f"patchcore_{v}.ckpt" for v in VIEWS]


def list_products():
    if not MODELS_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and all((p / name).is_file() for name in required_files())
    )


def get_thresholds(product):
    if product not in _threshold_cache:
        with open(MODELS_DIR / product / "thresholds.json") as file:
            _threshold_cache[product] = json.load(file)
    return _threshold_cache[product]


def get_yolo(product):
    if product not in _yolo_cache:
        _yolo_cache[product] = YOLO(str(MODELS_DIR / product / "yolo_type.pt"))
    return _yolo_cache[product]


def make_picture(path, anomaly_map, found, title):
    """Input | anomaly heatmap | YOLO boxes, returned as a base64 PNG."""
    img = Image.open(path).convert("RGB")
    fig = Figure(figsize=(15, 5))
    axes = fig.subplots(1, 3)
    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[1].imshow(img.resize((anomaly_map.shape[1], anomaly_map.shape[0])))
    axes[1].imshow(anomaly_map, cmap="jet", alpha=0.5)
    axes[1].set_title("Anomaly heatmap")
    axes[2].imshow(found.plot()[..., ::-1])
    axes[2].set_title(title)
    for axis in axes:
        axis.axis("off")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=80)
    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/products", dependencies=[Depends(check_key)])
def products():
    return {"products": list_products()}


@app.post("/predict", dependencies=[Depends(check_key)])
def predict(
    product: str = Form(...),
    view: str = Form(...),
    files: List[UploadFile] = File(...),
):
    view = view.upper()
    if view not in VIEWS:
        raise HTTPException(400, f"view must be one of {VIEWS}")
    if product not in list_products():  # also blocks path tricks like ../
        raise HTTPException(404, f"Unknown product: {product}")
    if not files or len(files) > MAX_FILES:
        raise HTTPException(400, f"Send between 1 and {MAX_FILES} images per request")

    thresholds = get_thresholds(product)
    threshold = float(thresholds[view])
    ckpt = MODELS_DIR / product / f"patchcore_{view}.ckpt"

    work = Path(tempfile.mkdtemp(prefix="pred_"))
    image_dir = work / "img"
    image_dir.mkdir()
    try:
        index_of = {}
        for number, upload in enumerate(files):
            extension = Path(upload.filename or "").suffix.lower()
            if extension not in IMAGE_EXT:
                raise HTTPException(400, f"Unsupported file type: {upload.filename}")
            data = upload.file.read()
            if len(data) > MAX_FILE_MB * 1024 * 1024:
                raise HTTPException(413, f"{upload.filename} is larger than {MAX_FILE_MB} MB")
            temp_name = f"image_{number:06d}{extension}"
            (image_dir / temp_name).write_bytes(data)
            index_of[temp_name] = number

        with _lock:
            model = Patchcore()
            engine = Engine(
                accelerator="auto",
                devices=1,
                enable_progress_bar=False,
                default_root_dir=str(work / "out"),
            )
            predictions = engine.predict(
                model=model,
                data_path=str(image_dir),
                ckpt_path=str(ckpt),
            )

            yolo = get_yolo(product)
            results = {}
            for batch in predictions:
                for i in range(len(batch.image_path)):
                    temp_name = os.path.basename(str(batch.image_path[i]))
                    if temp_name not in index_of or temp_name in results:
                        continue

                    path = str(image_dir / temp_name)
                    score = float(batch.pred_score[i])
                    anomaly_map = batch.anomaly_map[i].squeeze().cpu().numpy()
                    defective = score >= threshold

                    types = ""
                    picture = ""
                    if defective:
                        found = yolo(path, verbose=False)
                        names = [
                            f"{yolo.names[int(box.cls)]} ({float(box.conf):.2f})"
                            for box in found[0].boxes
                        ]
                        types = ", ".join(names) if names else "type not found"
                        picture = make_picture(path, anomaly_map, found[0], types)

                    results[temp_name] = {
                        "index": index_of[temp_name],
                        "score": score,
                        "threshold": threshold,
                        "defective": defective,
                        "types": types,
                        "picture": picture,
                    }

        if len(results) != len(files):
            raise HTTPException(
                500,
                f"{view}: PatchCore returned {len(results)} predictions for {len(files)} images.",
            )
        ordered = sorted(results.values(), key=lambda r: r["index"])
        return {"results": ordered}
    finally:
        shutil.rmtree(work, ignore_errors=True)
