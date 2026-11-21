"""
Defect detection - all-in-one Streamlit app (no separate API server).

The model files are NOT in the GitHub code repo. For every product the app
downloads "PRODUCT.zip" (thresholds.json, yolo_type.pt, patchcore_C1..C5.ckpt)
from a link you put in the Streamlit secret MODEL_ZIP_URLS, the first time
that product is used.

Secret example (Streamlit Cloud -> Settings -> Secrets):
    MODEL_ZIP_URLS = "https://github.com/Ali-AlBasri0/Defect_Detection/releases/download/models/pcb.zip,https://github.com/Ali-AlBasri0/Defect_Detection/releases/download/models/audiojack.zip"
"""
import gc
import io
import json
import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Defect detection", page_icon="🔍", layout="wide")

VIEWS = ["C1", "C2", "C3", "C4", "C5"]
IMAGE_EXT = (".jpg", ".jpeg", ".png")
REQUIRED = ["thresholds.json", "yolo_type.pt"] + [f"patchcore_{v}.ckpt" for v in VIEWS]
BATCH = 4                    # images processed together (small = less memory)
MAX_ZIP_MB = 300
MAX_IMAGES = 2000
MODEL_ROOT = Path(tempfile.gettempdir()) / "defect_models"


def get_setting(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


def parse_model_zips():
    raw = get_setting("MODEL_ZIP_URLS")
    if isinstance(raw, (list, tuple)):
        urls = [str(u).strip() for u in raw]
    else:
        urls = [u.strip() for u in str(raw).replace("\n", ",").split(",")]
    return {Path(unquote(urlparse(u).path)).stem: u for u in urls if u}


MODEL_ZIPS = parse_model_zips()


# --------------------------------------------------------------------------
# Hints shown to the user
# --------------------------------------------------------------------------
VIEW_HINTS = {
    "C1": {
        "icon": "⬇️",
        "title": "C1 · Top view",
        "how": "Hold the camera **directly above** the product, pointing **straight down**.",
        "tips": [
            "Keep the phone flat, parallel to the table.",
            "Put the product in the middle of the frame.",
        ],
    },
    "C2": {
        "icon": "↗️",
        "title": "C2 · Angled view 1 (front-left)",
        "how": "Stand at the **front-left corner** and tilt the camera down at about **45°**.",
        "tips": ["Aim at the centre of the product.", "Keep the same distance as in C1."],
    },
    "C3": {
        "icon": "↖️",
        "title": "C3 · Angled view 2 (front-right)",
        "how": "Move to the **front-right corner** and tilt the camera down at about **45°**.",
        "tips": ["Aim at the centre of the product.", "Keep the same distance as in C1."],
    },
    "C4": {
        "icon": "↙️",
        "title": "C4 · Angled view 3 (back-right)",
        "how": "Move to the **back-right corner** and tilt the camera down at about **45°**.",
        "tips": ["Aim at the centre of the product.", "Keep the same distance as in C1."],
    },
    "C5": {
        "icon": "↘️",
        "title": "C5 · Angled view 4 (back-left)",
        "how": "Move to the **back-left corner** and tilt the camera down at about **45°**.",
        "tips": ["Aim at the centre of the product.", "Keep the same distance as in C1."],
    },
}

COMMON_TIPS = [
    "Use bright, even light and avoid strong shadows or reflections.",
    "Show the whole product and keep the photo sharp (hold still).",
    "Use a white background and take the photo from a close distance.",
]


def show_hint(view):
    hint = VIEW_HINTS[view]
    st.info(f"### {hint['icon']} {hint['title']}\n\n{hint['how']}")
    st.markdown("\n".join(f"- {tip}" for tip in hint["tips"] + COMMON_TIPS))


FOLDER_STRUCTURE = """\
images.zip
└── images/                 <- any name, optional
    ├── C1/                 <- top view
    │   ├── 0001__a.jpg
    │   └── 0002__a.jpg
    ├── C2/                 <- angled view 1
    │   ├── 0001__a.jpg
    │   └── 0002__a.jpg
    ├── C3/                 <- angled view 2
    ├── C4/                 <- angled view 3
    └── C5/                 <- angled view 4
"""


# --------------------------------------------------------------------------
# Image zip helpers
# --------------------------------------------------------------------------
def get_id(file_name):
    return file_name.split("__")[0]


def extract_zip(uploaded, dest):
    """Unpack the C1..C5 folders of a zip into dest/<view>/. Returns image count."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)

    count = 0
    with zipfile.ZipFile(uploaded) as archive:
        if sum(info.file_size for info in archive.infolist()) > MAX_ZIP_MB * 1024 * 1024:
            raise ValueError(f"The zip is larger than {MAX_ZIP_MB} MB unpacked.")
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = [p for p in info.filename.replace("\\", "/").split("/") if p]
            if not parts or "__MACOSX" in parts:
                continue
            name = parts[-1]
            if name.startswith(".") or os.path.splitext(name)[1].lower() not in IMAGE_EXT:
                continue
            view = next((p.upper() for p in parts[:-1] if p.upper() in VIEWS), None)
            if view is None:
                continue
            if count >= MAX_IMAGES:
                raise ValueError(f"Too many images (limit {MAX_IMAGES}).")

            folder = os.path.join(dest, view)
            os.makedirs(folder, exist_ok=True)
            # only the base name is used, so a zip can never write outside dest
            with open(os.path.join(folder, name), "wb") as file:
                file.write(archive.read(info))
            count += 1
    return count


def scan_samples(root):
    samples = {}
    for view in VIEWS:
        folder = os.path.join(root, view)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            samples.setdefault(get_id(name), {}).setdefault(view, []).append(
                os.path.join(folder, name)
            )
    return samples


# --------------------------------------------------------------------------
# Models: download once, run on this server
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading the models... the first time takes a few minutes.")
def ensure_product(product, url):
    """Download PRODUCT.zip and unpack the model files. Returns the model folder."""
    target = MODEL_ROOT / product
    if all((target / name).is_file() for name in REQUIRED):
        return str(target)

    target.mkdir(parents=True, exist_ok=True)
    zip_path = MODEL_ROOT / f"{product}.zip"
    try:
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    file.write(chunk)
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                name = Path(info.filename).name  # base name only: no path tricks
                if info.is_dir() or name not in REQUIRED:
                    continue
                with archive.open(info) as source, open(target / name, "wb") as out:
                    shutil.copyfileobj(source, out)
    finally:
        zip_path.unlink(missing_ok=True)

    missing = [name for name in REQUIRED if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"{product}.zip is missing: {', '.join(missing)}")
    return str(target)


@st.cache_resource
def get_yolo(path):
    from ultralytics import YOLO

    return YOLO(path)


@st.cache_resource
def get_lock():
    return threading.Lock()  # one prediction at a time keeps memory low


def make_picture(path, anomaly_map, found, title):
    """Input | anomaly heatmap | YOLO boxes, returned as PNG bytes."""
    from matplotlib.figure import Figure
    from PIL import Image

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
    return buffer.getvalue()


def run_chunk(model_dir, view, threshold, chunk):
    """chunk = list of (sample_id, image_path). Returns one item per image."""
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    work = Path(tempfile.mkdtemp(prefix="pred_"))
    image_dir = work / "img"
    image_dir.mkdir()
    try:
        index_of = {}
        for number, (_, path) in enumerate(chunk):
            temp_name = f"image_{number:06d}{Path(path).suffix.lower()}"
            shutil.copy2(path, image_dir / temp_name)
            index_of[temp_name] = number

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
            ckpt_path=os.path.join(model_dir, f"patchcore_{view}.ckpt"),
        )

        yolo = get_yolo(os.path.join(model_dir, "yolo_type.pt"))
        done = {}
        for batch in predictions:
            for i in range(len(batch.image_path)):
                temp_name = os.path.basename(str(batch.image_path[i]))
                if temp_name not in index_of or temp_name in done:
                    continue

                number = index_of[temp_name]
                sample_id, path = chunk[number]
                score = float(batch.pred_score[i])
                anomaly_map = batch.anomaly_map[i].squeeze().cpu().numpy()
                defective = score >= threshold

                types = ""
                picture = None
                if defective:
                    found = yolo(path, verbose=False)
                    names = [
                        f"{yolo.names[int(box.cls)]} ({float(box.conf):.2f})"
                        for box in found[0].boxes
                    ]
                    types = ", ".join(names) if names else "type not found"
                    picture = make_picture(path, anomaly_map, found[0], types)

                done[temp_name] = {
                    "sample_id": sample_id,
                    "view": view,
                    "path": path,
                    "score": score,
                    "threshold": threshold,
                    "defective": defective,
                    "types": types,
                    "picture": picture,
                }

        if len(done) != len(chunk):
            raise RuntimeError(
                f"{view}: PatchCore returned {len(done)} predictions for {len(chunk)} images. "
                "No partial result was saved."
            )
        return [done[name] for name, _ in sorted(index_of.items(), key=lambda kv: kv[1])]
    finally:
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()


def check_view(product, view, files):
    """files contains (sample_id, image_path) pairs."""
    model_dir = ensure_product(product, MODEL_ZIPS[product])
    with open(os.path.join(model_dir, "thresholds.json")) as file:
        threshold = float(json.load(file)[view])

    items = []
    for start in range(0, len(files), BATCH):
        with get_lock():
            items.extend(run_chunk(model_dir, view, threshold, files[start:start + BATCH]))
    return items


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
if not MODEL_ZIPS:
    st.error(
        "No models configured. Add the secret MODEL_ZIP_URLS "
        "(links to PRODUCT.zip files, separated by commas)."
    )
    st.stop()

products = sorted(MODEL_ZIPS)

if "workdir" not in st.session_state:
    st.session_state["workdir"] = tempfile.mkdtemp(prefix="defect_")
workdir = st.session_state["workdir"]

product = st.sidebar.selectbox("Product", products)
page = st.sidebar.radio("Page", ["Check samples", "Manual check", "Report"])

st.title("Defect detection - " + product)

images_root = os.path.join(workdir, product, "images")
samples = scan_samples(images_root)

# New key starts with clean results after the image-count fix.
key = "results_v2_" + product
if key not in st.session_state:
    st.session_state[key] = {}
results = st.session_state[key]

chosen = None
WAIT_TEXT = "Checking... the first check can take a few minutes while the models download."


# --------------------------------------------------------------------------
# Check samples
# --------------------------------------------------------------------------
if page == "Check samples":
    st.subheader("1. Select your image folder")

    with st.expander("📁 How must my folder look?", expanded=not samples):
        st.markdown(
            "Put your photos in one folder with a sub-folder for each camera view "
            "(**C1** to **C5**), then compress it into a **.zip** file and upload it here."
        )
        st.code(FOLDER_STRUCTURE, language="text")
        st.markdown(
            "- **C1** top view · **C2–C5** angled views 1–4\n"
            "- File name = `<sample id>__<anything>.jpg`. The text before `__` is the sample ID, "
            "so photos of the same item share it (`0001__a.jpg` in C1 and in C2 belong to sample 0001).\n"
            "- Allowed: `.jpg`, `.jpeg`, `.png`\n"
            "- **How to zip:** Windows – right-click the folder → *Send to* → *Compressed (zipped) folder*. "
            "Mac – right-click → *Compress*."
        )

    zip_file = st.file_uploader("Select your image folder (.zip)", type=["zip"])
    if zip_file is not None:
        signature = (product, zip_file.name, zip_file.size)
        if st.session_state.get("zip_signature") != signature:
            try:
                with st.spinner("Unpacking..."):
                    count = extract_zip(zip_file, images_root)
            except Exception as error:
                st.error(f"Could not read the zip: {error}")
                st.stop()
            st.session_state["zip_signature"] = signature
            results.clear()
            samples = scan_samples(images_root)
            if count == 0:
                st.warning(
                    "No images found. Check that the zip contains folders named C1 to C5 "
                    "with .jpg/.png images inside."
                )

    if not samples:
        st.info("Upload a zip to start.")
    else:
        st.subheader("2. Check")
        counts = ", ".join(
            f"{view}: {sum(len(s.get(view, [])) for s in samples.values())}" for view in VIEWS
        )
        st.write(f"{len(samples)} samples found ({counts} images), {len(results)} already checked")
        chosen = st.selectbox("Sample ID", sorted(samples))
        check_one = st.button("Check this sample")
        check_all = st.button("Check all samples")

        ids_to_check = [chosen] if check_one else list(samples) if check_all else []

        if ids_to_check:
            for sample_id in ids_to_check:
                results[sample_id] = []

            progress = st.progress(0.0)
            try:
                with st.spinner(WAIT_TEXT):
                    for number, view in enumerate(VIEWS, start=1):
                        files = [
                            (sample_id, path)
                            for sample_id in ids_to_check
                            for path in samples[sample_id].get(view, [])
                        ]
                        if files:
                            for item in check_view(product, view, files):
                                results[item["sample_id"]].append(item)
                        progress.progress(number / len(VIEWS))
            except Exception as error:
                for sample_id in ids_to_check:
                    results.pop(sample_id, None)
                st.error(str(error))
                st.stop()


# --------------------------------------------------------------------------
# Manual check
# --------------------------------------------------------------------------
if page == "Manual check":
    st.write("Choose the camera view, add an image (take a photo or upload one) and press the button.")

    manual_view = st.selectbox("Camera view", VIEWS)
    show_hint(manual_view)

    source = st.radio("Image source", ["📷 Take a photo", "⬆️ Upload an image"], horizontal=True)

    image_bytes = None
    image_name = None
    if source.startswith("📷"):
        st.caption("Allow camera access when your browser asks. On a phone, tap the switch-camera icon to use the back camera.")
        photo = st.camera_input(f"Take the {manual_view} photo", key=f"camera_{manual_view}")
        if photo is not None:
            image_bytes, image_name = photo.getvalue(), "camera_photo.jpg"
    else:
        uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image_bytes, image_name = uploaded.getvalue(), uploaded.name

    if st.button("Check this image"):
        if image_bytes is None:
            st.warning("Take or upload an image first.")
        else:
            folder = os.path.join(workdir, product, "manual")
            os.makedirs(folder, exist_ok=True)
            extension = os.path.splitext(image_name)[1].lower() or ".jpg"
            path = os.path.join(folder, f"{uuid.uuid4().hex}{extension}")
            with open(path, "wb") as file:
                file.write(image_bytes)

            try:
                with st.spinner(WAIT_TEXT):
                    item = check_view(product, manual_view, [("manual", path)])[0]
            except Exception as error:
                st.error(str(error))
                st.stop()

            if item["defective"]:
                st.error(
                    f"{item['view']} - DEFECTIVE - score: {item['score']:.3f} "
                    f"(threshold {item['threshold']:.3f}) - {item['types']}"
                )
                st.image(item["picture"])
            else:
                st.success(
                    f"{item['view']} - NORMAL - score: {item['score']:.3f} "
                    f"(threshold {item['threshold']:.3f})"
                )
                st.image(item["path"], width=250)


# --------------------------------------------------------------------------
# Results / report
# --------------------------------------------------------------------------
rows = []
defective_ids = []
for sample_id in sorted(results):
    for item in results[sample_id]:
        result = "DEFECTIVE" if item["defective"] else "NORMAL"
        if item["defective"] and sample_id not in defective_ids:
            defective_ids.append(sample_id)

        rows.append({
            "Product": product,
            "Sample ID": sample_id,
            "View": item["view"],
            "Image": os.path.basename(item["path"]),
            "Score": round(item["score"], 3),
            "Threshold": round(item["threshold"], 3),
            "Result": result,
            "Defect type": item["types"],
        })

report = pd.DataFrame(rows)


if page == "Check samples" and chosen in results:
    items = results[chosen]
    bad = sum(item["defective"] for item in items)

    st.header("Sample " + chosen)
    if bad:
        st.error(f"DEFECTIVE in {bad} of {len(items)} images")
    else:
        st.success(f"NORMAL in all {len(items)} images")

    for item in items:
        if item["defective"]:
            st.error(
                f"{item['view']} - DEFECTIVE - score: {item['score']:.3f} - {item['types']}"
            )
            st.image(item["picture"])
        else:
            st.success(f"{item['view']} - NORMAL - score: {item['score']:.3f}")
            st.image(item["path"], width=250)


if page == "Report":
    if not results:
        st.write("Nothing checked yet. Go to 'Check samples' and check a sample.")
    else:
        st.write(f"{len(results)} samples checked, {len(defective_ids)} defective")
        st.download_button(
            "Download report (CSV)",
            report.to_csv(index=False),
            f"defect_report_{product}.csv",
            "text/csv",
        )
        st.dataframe(report)

        st.subheader("Defective samples")
        for sample_id in defective_ids:
            with st.expander("Sample " + sample_id):
                for item in results[sample_id]:
                    if item["defective"]:
                        st.error(
                            f"{item['view']} - score: {item['score']:.3f} - {item['types']}"
                        )
                        st.image(item["picture"])
