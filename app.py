import base64
import io
import os
import shutil
import tempfile
import uuid
import zipfile

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Defect detection", page_icon="🔍", layout="wide")

VIEWS = ["C1", "C2", "C3", "C4", "C5"]
IMAGE_EXT = (".jpg", ".jpeg", ".png")
BATCH = 6                    # images sent to the server per request
MAX_ZIP_MB = 300
MAX_IMAGES = 2000


def get_setting(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


API_URL = get_setting("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = get_setting("API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


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
    "Use the same background, distance and angle as the training photos.",
]


def view_map(view):
    """Small top-down map. The selected view is shown as [Cx]."""
    def tag(v):
        return f"[{v}]" if v == view else f" {v} "

    return "\n".join([
        f"{tag('C5')}{' ' * 14}{tag('C4')}",
        "  ┌" + "─" * 16 + "┐",
        "  │" + f"{tag('C1')} from above".center(16) + "│",
        "  │" + "PRODUCT".center(16) + "│",
        "  └" + "─" * 16 + "┘",
        f"{tag('C2')}{' ' * 14}{tag('C3')}",
        "(front)".center(22),
    ])


def show_hint(view):
    hint = VIEW_HINTS[view]
    left, right = st.columns([3, 2])
    with left:
        st.info(f"### {hint['icon']} {hint['title']}\n\n{hint['how']}")
        st.markdown("\n".join(f"- {tip}" for tip in hint["tips"] + COMMON_TIPS))
    with right:
        st.caption("Where to stand (looking down at the product):")
        st.code(view_map(view), language="text")


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


def template_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for view in VIEWS:
            archive.writestr(f"images/{view}/", "")
        archive.writestr(
            "images/README.txt",
            "Put the photos of each camera view inside its folder (C1 to C5).\n"
            "File name = <sample id>__<anything>.jpg   e.g. 0001__a.jpg\n"
            "Photos of the same item share the same sample id.\n",
        )
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Helpers
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


def error_detail(response):
    try:
        return response.json().get("detail", response.text[:200])
    except Exception:
        return response.text[:200]


@st.cache_data(ttl=60, show_spinner="Connecting to the server (it may be waking up)...")
def get_products():
    response = requests.get(f"{API_URL}/products", headers=HEADERS, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code} - {error_detail(response)}")
    return response.json()["products"]


def check_view(product, view, files):
    """files contains (sample_id, image_path) pairs. Calls the model server."""
    items = []
    for start in range(0, len(files), BATCH):
        chunk = files[start:start + BATCH]
        upload = []
        for _, path in chunk:
            with open(path, "rb") as file:
                upload.append(("files", (os.path.basename(path), file.read())))

        response = requests.post(
            f"{API_URL}/predict",
            headers=HEADERS,
            data={"product": product, "view": view},
            files=upload,
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(f"{view}: server error {response.status_code} - {error_detail(response)}")

        results = sorted(response.json()["results"], key=lambda r: r["index"])
        if len(results) != len(chunk):
            raise RuntimeError(
                f"{view}: server returned {len(results)} predictions for {len(chunk)} images. "
                "No partial result was saved."
            )

        for (sample_id, path), result in zip(chunk, results):
            items.append({
                "sample_id": sample_id,
                "view": view,
                "path": path,
                "score": result["score"],
                "threshold": result["threshold"],
                "defective": result["defective"],
                "types": result["types"],
                "picture": base64.b64decode(result["picture"]) if result["picture"] else None,
            })
    return items


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
try:
    products = get_products()
except Exception as error:
    st.error(
        "Cannot reach the detection server. If it was asleep, wait a minute and refresh.\n\n"
        + str(error)
    )
    st.stop()

if not products:
    st.error("The server has no product models yet.")
    st.stop()

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
        st.download_button(
            "Download an empty folder template (.zip)",
            template_zip(),
            "images_template.zip",
            "application/zip",
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
                with st.spinner("Checking... the first request can take a minute if the server was asleep."):
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
                with st.spinner("Checking... the first request can take a minute if the server was asleep."):
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
