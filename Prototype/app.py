import os, json
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from anomalib.models import Patchcore
from anomalib.engine import Engine

category = "audiojack"   # the same as in the notebook
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models_" + category)

f = open(os.path.join(save_dir, "thresholds.json"))
thresholds = json.load(f)
f.close()

st.title("Defect detection - " + category)

view = st.selectbox("View (camera)", ["C1", "C2", "C3", "C4", "C5"])
upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if upload is not None:
    img = Image.open(upload).convert("RGB")
    img.save("temp.png")   # PatchCore reads the image from a file

    # load the saved PatchCore of this view and run it
    model = Patchcore()
    engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)
    result = engine.predict(model=model, data_path="temp.png",
                            ckpt_path=os.path.join(save_dir, "patchcore_" + view + ".ckpt"))
    score = float(result[0].pred_score)
    amap = result[0].anomaly_map.squeeze().cpu().numpy()

    # normal or defective
    if score >= thresholds[view]:
        st.error("DEFECTIVE - score: " + str(round(score, 3)))
    else:
        st.success("NORMAL - score: " + str(round(score, 3)))

    # where: the heatmap on top of the image (red = defect)
    img = img.resize((amap.shape[1], amap.shape[0]))
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(amap, cmap="jet", alpha=0.5)
    plt.title("Anomaly heatmap")
    plt.axis("off")
    st.pyplot(fig)
