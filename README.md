# Defect Detection with PatchCore + YOLO

An AI visual inspection app for electronics. It checks photos of a part and tells you:

1. **Is it defective?** (PatchCore)
2. **Where is the problem?** (heatmap)
3. **What kind of defect is it?** (YOLO)

**Live app:** https://defect-detection0.streamlit.app/

## How it works

```
Photo (5 camera views) -> PatchCore -> normal or defective?
                                           |
                                    if defective
                                           v
                                  YOLO -> defect type + box
                                           |
                                           v
                          Verdict + heatmap + CSV report
```

- **PatchCore** learns only from normal images. There is one model per camera (C1 to C5).
- **YOLO** only looks at the images PatchCore flagged.
- Products: **audiojack** and **pcb**.
- Data: [Real-IAD](https://github.com/Tencent/Real-IAD) dataset.

## Results

| | Audio jack | PCB |
|---|---|---|
| Detection (AUROC, 5 cameras) | 0.83 - 0.91 | 0.88 - 0.92 |
| Defect type (mAP50) | 0.91 | 0.87 |

## How to use the app

1. Pick a product in the sidebar.
2. **Check samples:** upload a `.zip` of your images, choose a sample ID (or all), and press check.
3. **Manual check:** choose the camera view (C1 to C5), take a photo or upload one, and press check. A hint shows how to take the photo for each view.
4. **Report:** see all results and download them as a CSV.

The first check of a product can take a few minutes, because the app downloads that product's models first.

### Image zip structure

```
images.zip
└── images/
    ├── C1/   0001__a.jpg, 0002__a.jpg ...
    ├── C2/
    ├── C3/
    ├── C4/
    └── C5/
```

- Folders **C1 to C5** are the camera views (C1 = top view, C2 to C5 = angled views).
- File name = `<sample id>__<anything>.jpg`. Photos of the same part share the same sample ID.
- Allowed types: `.jpg`, `.jpeg`, `.png`.

## Files in this repo

| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python packages |
| `packages.txt` | System packages for Streamlit Cloud |
| `defect_detection_per_view0.ipynb` | Trains the PatchCore models (one per camera) |
| `defect_type_yolo.ipynb` | Trains the YOLO defect-type model |

## Models

The models are too big for the repo (about 1.1 GB per product), so they are stored as zip files in the **Releases** section of this repo (`pcb.zip`, `audiojack.zip`). Each zip contains `thresholds.json`, `yolo_type.pt` and `patchcore_C1.ckpt` to `patchcore_C5.ckpt`.

The app downloads a product's zip the first time it is used. The product list comes from the `MODEL_ZIP_URLS` secret: the zip links, separated by commas. The product name is the zip file name.

## Limitations

- About 3 in 10 audio jack defects and 2 in 10 PCB defects are not flagged. Use it as a second pair of eyes, and let a person confirm.
- The pass/fail lines were tuned on the same images used for the results, so the scores are a bit optimistic.
- YOLO knows only 4 defect types per product, and subtle ones (for example PCB contamination) are weaker.
- It was tested only on the Real-IAD images. Other cameras, lighting or products need testing.

## Author

Ali Yusuf Al Basri
