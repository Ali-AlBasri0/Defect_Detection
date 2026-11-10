# Visual Defect Detection with PatchCore (Real-IAD)

## Problem Statement

Factories need to check every product for visual defects (scratches, damage, missing or deformed parts). Doing this by eye is slow and depends on the person doing the checking. Defective images are also rare and come in many different types, so it is hard to collect enough labelled examples of every defect to train a normal classifier.

This project builds a system that, given a photo of a product, answers two questions:

1. Is the product **normal or defective**?
2. **Where** is the defect? (shown as a heatmap on the image)

## Impact

- **Consistency:** the same model checks every image with the same rule (a threshold), instead of relying on a person's judgement.
- **Speed:** a model gives a score for an image automatically.
- **No defect labels needed to train:** PatchCore learns only from normal images, so it can find defect types it has never seen before.
- **Explainable result:** the heatmap shows the inspector where to look, so the result can be checked quickly.

## Scope

**Dataset:** [Real-IAD](https://huggingface.co/datasets/Real-IAD/Real-IAD) (`realiad_512`). Each product is photographed from 5 camera views (C1 to C5).

The project uses the `realiad_512` version and focuses on the `audiojack` category.

**Approach:** One PatchCore model (Anomalib) is trained for each view, using only normal images. Each view gets its own threshold, chosen from the ROC curve. The notebook is run for one product category at a time (the `category` variable). This prototype was run on the **audiojack** category.

**Audience:** quality-inspection use cases in manufacturing, and the course reviewers of this capstone.

**Deliverables at this stage:**

- `defect_detection_per_view0.ipynb`: data preparation, training of the 5 models, evaluation, and a detection example with heatmap.
- `app.py`: a Streamlit web app. The user chooses a view, uploads an image, and gets NORMAL/DEFECTIVE with the score and the heatmap.
- This README.

**Not included in this repository:** the trained model files (`.ckpt`, about 255 MB each) and the dataset, because they are too big for GitHub.

## Data Preparation (audiojack)

- The Real-IAD JSON file lists the train images (normal only) and the test images.
- In the JSON: 1235 train images and 3815 test images.
- Images are sorted into folders for each view: `train_good`, `test_good`, `test_bad` (test label `OK` = normal, anything else = defective).
- Images per view (train / test normal / test defective):

| View | Train | Test normal | Test defective |
| ---- | ----- | ----------- | -------------- |
| C1   | 247   | 550         | 213            |
| C2   | 247   | 477         | 286            |
| C3   | 247   | 518         | 245            |
| C4   | 247   | 489         | 274            |
| C5   | 247   | 485         | 278            |

## Early Results (audiojack)

Image AUROC for each view model:

| View | AUROC |
| ---- | ----- |
| C1   | 0.831 |
| C2   | 0.818 |
| C3   | 0.902 |
| C4   | 0.848 |
| C5   | 0.843 |

Classification report of all views together with the per-view thresholds (class 1 = defective):

| Class         | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| 0 (normal)    | 0.86      | 0.91   | 0.88     | 1261    |
| 1 (defective) | 0.80      | 0.70   | 0.75     | 649     |

Overall accuracy: 0.84.
