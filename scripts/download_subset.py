"""
Downloads a small, standard subset of COCO val2017 for evaluation.

Why val2017 (not train2017 / not coco128):
YOLOv8's official pretrained weights were trained on COCO train2017.
Evaluating on val2017 images means the model has NOT seen these images
during training, so the resulting mAP/precision/recall numbers are a
meaningful accuracy measurement rather than inflated training-set scores.

What this script does:
1. Downloads the official COCO 2017 annotations zip (contains ground-truth
   boxes + category labels for all 80 classes) and extracts instances_val2017.json.
2. Picks the first N image ids from val2017 that actually have at least one
   annotated object.
3. Downloads only those N images (not the full 5000-image / ~1GB val2017 set).

Everything downloaded here is the untouched, official COCO dataset --
no images or annotations are created or modified.
"""

import json
import zipfile
from pathlib import Path

import requests
from pycocotools.coco import COCO

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ANN_DIR = DATA_DIR / "annotations"
IMG_DIR = DATA_DIR / "val2017_subset"
N_IMAGES = 300

ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{file_name}"


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"  already have {dest.name}, skipping download")
        return
    print(f"  downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def ensure_annotations():
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    instances_path = ANN_DIR / "instances_val2017.json"
    if instances_path.exists():
        print("Annotations already present.")
        return instances_path

    zip_path = DATA_DIR / "annotations_trainval2017.zip"
    print("Step 1: downloading official COCO annotations (~241MB, one-time)...")
    download_file(ANNOTATIONS_ZIP_URL, zip_path)

    print("Extracting instances_val2017.json ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extract("annotations/instances_val2017.json", DATA_DIR)

    zip_path.unlink()  # free disk space, we only needed one file from it
    return instances_path


def download_image_subset(instances_path: Path):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(instances_path))

    # Only keep images that actually have annotated objects, sorted for reproducibility
    img_ids_with_anns = sorted({ann["image_id"] for ann in coco.anns.values()})
    subset_ids = img_ids_with_anns[:N_IMAGES]

    print(f"Step 2: downloading {len(subset_ids)} val2017 images...")
    for i, img_id in enumerate(subset_ids, 1):
        info = coco.loadImgs(img_id)[0]
        dest = IMG_DIR / info["file_name"]
        download_file(IMAGE_URL_TEMPLATE.format(file_name=info["file_name"]), dest)
        if i % 50 == 0:
            print(f"  {i}/{len(subset_ids)} images done")

    # Save the list of chosen image ids so evaluate.py restricts COCOeval to exactly these
    with open(DATA_DIR / "subset_image_ids.json", "w") as f:
        json.dump(subset_ids, f)

    print(f"Done. {len(subset_ids)} images saved to {IMG_DIR}")


if __name__ == "__main__":
    instances_path = ensure_annotations()
    download_image_subset(instances_path)
