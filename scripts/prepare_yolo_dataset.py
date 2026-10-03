"""
Builds a YOLO-format train/val dataset from COCO val2017 (official images + official annotations).

Why this dataset: COCO is the standard object-detection benchmark (80 everyday classes, annotated by the
COCO team; we create no annotations). We use 600 val2017 images (the first 600 ids that have at least one
object, so the 300 already downloaded are reused) and split them 80/20 with a fixed seed:
    480 train images / 120 val images, no image appears in both.
Metrics are always computed on the 120 val images, which no model was trained on.
(val2017 is used rather than train2017 so the pretrained yolov8n.pt reference run has never seen these images.)

Output: data/yolo_subset/{images,labels}/{train,val}/ and data/yolo_subset/coco_subset.yaml
"""

import json
import random
import shutil
from pathlib import Path

import requests
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC_IMG = DATA / "val2017_subset"
OUT = DATA / "yolo_subset"
N_IMAGES, VAL_FRACTION, SEED = 600, 0.2, 0

coco = COCO(str(DATA / "annotations" / "instances_val2017.json"))
cat_ids = sorted(coco.getCatIds())
cat_to_yolo = {c: i for i, c in enumerate(cat_ids)}  # COCO ids 1-90 (with gaps) -> YOLO ids 0-79
names = {cat_to_yolo[c["id"]]: c["name"] for c in coco.loadCats(cat_ids)}

ids = sorted({a["image_id"] for a in coco.anns.values()})[:N_IMAGES]
rng = random.Random(SEED)
shuffled = ids[:]
rng.shuffle(shuffled)
n_val = int(len(ids) * VAL_FRACTION)
split = {i: "val" for i in shuffled[:n_val]} | {i: "train" for i in shuffled[n_val:]}

SRC_IMG.mkdir(parents=True, exist_ok=True)
for s in ("train", "val"):
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

counts = {"train": [0, 0], "val": [0, 0]}  # [images, boxes]
for img_id in ids:
    info = coco.loadImgs(img_id)[0]
    src = SRC_IMG / info["file_name"]
    if not src.exists():
        r = requests.get(f"http://images.cocodataset.org/val2017/{info['file_name']}", timeout=60)
        r.raise_for_status()
        src.write_bytes(r.content)
    s = split[img_id]
    shutil.copy(src, OUT / "images" / s / info["file_name"])
    W, H = info["width"], info["height"]
    lines = []
    for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False)):
        x, y, w, h = a["bbox"]  # COCO: top-left corner + size, in pixels
        if w <= 1 or h <= 1:
            continue
        # YOLO: class, centre x, centre y, width, height, all divided by image size
        lines.append(f"{cat_to_yolo[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    (OUT / "labels" / s / (Path(info["file_name"]).stem + ".txt")).write_text("\n".join(lines))
    counts[s][0] += 1
    counts[s][1] += len(lines)

yaml = f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: 80\nnames:\n" + "".join(
    f"  {i}: {n}\n" for i, n in sorted(names.items())
)
(OUT / "coco_subset.yaml").write_text(yaml)
(OUT / "split.json").write_text(json.dumps({"seed": SEED, "train": sorted(k for k, v in split.items() if v == "train"),
                                              "val": sorted(k for k, v in split.items() if v == "val")}))
print("Dataset: COCO val2017 subset | classes: 80")
for s, (n, b) in counts.items():
    print(f"  {s}: {n} images, {b} boxes")
