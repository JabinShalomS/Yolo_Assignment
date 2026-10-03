"""
Runs the pretrained YOLOv8n model (trained on COCO) over our val2017 image
subset, and scores its predictions against the official COCO ground-truth
annotations using pycocotools -- the same evaluation tool used to produce
official published YOLO/COCO benchmark numbers.

Pipeline:
  pretrained weights (yolov8n.pt, trained on COCO train2017)
        -> model.predict() on each subset image
        -> convert boxes to COCO result format {image_id, category_id, bbox, score}
        -> COCOeval compares predictions to instances_val2017.json ground truth

Two inference passes, because the two kinds of metric need different thresholds:
  1. conf=0.001 -> mAP. mAP sweeps over every confidence level, so it needs
     (almost) all detections. This is the standard protocol Ultralytics uses for
     its published numbers, so our mAP is directly comparable to theirs.
  2. conf=0.25  -> precision / recall / F1. These describe one operating point:
     "if we only keep boxes with confidence >= 0.25, how many are right
     (precision) and how many real objects do we find (recall)?"
"""

import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "val2017_subset"
ANN_PATH = DATA_DIR / "annotations" / "instances_val2017.json"
SUBSET_IDS_PATH = DATA_DIR / "subset_image_ids.json"
OUT_DIR = ROOT / "outputs"

MODEL_NAME = "yolov8n.pt"  # nano: smallest/fastest pretrained YOLOv8 checkpoint, CPU-friendly
MAP_CONF = 0.001  # keep nearly all boxes for mAP (standard COCO/Ultralytics protocol)
PR_CONF = 0.25  # standard default confidence cutoff for reporting P/R/F1
PR_IOU = 0.5  # a prediction counts as correct if it overlaps a same-class GT box by IoU >= 0.5


def run_inference(model, coco: COCO, subset_ids, conf, save_samples=False):
    # Ultralytics class ids (0-79, contiguous) -> COCO category ids (1-90, with gaps)
    coco_cat_ids = sorted(coco.getCatIds())

    results_json = []
    for i, img_id in enumerate(subset_ids, 1):
        info = coco.loadImgs(img_id)[0]
        img_path = IMG_DIR / info["file_name"]

        preds = model.predict(str(img_path), conf=conf, verbose=False)[0]

        for box in preds.boxes:
            # Ultralytics gives box centre + size; COCO wants top-left corner + size
            x_c, y_c, w, h = box.xywh[0].tolist()
            x_min, y_min = x_c - w / 2, y_c - h / 2
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            results_json.append({
                "image_id": img_id,
                "category_id": coco_cat_ids[cls_id],
                "bbox": [x_min, y_min, w, h],
                "score": score,
            })

        if i % 50 == 0:
            print(f"  [conf={conf}] inference {i}/{len(subset_ids)} images done")

        # save a few annotated sample images for the report/demo
        if save_samples and i <= 6:
            preds.save(filename=str(OUT_DIR / f"sample_{info['file_name']}"))

    return results_json


def make_coco_eval(coco: COCO, results_json, subset_ids):
    coco_dt = coco.loadRes(results_json)
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = subset_ids  # restrict scoring to our subset only
    coco_eval.evaluate()
    coco_eval.accumulate()
    return coco_eval


def per_class_ap(coco: COCO, coco_eval: COCOeval):
    # coco_eval.eval["precision"] shape: [IoU thr, recall thr, class, area range, maxDet]
    precision = coco_eval.eval["precision"][:, :, :, 0, -1]  # area='all', maxDet=100
    rows = []
    for k, cat_id in enumerate(coco_eval.params.catIds):
        p = precision[:, :, k]
        if not (p > -1).any():
            continue  # class has no GT boxes in our subset -> no AP defined
        num_gt = len(coco.getAnnIds(imgIds=coco_eval.params.imgIds, catIds=[cat_id], iscrowd=False))
        rows.append({
            "class": coco.cats[cat_id]["name"],
            "num_gt": num_gt,
            "AP_50_95": float(p[p > -1].mean()),
            "AP_50": float(p[0][p[0] > -1].mean()),
        })
    return sorted(rows, key=lambda r: r["AP_50_95"], reverse=True)


def precision_recall_f1(coco_eval: COCOeval):
    """
    True precision/recall at one operating point (conf >= PR_CONF, IoU >= PR_IOU).
    COCOeval.evaluate() already did the per-class greedy matching of predictions
    to GT boxes; evalImgs holds, for every (image, class), which predictions got
    matched. We just count:
      TP = predictions matched to a GT box
      FP = predictions not matched to any GT box
      FN = GT boxes no prediction matched
    Crowd-region GT boxes and predictions landing on them are "ignored", as in COCO.
    """
    t = int(np.where(np.isclose(coco_eval.params.iouThrs, PR_IOU))[0][0])
    all_area = coco_eval.params.areaRng[0]

    tp = fp = num_gt = 0
    for e in coco_eval.evalImgs:
        if e is None or e["aRng"] != all_area:
            continue
        dt_ignore = np.asarray(e["dtIgnore"][t], dtype=bool)
        dt_matched = np.asarray(e["dtMatches"][t]) > 0
        tp += int((dt_matched & ~dt_ignore).sum())
        fp += int((~dt_matched & ~dt_ignore).sum())
        num_gt += int((~np.asarray(e["gtIgnore"], dtype=bool)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": num_gt - tp, "num_gt": num_gt,
            "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    coco = COCO(str(ANN_PATH))
    with open(SUBSET_IDS_PATH) as f:
        subset_ids = json.load(f)

    model = YOLO(MODEL_NAME)  # auto-downloads pretrained weights on first use

    # ---- Pass 1: mAP (conf=0.001) ----
    print(f"\nPass 1/2: inference at conf={MAP_CONF} for mAP")
    map_preds = run_inference(model, coco, subset_ids, MAP_CONF)
    map_eval = make_coco_eval(coco, map_preds, subset_ids)
    print("\n===== COCOeval summary (standard COCO metrics, conf=0.001) =====")
    map_eval.summarize()

    # ---- Pass 2: precision / recall / F1 (conf=0.25) ----
    print(f"\nPass 2/2: inference at conf={PR_CONF} for precision/recall/F1")
    pr_preds = run_inference(model, coco, subset_ids, PR_CONF, save_samples=True)
    with open(OUT_DIR / "predictions.json", "w") as f:
        json.dump(pr_preds, f)
    pr_eval = make_coco_eval(coco, pr_preds, subset_ids)
    pr = precision_recall_f1(pr_eval)

    print(f"\n===== Precision / Recall / F1 (conf >= {PR_CONF}, IoU >= {PR_IOU}) =====")
    print(f"TP={pr['tp']}  FP={pr['fp']}  FN={pr['fn']}  (GT boxes={pr['num_gt']})")
    print(f"Precision: {pr['precision']:.4f}")
    print(f"Recall:    {pr['recall']:.4f}")
    print(f"F1:        {pr['f1']:.4f}")

    s = map_eval.stats
    summary = {
        "model": MODEL_NAME,
        "num_images": len(subset_ids),
        "num_classes": len(coco.getCatIds()),
        "num_classes_present_in_subset": len(per_class_ap(coco, map_eval)),
        "mAP": {
            "confidence_threshold": MAP_CONF,
            "mAP_50_95": float(s[0]),
            "mAP_50": float(s[1]),
            "mAP_75": float(s[2]),
            "mAP_small": float(s[3]),
            "mAP_medium": float(s[4]),
            "mAP_large": float(s[5]),
        },
        "precision_recall": {
            "confidence_threshold": PR_CONF,
            "iou_threshold": PR_IOU,
            **pr,
        },
        "per_class": per_class_ap(coco, map_eval),
    }
    with open(OUT_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics_summary.json, predictions.json and sample images to {OUT_DIR}")
