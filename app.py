"""
Small web UI for the demo:
  Tab 1 "Detect"   -> upload any image, pretrained YOLOv8n draws the boxes.
                      If the image is one of our COCO val2017 subset images, the
                      official ground-truth boxes are drawn next to it for comparison.
  Tab 2 "Results"  -> the evaluation numbers produced by scripts/evaluate.py.

Run:  python app.py   then open http://127.0.0.1:7860
"""

import json
from pathlib import Path

import cv2
import gradio as gr
from pycocotools.coco import COCO
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "data" / "val2017_subset"
ANN_PATH = ROOT / "data" / "annotations" / "instances_val2017.json"
METRICS_PATH = ROOT / "outputs" / "metrics_summary.json"

model = YOLO("yolov8n.pt")  # same pretrained checkpoint as the evaluation, no training
coco = COCO(str(ANN_PATH))
file_to_img_id = {img["file_name"]: img_id for img_id, img in coco.imgs.items()}


def draw_ground_truth(image_path: str):
    """Draw official COCO annotation boxes, or return None if the image isn't from COCO val2017."""
    img_id = file_to_img_id.get(Path(image_path).name)
    if img_id is None:
        return None
    img = cv2.imread(image_path)
    for ann in coco.imgToAnns[img_id]:
        x, y, w, h = map(int, ann["bbox"])  # COCO format: top-left x, y, width, height
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(img, coco.cats[ann["category_id"]]["name"], (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect(image_path: str, conf: float):
    if image_path is None:
        return None, None, []
    result = model.predict(image_path, conf=conf, verbose=False)[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)  # plot() returns BGR

    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append([model.names[int(box.cls)], round(float(box.conf), 3),
                     round(x1), round(y1), round(x2), round(y2)])
    return annotated, draw_ground_truth(image_path), rows


def results_markdown():
    if not METRICS_PATH.exists():
        return "No results yet — run `python scripts/evaluate.py` first."
    m = json.loads(METRICS_PATH.read_text())
    a, p = m["mAP"], m["precision_recall"]
    return f"""
**Dataset:** COCO val2017, {m['num_images']}-image subset · **Classes:** {m['num_classes']}
({m['num_classes_present_in_subset']} appear in the subset) · **Model:** `{m['model']}` (pretrained, no fine-tuning)

### Accuracy (mAP, conf ≥ {a['confidence_threshold']}, standard COCO protocol)
| Metric | Value |
|---|---|
| mAP@[0.5:0.95] | {a['mAP_50_95']:.3f} |
| mAP@0.5 | {a['mAP_50']:.3f} |
| mAP@0.75 | {a['mAP_75']:.3f} |
| mAP small / medium / large objects | {a['mAP_small']:.3f} / {a['mAP_medium']:.3f} / {a['mAP_large']:.3f} |

### Precision / Recall / F1 (conf ≥ {p['confidence_threshold']}, IoU ≥ {p['iou_threshold']})
| Precision | Recall | F1 | TP | FP | FN (missed) |
|---|---|---|---|---|---|
| {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['tp']} | {p['fp']} | {p['fn']} |
"""


def per_class_rows():
    if not METRICS_PATH.exists():
        return []
    m = json.loads(METRICS_PATH.read_text())
    return [[r["class"], r["num_gt"], round(r["AP_50_95"], 3), round(r["AP_50"], 3)]
            for r in m["per_class"]]


examples = [[str(IMG_DIR / name)] for name in
            ["000000000139.jpg", "000000000632.jpg", "000000000785.jpg", "000000001000.jpg"]
            if (IMG_DIR / name).exists()]

with gr.Blocks(title="YOLOv8 on COCO") as demo:
    gr.Markdown("# Object detection — pretrained YOLOv8n on COCO")
    with gr.Tab("Detect"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="filepath", label="Input image")
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
                btn = gr.Button("Detect", variant="primary")
                gr.Examples(examples, inputs=inp, label="COCO val2017 examples (ground truth shown too)")
            with gr.Column():
                out_pred = gr.Image(label="Model predictions")
                out_gt = gr.Image(label="COCO ground truth (only for val2017 images)")
        table = gr.Dataframe(headers=["class", "confidence", "x1", "y1", "x2", "y2"],
                             label="Detections (pixel coordinates)")
        btn.click(detect, [inp, conf], [out_pred, out_gt, table])
        inp.change(detect, [inp, conf], [out_pred, out_gt, table])
        conf.release(detect, [inp, conf], [out_pred, out_gt, table])

    with gr.Tab("Results"):
        gr.Markdown(results_markdown())
        gr.Markdown("### Per-class AP (sorted best → worst)")
        gr.Dataframe(value=per_class_rows(), headers=["class", "GT boxes", "AP@[.5:.95]", "AP@.5"])

if __name__ == "__main__":
    demo.launch()
