# TA Walkthrough Guide

Your prep notes for the live evaluation (3 Oct, 1:30–3:00 PM). Read top to bottom once; during the session, use the headings to jump.

---

## 0. Before the session (checklist)

```bash
cd yolo-coco-detection
source venv/bin/activate
cat outputs/comparison.md      # the main result: 4 trained runs + zero-shot reference
python scripts/evaluate.py     # ~2 min, the older 300-image pycocotools evaluation (zero-shot reference)
python app.py                  # open http://127.0.0.1:7860 and keep the tab ready
```

Have these open in your editor: `README.md`, `outputs/comparison.md`, `scripts/train_compare.py`, `ultralytics-src/ultralytics/cfg/models/v8/yolov8-dino.yaml`, the `DINOv2` class in `ultralytics-src/ultralytics/nn/modules/block.py`, `scripts/evaluate.py`, `app.py`.

Retraining everything takes about 2 hours (`python scripts/train_compare.py --runs zeroshot baseline dino pretrained dino_pretrained --epochs 20 --imgsz 416`), so do not rerun it live; show the saved results.

---

## 1. The 30-second summary (say this first)

> "The brief was object detection preceded by a pre-trained model. I used a standard dataset, **COCO**, and trained **YOLOv8n** with identical settings in four ways: **from scratch**, **from scratch plus a frozen pre-trained DINOv2** (the approach on your lab page), **from COCO-pretrained weights**, and **pretrained weights plus DINOv2**. I trained on **480 images and evaluated on 120 held-out ones**. Pre-trained YOLO weights took **mAP50 from 0.001 to 0.469**. **DINOv2 did not help** in my short runs. My earlier 300-image pycocotools evaluation of the published weights is the zero-shot reference. I also built a small UI where you can upload an image and see the boxes."

---

## 2. How the assignment requirements are met

| Requirement | What I did | Where |
|---|---|---|
| Use an existing standard dataset, don't annotate | COCO val2017, official images + official `instances_val2017.json` annotations, downloaded unmodified; converted to YOLO format, 480 train / 120 val | `scripts/download_subset.py`, `scripts/prepare_yolo_dataset.py` |
| Run YOLO through training with a train/val split; report F1, mAP50 | Four 20-epoch runs with identical settings, evaluated on the 120 held-out images | `scripts/train_compare.py`, `outputs/comparison.md` |
| Show how the metrics change with a pre-trained model | Pre-trained YOLO weights (mAP50 0.469 vs 0.001 from scratch) and pre-trained DINOv2 attached to YOLO (did not help) | section 4c, README |
| Reference: the published weights with no training | `YOLO("yolov8n.pt")` evaluated as-is (zero-shot) | `scripts/evaluate.py:141`, `train_compare.py --runs zeroshot` |
| Report dataset + why | COCO: the standard detection benchmark; val2017 because the model was trained on train2017 | README "Dataset" |
| Report number of classes | 80 (77 of them appear in our 300-image subset — `toaster`, `hair drier`, `toothbrush` don't) | `metrics_summary.json` |
| Report outputs / metrics | mAP, mAP@0.5, mAP@0.75, small/medium/large mAP, precision, recall, F1, per-class AP | README "Results", UI "Results" tab |
| Explain the integration code | 4c (DINOv2 hybrid) and sections 5-6 (evaluation pipeline) | `block.py`, `yolov8-dino.yaml`, `train_compare.py`, `evaluate.py` |

---

## 3. The dataset — COCO val2017

- **COCO** = "Common Objects in Context", from Microsoft. The benchmark every detector (YOLO, Faster R-CNN, DETR…) reports on, so our numbers are comparable to published ones.
- **80 classes** of everyday objects: person, car, dog, chair, laptop, pizza, …
- Each annotation = a class + a bounding box `[x_min, y_min, width, height]` in pixels (top-left corner + size).
- **Splits:** train2017 (~118k images) and val2017 (5,000 images).
- **Why val2017:** YOLOv8n was trained on train2017. Testing on val2017 means the model has **never seen these images** — so the score measures real generalization, not memorization.
- **Training experiment split:** the first 600 annotated val2017 ids, split 80/20 with a fixed seed into **480 train / 120 val** (3,359 / 1,031 boxes), no overlap (`scripts/prepare_yolo_dataset.py`). The 300-image set below is the earlier zero-shot evaluation; it is the first 300 of those same 600.
- **Why only 300 images (zero-shot evaluation):** CPU-only laptop. The subset = the first 300 val2017 image IDs (sorted) that have at least one annotation — deterministic and reproducible, not cherry-picked. They contain **2,147 ground-truth boxes**; the most common class is `person` (686 boxes).

---

## 4. The model — YOLOv8n

- **YOLO** = "You Only Look Once": a *single-stage* detector — one forward pass of the network directly outputs boxes + classes (vs. two-stage detectors like Faster R-CNN that first propose regions, then classify them). That's why it's fast.
- **v8** = Ultralytics' 2023 version. **n** = "nano", the smallest size (n < s < m < l < x): ~3.2M parameters, fast on CPU, least accurate.
- **Architecture (if asked):**
  - **Backbone** (CNN) — extracts visual features from the image.
  - **Neck** (feature pyramid, FPN/PAN) — combines features at several scales, so it can find both small and large objects.
  - **Head** — for each location predicts a box and a score for each of the 80 classes. YOLOv8 is **anchor-free** (predicts boxes directly instead of adjusting preset "anchor" boxes).
- **What `predict()` does internally:**
  1. Resizes the image to 640 px (keeping aspect ratio, padding the rest — "letterbox").
  2. Forward pass through the network.
  3. Drops boxes below the **confidence threshold** (`conf`).
  4. **NMS (non-maximum suppression)** — when several boxes overlap the same object, keep only the highest-scoring one.
  5. Scales boxes back to the **original image's pixel coordinates**.
- **Published accuracy** on the full 5,000-image val2017: mAP@[0.5:0.95] = 0.373, mAP@0.5 = 0.526.

---

## 4b. Utility of the pre-trained model (the assignment asks us to explain this)

Say: "Training a detector from scratch needs many thousands of images and GPU-hours. A pre-trained checkpoint gives us that learning for free. I measured it: on my 120 held-out images, pre-trained weights give mAP50 0.469 against 0.001 from scratch with the same data and epochs. It can also be fine-tuned later on a custom dataset."

Link to the I3D Lab page (cambum.net/I3DLab/AI4DM.htm, "Object Detection using Pre-Trained Model"):
- Its two examples, YOLOv8 + DINOv2 and YOLOv8 + Depth Anything V2, both attach a pre-trained model to YOLOv8. I implemented the DINOv2 one and compared it with plain YOLOv8n (section 4c).
- DINOv2 adds a global scene embedding to the deepest YOLO layer as context. In my short runs it did not improve the metrics.
- Depth Anything V2 adds distance estimation from a monocular camera; I did not implement it (not needed for plain detection).

---

## 4c. The training comparison (what he asked for on 5 Sept)

One-liner: "I trained plain YOLOv8n and YOLOv8n + a frozen pre-trained DINOv2 with identical settings on 480 COCO images and evaluated on 120 held-out ones; I also compared with/without COCO-pretrained YOLO weights." Table: `outputs/comparison.md` (also in README).

Numbers to remember (mAP50 / F1): from scratch 0.001 / 0.002; + DINOv2 from scratch 0.002 / 0.003; pretrained fine-tuned 0.469 / 0.507; pretrained + DINOv2 0.153 / 0.223; pretrained as-is 0.480 / 0.505.

Likely questions:
- *Why is from-scratch ~0?* 480 images / 20 epochs is far too little to learn 80 classes from random weights; that is exactly the utility of pre-training.
- *Did DINOv2 help?* Not in our run, and I say so. From scratch both are noise; with pretrained YOLO it dropped mAP50 (0.469 -> 0.153). I did not test why. Probably: training is short, and the layers after the fusion no longer match the pretrained weights (two restart from random). The page's gain is for domain-specific data with much more training.
- *Why do you have two different mAP50 numbers for the same pretrained model (0.582 and 0.480)?* Different image sets and different evaluators: 0.582 is pycocotools on 300 images (earlier zero-shot evaluation, conf 0.001); 0.480 is Ultralytics' validator on the 120 held-out images at 416 px. A small sample shifts the score by several points. The training table is internally consistent because every row uses the same 120 images and validator.
- *Why didn't fine-tuning beat the zero-shot weights?* The data is COCO, which yolov8n.pt already knows.
- *Is DINOv2 being trained?* No: `no_grad` + eval mode; I checked all 175 weight tensors are unchanged after training. Only YOLO and the 1x1 projector learn.
- *What does DINOv2 add?* One global embedding of the whole image (384 numbers), projected to 256 channels and copied to every cell of YOLO's deepest feature map, then concatenated with SPPF. Code: `DINOv2` in `ultralytics-src/ultralytics/nn/modules/block.py`; layers 0, 11, 12 in `yolov8-dino.yaml`.
- *What did you have to change vs the page?* README list of 5 fixes (parse_model branch, YAML name/indices, normalisation/224px, freezing, weight-index remap).
- *Why this split?* 480/120 from official COCO val2017, fixed seed, no overlap; val2017 so the pretrained weights never saw them.
- *Limits:* one seed, small data, 20 epochs, 416 px; only big gaps are reliable.

## 5. Code walkthrough — the pretrained-model integration (the part they asked for)

### Pipeline at a glance
```
download_subset.py:  COCO annotations + 300 val2017 images  ->  data/
evaluate.py:         yolov8n.pt -> predict() -> convert to COCO format -> COCOeval -> metrics
app.py:              yolov8n.pt -> predict() -> draw boxes in a web page
```

### 5.1 Loading the pretrained model — `evaluate.py:141`
```python
model = YOLO(MODEL_NAME)   # "yolov8n.pt"
```
Loads the published weights (downloads them on first run). **No training** — we only call `predict()`.

### 5.2 Running inference — `evaluate.py:52`
```python
preds = model.predict(str(img_path), conf=conf, verbose=False)[0]
```
Returns a `Results` object; `preds.boxes` holds every detected box with its class id (`box.cls`), confidence (`box.conf`) and coordinates.

### 5.3 The "glue code" — converting YOLO output to COCO format (lines 54–66) ⭐
Two mismatches between Ultralytics and COCO must be fixed, otherwise every prediction would be scored as wrong:

**(a) Box format.** Ultralytics `xywh` = **centre** x, y + width, height. COCO = **top-left** x, y + width, height.
```python
x_c, y_c, w, h = box.xywh[0].tolist()
x_min, y_min = x_c - w / 2, y_c - h / 2       # move from centre to top-left corner
```
*Example:* centre (100, 80), size 40×20 → top-left (80, 70), size 40×20.

**(b) Class IDs.** Ultralytics numbers classes **0–79** with no gaps. COCO's official category IDs are **1–90 with gaps** (e.g. 12, 26, 29 … are unused, left over from the original 91-category plan).
```python
coco_cat_ids = sorted(coco.getCatIds())       # [1, 2, 3, ..., 11, 13, ..., 90]  (80 ids)
"category_id": coco_cat_ids[cls_id]           # YOLO 0 -> COCO 1 (person), YOLO 11 -> COCO 13 (stop sign)
```
This works because YOLO's class order is the same as COCO's sorted IDs (I verified all 80 names match).

Each prediction becomes: `{"image_id": 139, "category_id": 72, "bbox": [6.2, 166.2, 148.3, 95.6], "score": 0.93}` (that one's a TV).

### 5.4 Scoring — `make_coco_eval()`, lines 77–83
```python
coco_dt = coco.loadRes(results_json)             # load our predictions
coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
coco_eval.params.imgIds = subset_ids             # only score our 300 images
coco_eval.evaluate()                             # match predictions to ground truth
coco_eval.accumulate()                           # build precision-recall curves
```
Without the `imgIds` line, COCOeval would count all 5,000 images' objects as "missed".

### 5.5 Why two inference passes (conf 0.001 and 0.25)
- **mAP** measures quality across **all** confidence levels (it's the area under the precision–recall curve), so it needs nearly all boxes → `conf=0.001`. This is exactly how Ultralytics computes its published numbers, so ours are comparable.
- **Precision / recall / F1** describe **one operating point** — "if I only keep boxes the model is ≥ 25% confident about" → `conf=0.25` (Ultralytics' default for real use).
- Using 0.25 for mAP would cut off part of the curve and understate mAP.

### 5.6 Precision / recall / F1 — `precision_recall_f1()`, lines 104–134
COCOeval already matched each prediction to a ground-truth box. `evalImgs` records, per image and class, which predictions got a match. We count:
- **TP** (true positive) = prediction matched a GT box of the same class with IoU ≥ 0.5
- **FP** (false positive) = prediction with no match (wrong place, wrong class, or duplicate)
- **FN** (false negative) = GT box nobody matched (missed object)

Then precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2PR / (P + R).

### 5.7 The UI — `app.py`
- Same `YOLO("yolov8n.pt")` + `model.predict()`.
- `result.plot()` draws the predicted boxes.
- If the uploaded image is a COCO val2017 image, `draw_ground_truth()` draws the official annotation boxes in green, next to the prediction — a visual check.
- Built with **Gradio**, a Python library for quick ML demo web pages.

---

## 6. Metrics explained simply

### IoU (Intersection over Union)
How much a predicted box overlaps the real box: **overlap area ÷ combined area**. 1.0 = perfect, 0 = no overlap. A prediction "counts" if IoU ≥ the threshold (0.5 = at least roughly half overlapping).

### Precision and recall
- **Precision** = of the boxes the model drew, how many are correct? → *"When it says there's a dog, is there a dog?"*
- **Recall** = of all real objects, how many did it find? → *"Did it find all the dogs?"*
- Trade-off: raise the confidence threshold → fewer, surer boxes → precision ↑, recall ↓. (Show this with the slider in the UI.)

### AP and mAP
- **AP** (Average Precision) for one class = area under its precision–recall curve (COCO samples it at 101 recall points). One number summarizing precision at every recall level.
- **mAP** = mean of AP over all classes.
- **mAP@0.5** = IoU threshold 0.5 (box roughly right).
- **mAP@[0.5:0.95]** = the main COCO metric: average over 10 IoU thresholds 0.50, 0.55, …, 0.95 — also rewards **tight** boxes. Always lower than mAP@0.5.
- **small / medium / large** = objects < 32×32 px, 32²–96² px, > 96×96 px.

### Our numbers, in plain words
| Metric | Value | Meaning |
|---|---|---|
| mAP@[0.5:0.95] | 0.418 | Main COCO score (published full-set: 0.373) |
| mAP@0.5 | 0.582 | With a lenient overlap requirement |
| mAP@0.75 | 0.445 | With a strict overlap requirement |
| mAP small / med / large | 0.254 / 0.439 / 0.578 | Small objects are much harder |
| Precision (conf ≥ 0.25) | 0.712 | 71% of drawn boxes are correct |
| Recall (conf ≥ 0.25) | 0.519 | Finds about half of all objects |
| F1 | 0.600 | Balance of the two |
| TP / FP / FN | 1115 / 452 / 1032 | Out of 2,147 real objects |

**Per-class (UI Results tab):** best — giraffe 1.00, bear 0.94, toilet 0.90, dog 0.87 (large, distinctive). Worst — remote 0.02, donut 0.01, scissors 0.00 (small, cluttered). person (686 boxes): 0.53.
⚠️ Classes with only 2–9 boxes in the subset have noisy AP — don't over-interpret single classes.

---

## 7. Demo script (~5 minutes)

1. **Summary** (section 1) — 30 sec.
2. **README** — dataset, why val2017, 80 classes — 1 min.
3. **`evaluate.py`** — walk through 5.1 → 5.4, stressing the **glue code** (5.3) — 2 min.
4. **UI → Results tab** — read out mAP, precision, recall — 1 min.
5. **UI → Detect tab:**
   - Click a COCO example → predictions vs. green ground truth side by side.
   - Move the confidence slider up → boxes disappear (precision ↑, recall ↓), down → more boxes appear.
   - Upload a photo the TA picks (or one from your phone).

---

## 8. Likely TA questions — with answers

**Q: Did you train anything?**
Yes: four short runs for the comparison, 20 epochs each on 480 images, evaluated on 120 held-out ones. I also evaluated the published `yolov8n.pt` with no training as a reference.

**Q: Why COCO? Why val2017 and not train2017?**
It's the standard detection benchmark, so results are comparable. The model was trained on train2017; val2017 is unseen data, so the score reflects generalization.

**Q: Why only 300 images?**
Runtime on a CPU laptop. The images and annotations are unmodified official COCO data. The subset is the first 300 annotated images by ID — deterministic, not hand-picked.

**Q: Your mAP (0.418) is higher than the official 0.373. Why?**
Sample size. With 300 images the score varies by a few points depending on which images are in the set; some classes appear only 2–3 times. The 5,000-image number is the more reliable estimate. The evaluation method is the same (pycocotools, conf 0.001).

**Q: What's the difference between mAP@0.5 and mAP@[0.5:0.95]?**
0.5 only asks for boxes that roughly overlap; [0.5:0.95] averages over stricter overlap requirements too, so it also rewards precise box placement.

**Q: Why is recall only 0.52?**
It's the smallest YOLO model, and COCO has many small, crowded, partially hidden objects. Small-object mAP (0.25) is far below large-object mAP (0.58). Lowering the confidence threshold would raise recall at the cost of precision; a bigger model (yolov8s/m) would raise both.

**Q: Why did you need to convert the box format / class IDs?**
Ultralytics and COCO use different conventions (centre vs. top-left corner; 0–79 vs. 1–90 with gaps). Without converting, every box would land in the wrong place or be scored as the wrong class and mAP would be ~0.

**Q: How do you know the boxes are right?**
(a) mAP is in the expected range — a coordinate or ID bug would give near-zero. (b) The UI draws ground truth next to predictions and they line up. (c) For predictions with confidence ≥ 0.5, the median overlap (IoU) with the best-matching real box is 0.89.

**Q: What is NMS?**
Non-maximum suppression: the network often outputs several overlapping boxes for one object; NMS keeps the highest-confidence one and removes others that overlap it heavily.

**Q: What does the confidence score mean?**
The model's score (0–1) for "there's an object of this class in this box". Boxes below the threshold are discarded.

**Q: What is a false positive vs a false negative here?**
FP: a box where there's no matching real object (or a duplicate / wrong class). FN: a real object with no correct box.

**Q: How would you improve the results?**
Use a larger model (yolov8s/m/l/x), a higher input resolution for small objects, or fine-tune on a specific domain. For the DINOv2 hybrid: train longer on more data, and initialise or warm up the layers after the fusion so the pretrained weights still fit.

**Q: Did you write the code yourself?**
I used an AI coding assistant, which the assignment allows. I can explain every part (be ready to point to sections 5.1–5.6).

---

## 9. Glossary

| Term | Meaning |
|---|---|
| Bounding box | Rectangle around an object |
| Ground truth (GT) | The human-made correct annotations |
| Pretrained model | A model already trained by someone else on a large dataset; used as-is (zero-shot) or as the starting point for fine-tuning |
| Fine-tuning | Continuing training of a pretrained model on your own data |
| DINOv2 | Pre-trained self-supervised vision model (ViT); we use it frozen to give YOLO a whole-image summary vector |
| Frozen | Weights not updated during training |
| Inference | Running a trained model to get predictions (no learning) |
| IoU | Overlap between two boxes (0–1) |
| Confidence threshold | Minimum score for a box to be kept |
| NMS | Removes duplicate overlapping boxes |
| TP / FP / FN | Correct box / wrong box / missed object |
| Precision / Recall | Correctness of boxes drawn / fraction of objects found |
| AP / mAP | Area under the precision-recall curve / its mean over classes |
| pycocotools / COCOeval | Official COCO evaluation library |
| Gradio | Python library for simple ML web demos |

---

## 10. Files in the project

| File | Purpose |
|---|---|
| `scripts/download_subset.py` | Downloads COCO annotations + the 300 images |
| `scripts/prepare_yolo_dataset.py` | Builds the 480 train / 120 val YOLO-format dataset |
| `scripts/train_compare.py` | Trains the 4 variants, evaluates them and the zero-shot reference, writes the comparison |
| `ultralytics-src/` | Editable Ultralytics v8.4.163 with `ConvDummy`, `DINOv2` and the `tasks.py` branch added |
| `ultralytics-src/ultralytics/cfg/models/v8/yolov8-dino.yaml` | The hybrid architecture |
| `outputs/comparison.md` / `.json` | Main results table |
| `outputs/runs/` | Per-run results and training curves |
| `scripts/evaluate.py` | Zero-shot pycocotools evaluation of `yolov8n.pt` on 300 images (reference) |
| `app.py` | Web UI demo |
| `yolov8n.pt` | Pretrained model weights |
| `data/annotations/instances_val2017.json` | Official COCO ground truth |
| `data/subset_image_ids.json` | The 300 image IDs used |
| `outputs/metrics_summary.json` | All metrics incl. per-class AP |
| `outputs/predictions.json` | Every prediction (conf ≥ 0.25) in COCO format |
| `outputs/sample_*.jpg` | 6 annotated example images |
