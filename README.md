# Object Detection Assignment — YOLOv8 on COCO

## Task
Object detection preceded by a pre-trained model, on a standard annotated dataset, with the utility of the pre-trained model shown (see the training experiment below, then the original inference-only evaluation).

## Dataset
> The training experiment uses a 600-image COCO val2017 subset split 480 train / 120 val (see *Training experiment* below). The sections *Dataset*, *Model* and *Pipeline* here, and *Reference: inference-only evaluation*, describe the earlier **zero-shot** evaluation of the published weights on 300 images, kept as a reference.

**COCO val2017** (Common Objects in Context), official validation split.

Why this dataset:
- It is the standard benchmark dataset for object detection — every major detector (YOLO, Faster R-CNN, DETR, etc.) reports scores on it, so results are directly comparable to published numbers.
- 80 object classes across everyday scenes (people, vehicles, animals, household objects, etc.), with high-quality bounding-box + category annotations done by the COCO team — no annotation was created for this assignment.
- We use the **val2017** split specifically (not train2017) because the pretrained model's weights were trained on train2017. Evaluating on val2017 means the model has not seen these exact images before, so the reported accuracy reflects genuine generalization, not memorized training data.
- For runtime reasons (CPU-only laptop), we evaluate on a **300-image subset** of val2017 rather than the full 5,000-image split. All 300 images and their annotations are the untouched, official COCO data — just a smaller slice of it.

**Classes**: 80 (see `data/annotations/instances_val2017.json` categories, e.g. person, bicycle, car, dog, chair, laptop, ...).

## Model
**YOLOv8n** (`yolov8n.pt`), the "nano" pretrained checkpoint from Ultralytics, trained on COCO train2017. In this zero-shot evaluation no training is performed — the checkpoint is used exactly as published. (Training runs are in *Training experiment* below.)

## Pipeline
1. `scripts/download_subset.py` — downloads the official COCO annotations file and 300 val2017 images that have at least one annotated object.
2. `scripts/evaluate.py`:
   - Loads pretrained `yolov8n.pt` via `ultralytics.YOLO(...)`.
   - Runs `model.predict()` on each of the 300 images, twice:
     - at confidence 0.001 → used for **mAP** (mAP sweeps over all confidence levels, so it needs nearly all boxes; this is the standard protocol behind Ultralytics' published numbers).
     - at confidence 0.25 → used for **precision / recall / F1** at a realistic operating point.
   - Converts YOLO's output boxes into COCO's result format (`image_id`, `category_id`, `bbox`, `score`).
   - Scores predictions against ground truth using `pycocotools.cocoeval.COCOeval` — the same tool used to compute official published COCO benchmark numbers.
   - Saves 6 sample annotated images, `predictions.json` and `metrics_summary.json` (overall + per-class AP) to `outputs/`.
3. `app.py` — a small Gradio web UI for the live demo (see below).

## Training experiment: YOLO vs YOLO + pre-trained DINOv2 (main deliverable)

The assignment: *object detection preceded by a pre-trained model, on a standard dataset, with the effect of the
pre-trained model shown.* The I3D Lab page's two examples both attach a pre-trained model (DINOv2, Depth Anything V2)
to a YOLOv8. We follow the DINOv2 one and compare it with plain YOLOv8n.

**Dataset - COCO (Common Objects in Context).** Created by Microsoft and collaborators (Lin et al., 2014) to push
object recognition in *everyday scenes*: objects in natural, cluttered context rather than centred product-style photos.
Every object is annotated with a bounding box and a class by human annotators. **80 classes** (person, car, dog, chair,
laptop, ...). We create no annotations. We use 600 official val2017 images (first 600 ids with >= 1 object),
split with a fixed seed into **480 train / 120 val** (3,359 / 1,031 boxes) - no image is in both. A subset keeps
CPU training feasible. val2017 is used (not train2017) so the pre-trained `yolov8n.pt`, trained on train2017, has never seen these
images. Built by `scripts/prepare_yolo_dataset.py` (converts COCO boxes to YOLO format: class, centre x/y, width/height
normalised by image size; COCO category ids 1-90 -> YOLO ids 0-79).

**Models** (`scripts/train_compare.py`; every run uses the same data, 20 epochs, 416 px, batch 16, seed 0, CPU):
1. `YOLOv8n` from scratch (random weights) - baseline.
2. `YOLOv8n + DINOv2` from scratch - the page's hybrid.
3. `YOLOv8n` starting from COCO-pretrained `yolov8n.pt`, fine-tuned.
4. The hybrid whose YOLO layers start from `yolov8n.pt` (extra), plus a no-training reference: `yolov8n.pt` evaluated as-is.

**Results** on the 120 held-out val images (mAP50-95 also reported; precision/recall are class means at the
confidence that maximises F1):

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Params (trainable) | Train min |
|---|---|---|---|---|---|---|---|
| YOLOv8n pretrained, no fine-tuning (reference) | 0.480 | 0.351 | 0.617 | 0.427 | 0.505 | 3.16M (none) | - |
| YOLOv8n (from scratch) | 0.001 | 0.000 | 0.001 | 0.014 | 0.002 | 3.16M (3.16M) | 17.1 |
| YOLOv8n + DINOv2 (from scratch) | 0.002 | 0.000 | 0.002 | 0.017 | 0.003 | 25.41M (3.35M) | 46.5 |
| YOLOv8n pretrained (fine-tuned) | 0.469 | 0.341 | 0.629 | 0.425 | 0.507 | 3.16M (3.16M) | 22.9 |
| YOLOv8n-pretrained + DINOv2 (fine-tuned) | 0.153 | 0.092 | 0.357 | 0.162 | 0.223 | 25.41M (3.35M) | 24.2 |

**What this shows**
- **Pre-trained weights are what make detection work here.** From scratch, 480 images and 20 epochs give mAP50 about 0.001 (essentially nothing learned);
  starting from pre-trained weights gives 0.469, a difference of several hundred times. A detector needs many thousands of images to learn from scratch; the pre-trained
  checkpoint carries that learning.
- **Fine-tuning did not beat using the weights as-is (0.469 vs 0.480).** Expected: the data *is* COCO, which the model already knows, so
  there is nothing new to learn. Fine-tuning would matter on a new domain.
- **DINOv2 did not help in our setting.** From scratch both models are near zero, so the comparison is inconclusive (0.002 vs 0.001 is noise).
  With pre-trained YOLO, adding DINOv2 *lowered* mAP50 from 0.469 to 0.153. Likely reasons (not separately tested): the fusion changes the input of
  two head layers, which restart from random weights (the other 353 tensors are copied), the new DINOv2 projector is untrained, and 20 epochs on 480 images
  is too little to recover. The page's gain is for domain-specific scenes (e.g. factories) trained on far more data; we did not reproduce it.
- Caveat: one seed, small data, short training. Differences of a few points would not be reliable; the big gaps (pre-trained vs scratch) are.

**How the hybrid works (for the walkthrough)**
- YOLOv8's backbone sees local edges and textures. DINOv2 (ViT-S/14, trained on ~142M images without labels) sees the *whole image* and outputs one 384-number
  summary ("what kind of scene is this").
- `yolov8-dino.yaml`: layer 0 `ConvDummy` passes the raw image through unchanged so layer 11 `DINOv2` can read it (`from: 0`); layer 12 concatenates
  DINOv2's output with YOLO's deepest features (SPPF, layer 10), and the head uses that fused map.
- `DINOv2` (`ultralytics-src/ultralytics/nn/modules/block.py`): resize to 224 px, ImageNet-normalise, run frozen DINOv2 under `no_grad`, apply a trainable 1x1 conv (384 -> 256 channels), and copy the one vector over every cell of the stride-32 grid.
- DINOv2's weights never change: verified after training, all 175 tensors identical to the originals. Only YOLO and the small projector train.

**Where the page's reference code needed fixes**
1. `parse_model` (`nn/tasks.py`) needs an explicit `DINOv2` branch. Otherwise it records the wrong output channels and calls `DINOv2(1024)` instead of `DINOv2(c1, c2)`.
2. The YAML must be named `yolov8-dino.yaml`: Ultralytics strips the scale letter from `yolov8n-dino.yaml` when locating the file. All layer indices after 0 shift by one, so the head's `from` indices were rewritten.
3. DINOv2 gets ImageNet mean/std (not 0.5), and runs at 224 px instead of 512 to stay CPU-friendly. The page's `Upsample(scale_factor=1)` did nothing, so it is replaced by broadcasting the vector.
4. Ultralytics sets `requires_grad=True` on all parameters when training starts, so the freeze is enforced inside the module (`no_grad` + forced eval mode).
5. Loading `yolov8n.pt` into the hybrid by name would put weights in the wrong layers (indices shifted); `remap_pretrained_into_hybrid()` renames keys first (old layer i -> i+1 for i <= 9, i+3 after).

**Reproduce**
```bash
source venv/bin/activate
pip install -e ultralytics-src            # patched Ultralytics v8.4.163 (ConvDummy, DINOv2, tasks.py branch)
python scripts/prepare_yolo_dataset.py    # builds data/yolo_subset (480 train / 120 val)
python scripts/train_compare.py --runs zeroshot baseline dino pretrained dino_pretrained --epochs 20 --imgsz 416
```
Outputs: `outputs/comparison.md` / `.json`, per-run curves in `outputs/runs/train/<run>/`. About 2 hours on an M1 CPU. The first DINOv2 use downloads it via torch.hub.

The original inference-only evaluation on 300 images (below) is kept as a reference; its numbers come from a different, larger image set and are not comparable to this table.

## Reference: inference-only evaluation of pretrained YOLOv8n

Evaluated on 300 COCO val2017 images (2,147 ground-truth boxes, 77 of the 80 classes present), model = `yolov8n.pt`.

**mAP** (standard COCO protocol, conf ≥ 0.001):

| Metric | Value |
|---|---|
| mAP@[0.5:0.95] | 0.418 |
| mAP@0.5 | 0.582 |
| mAP@0.75 | 0.445 |
| mAP small / medium / large objects | 0.254 / 0.439 / 0.578 |

**Precision / Recall / F1** (conf ≥ 0.25, a prediction is correct if IoU ≥ 0.5 with a same-class ground-truth box):

| Precision | Recall | F1 | TP | FP | FN (missed) |
|---|---|---|---|---|---|
| 0.712 | 0.519 | 0.600 | 1115 | 452 | 1032 |

How to read these:
- **Precision 0.71** — of the boxes the model draws, 71% are real objects of the right class in the right place.
- **Recall 0.52** — the model finds about half of all annotated objects. Small objects are the hardest (small-object mAP 0.25 vs 0.58 for large), which is expected for the smallest "nano" model.
- **mAP@0.5 vs mAP@[0.5:0.95]** — the second one also demands tight boxes (IoU up to 0.95), so it's always lower.

For reference, Ultralytics' published `yolov8n` result on the **full** 5,000-image val2017 set is mAP@[0.5:0.95] = 0.373, mAP@0.5 = 0.526, evaluated with the same protocol. Our number is a bit higher because 300 images is a small sample: which images you pick changes the score by a few points. The full-set number is the more reliable estimate of the model's true accuracy.

Per-class AP is in `outputs/metrics_summary.json` and in the UI's "Results" tab. Sample annotated detections: `outputs/sample_*.jpg`.

## Utility of the pre-trained model

Measured above: pre-trained weights give mAP50 0.469 vs 0.001 from scratch on the same data and epochs.

Why use a pre-trained detector instead of training one:
- **No training from scratch.** YOLOv8n was already trained on COCO train2017 (~118k images). Reproducing that needs many GPU-hours; here we download the weights and run inference on a CPU laptop. Our training runs show the same thing: from scratch reaches mAP50 0.001 on 480 images.
- **Works out of the box.** It detects the 80 COCO classes immediately, with no labelling or training effort on our side.
- **Generalizes to unseen data.** On val2017 images it never trained on, it scores mAP@[0.5:0.95] = 0.418 on our 300-image subset (Ultralytics publishes 0.373 on the full 5,000-image set).
- **A base to build on.** The weights can be fine-tuned on a custom dataset (transfer learning) or extended with other models.

Relation to the examples on the I3D Lab page (https://cambum.net/I3DLab/AI4DM.htm), "Object Detection using Pre-Trained Model":
- Both examples there are built on **YOLOv8**: *YOLOv8 + DINOv2* (adds a global scene embedding to the deepest YOLO layer as context) and *YOLOv8 + Depth Anything V2* (YOLO detects, Depth Anything V2 estimates distance from a monocular camera). `yolov8n.pt` is the same base model; in the training experiment above we also attached DINOv2 to it.
- Limits we measured: recall 0.52 and small-object mAP 0.25. Extra context, as in the DINOv2 hybrid, is one way to address such misses; we implemented and tested it above, and in our short runs it did not help.

## Demo UI
```bash
source venv/bin/activate
python app.py        # then open http://127.0.0.1:7860
```
- **Detect** tab: upload any image (or click a COCO example), adjust the confidence slider, and see the predicted boxes plus a table of class / confidence / pixel coordinates. For COCO val2017 images, the official ground-truth boxes are drawn alongside so you can compare them visually.
- **Results** tab: the evaluation metrics above, plus AP for each class.

## How to reproduce
```bash
cd yolo-coco-detection
source venv/bin/activate
python scripts/download_subset.py   # one-time, ~250MB
python scripts/evaluate.py         # ~2 min on CPU
python app.py                      # optional: demo UI
```

## Explaining the pretrained-model integration (for the TA walkthrough)
Full prep notes (metrics explained, demo script, likely questions): see `TA_GUIDE.md`.

- `YOLO("yolov8n.pt")` in `evaluate.py` — Ultralytics auto-downloads the pretrained weights (trained on COCO) if not already cached locally.
- `model.predict(image, conf=...)` runs a forward pass: image → CNN backbone → detection head → boxes with class + confidence, dropping boxes below the confidence threshold (0.001 for mAP, 0.25 for P/R/F1 and the UI default).
- `evaluate.py` does no training (inference only). Training happens in `train_compare.py`.
- Box format conversion (`xywh` center-format from Ultralytics → COCO's `[x_min, y_min, w, h]`) and category id remapping (Ultralytics uses contiguous 0-79 ids; COCO's official category ids are 1-90 with gaps) is the main "glue code" — worth being ready to explain that mismatch.
- `COCOeval` handles IoU matching between predicted and ground-truth boxes per class, then aggregates precision/recall across IoU thresholds 0.5 to 0.95 to produce mAP.
- Precision/recall/F1 come from COCOeval's own box matching (`evalImgs`): each prediction is either matched to a ground-truth box (TP) or not (FP), and unmatched ground-truth boxes are misses (FN). See `precision_recall_f1()` in `evaluate.py`.
