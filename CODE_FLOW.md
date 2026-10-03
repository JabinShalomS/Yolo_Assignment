# Code Flow — A Beginner's Map of This Project

This explains **what each file does, in what order, and how data moves** through the project. No prior YOLO knowledge needed.

---

## 1. The big idea in plain words

The assignment: *build an object detector that is "preceded by a pre-trained model", on a standard dataset, and explain why the pre-trained model is useful.*

An **object detector** looks at a photo and draws boxes around things ("dog", "car", "person").
A **pre-trained model** is a model someone already taught on a huge number of images. We reuse that learning instead of starting from zero.

So we ran an experiment: **train the same small detector (YOLOv8n) four different ways and compare the scores.**

| # | Run name | Starts from | Gets a DINOv2 "hint"? |
|---|---|---|---|
| 1 | `baseline` | random weights (knows nothing) | no |
| 2 | `dino` | random weights | **yes** |
| 3 | `pretrained` | `yolov8n.pt` (already taught on COCO) | no |
| 4 | `dino_pretrained` | `yolov8n.pt` | **yes** |
| ref | `zeroshot` | `yolov8n.pt`, **no training at all** | no |

- **DINOv2** is a second pre-trained model (from Meta) that looks at the *whole* image and summarises it in one list of 384 numbers ("this is an indoor kitchen scene"). We hand that summary to YOLO as a hint. This is the idea from the I3D Lab web page.

---

## 2. The whole pipeline in one picture

```
 STEP 1: get data                    STEP 2: reshape data               STEP 3: train + score
 ────────────────                    ─────────────────────              ─────────────────────
 download_subset.py                  prepare_yolo_dataset.py            train_compare.py
 (official COCO images               (convert labels to YOLO            (trains 4 models, scores
  + annotations file)                 format, split 480 / 120)           them on the 120 val images)
        │                                    │                                   │
        ▼                                    ▼                                   ▼
 data/val2017_subset/  ───────────►  data/yolo_subset/  ────────────►  outputs/comparison.md
 data/annotations/                    ├─ images/train (480)             outputs/comparison.json
                                      ├─ images/val   (120)             outputs/runs/...
                                      └─ labels/...
                                                                     STEP 4 (extra): show it
                                                                     ───────────────────────
                                       evaluate.py  →  outputs/metrics_summary.json
                                       app.py       →  web demo (Gradio)
```

Run order if you start from nothing:

```bash
python scripts/download_subset.py        # 1. get the official data (needs internet)
python scripts/prepare_yolo_dataset.py   # 2. build the train/val split
python scripts/train_compare.py --runs zeroshot baseline dino pretrained dino_pretrained --epochs 20 --imgsz 416   # 3. (~2h)
python scripts/evaluate.py               # extra: older 300-image evaluation
python app.py                            # extra: web demo
```

---

## 3. File by file

### `scripts/download_subset.py` — "go get the data"
- Downloads COCO's official annotation file (`instances_val2017.json`) = the human-drawn correct boxes.
- Downloads the first 300 val2017 images that have at least one object. (Step 2 later tops this up to 600.)
- **Nothing is created or edited by us**; it's the official data.

### `scripts/prepare_yolo_dataset.py` — "put the data in the shape YOLO wants"
YOLO can't read COCO's annotation format directly, so this script:
1. Picks the first 600 images that have objects (downloads the missing ones).
2. **Shuffles with a fixed seed (0) and splits 80/20** → 480 `train` and 120 `val` images. No image is in both, which is what makes the final score honest.
3. Converts every box:
   - COCO says: `[left, top, width, height]` in pixels.
   - YOLO wants: `class, centre_x, centre_y, width, height`, each divided by the image size (so values are 0–1).
   - Class numbers also change: COCO ids are 1–90 with gaps; YOLO uses 0–79.
4. Writes one `.txt` label file per image and `coco_subset.yaml` (a small file telling YOLO where train/val live and the 80 class names).

### `scripts/train_compare.py` — "the main experiment"
For each run name it does the same thing:
1. **Build the model**: `YOLO("yolov8n.yaml")` = empty network; `YOLO("yolov8n.pt")` = network with pre-trained weights; `YOLO("yolov8n-dino.yaml")` = network with the DINOv2 hint.
2. **Train**: `model.train(...)` with identical settings every time (20 epochs, image size 416, batch 16, seed 0, CPU). An *epoch* = one pass over all 480 training images.
3. **Score**: reload the best saved checkpoint and run `val()` on the 120 held-out images → mAP50, mAP50-95, precision, recall; F1 is computed from precision and recall.
4. **Save** `outputs/runs/<name>.json`.
At the end it gathers every JSON into `outputs/comparison.md` (the table you show).

Special helper, `remap_pretrained_into_hybrid()` (only for run 4): the DINOv2 network has extra layers, so every layer after the first gets a new number. This function renames the pre-trained weights' layer numbers so each weight lands in the matching layer (otherwise they'd go to the wrong places).

The `zeroshot` run just calls `val()` on `yolov8n.pt` with no training.

### The patched Ultralytics: `ultralytics-src/` — "teach YOLO about DINOv2"
Ultralytics (the YOLO library) doesn't know what a "DINOv2 layer" is, so we installed an editable copy and added three small things:

| File | What we added | Why |
|---|---|---|
| `nn/modules/conv.py` | `ConvDummy` | A layer that does nothing; it just keeps the raw image available as "layer 0" so DINOv2 can read it later |
| `nn/modules/block.py` | `DINOv2` | The module: resize image → frozen DINOv2 → one 384-number summary → small trainable layer → copy that summary across the feature map |
| `nn/tasks.py` | a `DINOv2` branch in `parse_model` | The YAML reader needs to know the DINOv2 layer's output size; without it the network wiring breaks |
| `cfg/models/v8/yolov8-dino.yaml` | the architecture recipe | Same as normal YOLOv8 plus layers 0, 11, 12 for the DINOv2 hint |

Inside the hybrid, data flows like this:

```
raw image ─┬─► normal YOLO backbone ─► deepest features (SPPF, 256 channels) ──┐
           │                                                                    ├─► Concat ─► YOLO head ─► boxes
           └─► DINOv2 (frozen) ─► 1 summary vector ─► copy over grid (256 ch) ──┘
```

"Frozen" = DINOv2's own weights never change during training; only YOLO and the tiny 1×1 conversion layer learn.

### `scripts/evaluate.py` — "the older zero-shot score"
- Runs the published `yolov8n.pt` (no training) on the 300 images.
- Converts YOLO's boxes back into COCO's format (the reverse of step 2) and scores them with **pycocotools**, the official COCO scorer.
- Does two passes: confidence 0.001 for mAP, confidence 0.25 for precision/recall/F1 (details in `TA_GUIDE.md` section 5).
- Saves `outputs/metrics_summary.json`, `predictions.json` and 6 sample pictures.

### `app.py` — "the demo"
A small web page built with Gradio:
- **Detect tab**: upload a photo (or pick a COCO example), move the confidence slider, see predicted boxes (and the real boxes for COCO images).
- **Results tab**: shows the numbers saved by `evaluate.py`.
It uses the pre-trained `yolov8n.pt`; it doesn't depend on the training runs.

---

## 4. What's in `outputs/` and `data/`

| Path | What it is |
|---|---|
| `outputs/comparison.md` / `.json` | **Main results table** |
| `outputs/runs/<name>.json` | One run's scores |
| `outputs/runs/train/<name>/` | Training curves (`results.png`), confusion matrix, sample batches, and `weights/best.pt` |
| `outputs/train_log.txt` | Raw log of the training session |
| `outputs/metrics_summary.json`, `predictions.json`, `sample_*.jpg` | From the older `evaluate.py` |
| `data/annotations/instances_val2017.json` | Official COCO ground truth |
| `data/val2017_subset/` | The 600 downloaded images |
| `data/yolo_subset/` | The YOLO-format train/val split + `split.json` listing which image went where |

---

## 5. Words you'll hear

| Word | Simple meaning |
|---|---|
| Epoch | One full pass over the training images |
| Train / val split | Images the model learns from / images used only for scoring |
| Weights | The numbers inside the network that training adjusts |
| From scratch | Start with random weights |
| Pre-trained | Start with weights someone already trained |
| Fine-tune | Keep training pre-trained weights on your data |
| Zero-shot | Use a model as-is, no training |
| Frozen | Weights that are not allowed to change |
| mAP50 | How well boxes match real ones (overlap ≥ 50%), averaged over classes; higher is better, max 1.0 |
| Precision / Recall | Of boxes drawn, how many are right / of real objects, how many were found |
| F1 | One number combining precision and recall |

---

## 6. The results in one sentence each

- Starting from pre-trained weights: **mAP50 0.469** vs **0.001** from scratch → pre-training is what makes detection work with little data.
- Adding DINOv2: **did not help** in our short runs (0.153 with pre-trained YOLO, vs 0.469 without). We did not test why; probably short training and the layers after the fusion no longer matching the pre-trained weights.
- Fine-tuning vs using the weights as-is: about the same (0.469 vs 0.480), because the data is COCO, which the weights already know.

For likely questions and answers, see `TA_GUIDE.md` (section 4c and section 8). For the full method and fixes to the reference code, see `README.md`.
