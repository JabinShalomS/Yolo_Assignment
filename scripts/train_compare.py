"""
Trains YOLO variants with identical settings and compares them on the held-out val split.

Runs (all: same data, epochs, imgsz, batch, seed, optimizer defaults):
  baseline      yolov8n.yaml       plain YOLOv8n, random init           (baseline)
  dino          yolov8n-dino.yaml  YOLOv8n + frozen DINOv2 global hint, random init   (the reference hybrid)
  pretrained    yolov8n.pt         YOLOv8n starting from COCO-pretrained weights, fine-tuned   (reference)
  zeroshot      yolov8n.pt evaluated as-is on the val split, no training (reference)
  dino_pretrained  hybrid whose YOLO layers start from yolov8n.pt (layer indices remapped), DINOv2 frozen (extra)

Usage:  python scripts/train_compare.py --runs baseline dino --epochs 20 --imgsz 416
Each run writes outputs/runs/<name>.json; the table is rebuilt from whatever runs exist.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "yolo_subset" / "coco_subset.yaml"
OUT = ROOT / "outputs"
RUN_DIR = OUT / "runs"
LABELS = {
    "zeroshot": "YOLOv8n pretrained, no fine-tuning (reference)",
    "baseline": "YOLOv8n (from scratch)",
    "dino": "YOLOv8n + DINOv2 (from scratch)",
    "pretrained": "YOLOv8n pretrained (fine-tuned)",
    "dino_pretrained": "YOLOv8n-pretrained + DINOv2 (fine-tuned)",
}


def remap_pretrained_into_hybrid(hybrid: YOLO, pt_path: str = "yolov8n.pt") -> Path:
    """Copy yolov8n.pt weights into the hybrid, fixing layer numbers.

    The hybrid inserts ConvDummy (layer 0) and DINOv2+Concat (layers 11-12), so old layer i becomes
    i+1 for the backbone (i<=9) and i+3 for the head (i>=10). Ultralytics' own .load() matches by layer *name*,
    which would copy weights into the wrong layers, so we rename the keys first. Layers whose shapes changed
    (the head blocks that now see 256 extra DINOv2 channels) keep their random init.
    """
    old = YOLO(pt_path).model.float().state_dict()
    new = hybrid.model.state_dict()
    copied = skipped = 0
    for k, v in old.items():
        _, idx, rest = k.split(".", 2)
        idx = int(idx)
        nk = f"model.{idx + 1 if idx <= 9 else idx + 3}.{rest}"
        if nk in new and new[nk].shape == v.shape:
            new[nk] = v
            copied += 1
        else:
            skipped += 1
    hybrid.model.load_state_dict(new)
    print(f"[dino_pretrained] copied {copied} tensors from {pt_path}, left {skipped} at random init (shape changed)")
    path = RUN_DIR / "dino_pretrained_init.pt"
    torch.save({"model": hybrid.model, "train_args": {}}, path)
    return path


def summarise(name, m, extra=None):
    p, r = float(m.mp), float(m.mr)
    return {"name": name, "label": LABELS[name], "map50": float(m.map50), "map50_95": float(m.map), "precision": p,
            "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, **(extra or {})}


def eval_zeroshot(args) -> dict:
    m = YOLO("yolov8n.pt").val(data=str(DATA_YAML), imgsz=args.imgsz, batch=args.batch, device=args.device,
                               workers=args.workers, plots=False, verbose=False).box
    res = summarise("zeroshot", m, {"settings": {"imgsz": args.imgsz, "note": "no training"}})
    (RUN_DIR / "zeroshot.json").write_text(json.dumps(res, indent=2))
    return res


def train_one(name: str, args) -> dict:
    src = {"baseline": "yolov8n.yaml", "dino": "yolov8n-dino.yaml", "pretrained": "yolov8n.pt",
           "dino_pretrained": "yolov8n-dino.yaml"}[name]
    model = YOLO(src)
    extra = {}
    if name == "dino_pretrained":
        extra["pretrained"] = str(remap_pretrained_into_hybrid(model))
    n_params = sum(p.numel() for p in model.model.parameters())
    # Only DINOv2's 22M weights are frozen; everything else trains. (requires_grad on a loaded .pt is False until
    # training starts, so count explicitly instead of reading it.)
    n_frozen = sum(p.numel() for n_, p in model.model.named_parameters() if ".model.11.model." in f".{n_}") if "dino" in name else 0
    n_trainable = n_params - n_frozen

    t0 = time.time()
    model.train(data=str(DATA_YAML), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=args.seed,
                device=args.device, workers=args.workers, project=str(RUN_DIR / "train"), name=name, exist_ok=True,
                plots=True, fraction=args.fraction, patience=0, verbose=False, **extra)
    minutes = (time.time() - t0) / 60

    # Re-evaluate the best checkpoint on the held-out val images (conf=0.001, the standard mAP protocol)
    best = YOLO(RUN_DIR / "train" / name / "weights" / "best.pt")
    m = best.val(data=str(DATA_YAML), imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
                 plots=False, verbose=False).box
    res = summarise(name, m, {"params_total": n_params,
           "params_trainable": n_trainable, "train_minutes": round(minutes, 1),
           "settings": {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "seed": args.seed,
                        "fraction": args.fraction, "device": args.device}})
    (RUN_DIR / f"{name}.json").write_text(json.dumps(res, indent=2))
    return res


def write_table():
    rows = [json.loads(p.read_text()) for n in LABELS if (p := RUN_DIR / f"{n}.json").exists()]
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))
    lines = ["| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Params (trainable) | Train min |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        size = f"{r['params_total'] / 1e6:.2f}M ({r['params_trainable'] / 1e6:.2f}M)" if "params_total" in r else "3.16M (none)"
        lines.append(f"| {r['label']} | {r['map50']:.3f} | {r['map50_95']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{r['f1']:.3f} | {size} | {r.get('train_minutes', '-')} |")
    s = next((r["settings"] for r in rows if r["name"] == "baseline"), {})
    lines += ["", f"Val split: 120 held-out COCO val2017 images. Settings for every run: {s}.",
              "Precision/Recall = class-mean at the confidence that maximises F1 (Ultralytics); F1 = 2PR/(P+R)."]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["baseline", "dino", "pretrained"], choices=list(LABELS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train images (smoke tests only)")
    a = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for n in a.runs:
        print(f"\n===== {n} =====")
        eval_zeroshot(a) if n == "zeroshot" else train_one(n, a)
    write_table()
