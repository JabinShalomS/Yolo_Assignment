"""Grouped bar chart of mAP50 / mAP50-95 per model, from outputs/comparison.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "outputs" / "comparison.json").read_text())

labels = {
    "zeroshot": "YOLOv8n pretrained\n(no fine-tuning)",
    "baseline": "YOLOv8n\n(from scratch)",
    "dino": "YOLOv8n + DINOv2\n(from scratch)",
    "pretrained": "YOLOv8n pretrained\n(fine-tuned)",
}
order = ["baseline", "dino", "pretrained", "zeroshot"]
by = {r["name"]: r for r in rows}
order += [r["name"] for r in rows if r["name"] not in order]
for n in order:
    labels.setdefault(n, by[n]["label"].replace(" (", "\n("))

import numpy as np
x = np.arange(len(order))
w = 0.36
m50 = [by[n]["map50"] for n in order]
m95 = [by[n]["map50_95"] for n in order]

fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150, facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
b1 = ax.bar(x - w/2 - 0.01, m50, w, color="#2a78d6", label="mAP50")
b2 = ax.bar(x + w/2 + 0.01, m95, w, color="#eb6834", label="mAP50-95")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=10, color="#0b0b0b")
ax.set_xticks(x)
ax.set_xticklabels([labels[n] for n in order], fontsize=10, color="#0b0b0b")
ax.set_ylim(0, 0.58)
ax.set_xlabel("Model (trained models: 20 epochs, 416 px, CPU, 480 COCO images)", fontsize=11, color="#52514e", labelpad=10)
ax.set_ylabel("Mean average precision (higher is better)", fontsize=11, color="#52514e")
ax.set_title("Effect of pre-training on COCO object detection (120 held-out images, 80 classes)",
             fontsize=13, color="#0b0b0b", loc="left", pad=14)
ax.yaxis.grid(True, color="#e5e4e0", linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#b9b8b2")
ax.tick_params(colors="#52514e")
ax.legend(frameon=False, loc="upper left", fontsize=10)
fig.tight_layout()
out = ROOT / "outputs" / "results_chart.png"
fig.savefig(out, facecolor=fig.get_facecolor())
print(out)
