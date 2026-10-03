| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Params (trainable) | Train min |
|---|---|---|---|---|---|---|---|
| YOLOv8n pretrained, no fine-tuning (reference) | 0.480 | 0.351 | 0.617 | 0.427 | 0.505 | 3.16M (none) | - |
| YOLOv8n (from scratch) | 0.001 | 0.000 | 0.001 | 0.014 | 0.002 | 3.16M (3.16M) | 17.1 |
| YOLOv8n + DINOv2 (from scratch) | 0.002 | 0.000 | 0.002 | 0.017 | 0.003 | 25.41M (3.35M) | 46.5 |
| YOLOv8n pretrained (fine-tuned) | 0.469 | 0.341 | 0.629 | 0.425 | 0.507 | 3.16M (3.16M) | 22.9 |
| YOLOv8n-pretrained + DINOv2 (fine-tuned) | 0.153 | 0.092 | 0.357 | 0.162 | 0.223 | 25.41M (3.35M) | 24.2 |

Val split: 120 held-out COCO val2017 images. Settings for every run: {'epochs': 20, 'imgsz': 416, 'batch': 16, 'seed': 0, 'fraction': 1.0, 'device': 'cpu'}.
Precision/Recall = class-mean at the confidence that maximises F1 (Ultralytics); F1 = 2PR/(P+R).
