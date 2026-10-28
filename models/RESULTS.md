# Trained model results

One row per run. Never overwrite — append. A number that is not in this
table with the command that produced it does not go in the report.

| date | model | epochs | batch | lr | seed | val acc | val AUC | test acc | test AUC | wall | device | ckpt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-28 14:06 | PCamCNN FP32 | 30 | 256 | 0.001 | 0 | 0.8878 | 0.9604 | 0.8115 | 0.9134 | 42 min | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `models\fp32_best.pt` |

<!-- command: python.exe phase1_software\train.py --epochs 30 --seed 0 --out models\fp32_best.pt -->
| 2026-10-28 18:31 | PCamCNN FP32 | 10 | 256 | 0.0001 | 0 | 0.8831 | 0.9556 | 0.8311 | 0.9354 | 14 min | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `models/qat_int8_best.pt` |

<!-- command: python.exe phase1_software/qat.py --init models/fp32_best.pt --epochs 10 -->
