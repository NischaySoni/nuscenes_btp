# NuScenes-QA: Trimodal Fusion & Smart Ensembling

This repository provides an advanced implementation for Visual Question Answering on the **nuScenes dataset**. We push beyond standard baseline models by implementing a **Trimodal Fusion Architecture** and a **Smart Ensembling Pipeline**, achieving **58.86% accuracy** (Oracle Ceiling: 72.20%) on open-vocabulary autonomous driving QA tasks.

## 📊 Results Summary

| Strategy | Overall | Exist | Count | Object | Status | Comparison |
|---|---|---|---|---|---|---|
| **Grid Search (5-model)** | **58.86%** | 84.43 | 22.74 | 50.09 | 57.41 | 69.43 |
| Conf-Gated (0.7) | 58.68% | 84.25 | 22.74 | 49.98 | 56.90 | 69.23 |
| Agreement Router | 58.47% | 84.24 | 22.31 | 49.55 | 56.92 | 69.04 |
| Best Single Model (V15) | 57.69% | 83.90 | 21.95 | 48.47 | 55.51 | 67.87 |
| Oracle Ceiling | **72.20%** | 89.25 | 43.01 | 67.38 | 74.65 | 81.22 |

### Model Library

| Model | Config | Best Overall | Key Strength |
|---|---|---|---|
| V14 | `mcan_trimodal_v14_bert_ft` | 57.37% | Comparison (68.83%) |
| V15 | `mcan_trimodal_v15_bert_mh` | 57.69% | Count (21.95%), Multi-head |
| V16 | `mcan_trimodal_v16_bert_deep` | 57.58% | Status (56.23%) |
| V18 | `mcan_trimodal_v18_bert_base` | 57.64% | BERT-base encoder |
| V24 | `mcan_trimodal_v24_yoloworld` | 57.72% | Object (49.02%), YOLOWorld features |

---

## 🚀 Key Innovations

### 1. Trimodal Fusion Architecture (MCAN-based)
- Fuses **Language** (BERT-base / DistilBERT), **2D Visual** (YOLO/CLIP detection features), **3D Spatial** (LiDAR BEV + Radar), and **Semantic Map Priors**.
- Deeply fine-tuned BERT-base (4 unfrozen layers) replaces basic DistilBERT for superior language understanding.
- Multi-head classification routes answers through type-specific heads (`exist`, `count`, `object`, `status`, `comparison`).

### 2. YOLOWorld + CLIP RadarXFormer Features (V3)
- Open-vocabulary detection using **YOLOWorld** captures all **23 NuScenes categories** including rare classes (trailers, construction vehicles, ambulances, police cars, barriers, traffic cones) that standard COCO-trained YOLO completely misses.
- 48-dim features per object: **16-dim structured priors** (position, velocity, size, heading, category, attribute, confidence) + **32-dim PCA-compressed CLIP visual embeddings**.
- Attention-weighted multi-radar aggregation inspired by RadarXFormer's deformable cross-attention.
- Multi-view triangulation across 6 cameras with radar-primary depth refinement.

### 3. Smart Ensembling Pipeline (V1–V5)
Five progressively more sophisticated ensemble evaluation scripts:

| Script | Strategies |
|---|---|
| `ensemble_eval_v2.py` | Basic weighted averaging with auto key remapping |
| `ensemble_eval_v3.py` | Q-Type routed, majority vote, best-model, top-2 |
| `ensemble_eval_v4.py` | Oracle analysis, temperature sweep, confidence-weighted, hybrid, grid search |
| `ensemble_eval_v5.py` | Agreement router, confidence-gated, margin router, rank fusion (Borda), Dirichlet grid search for 4+ models |

---

## 🛠️ Feature Extraction Pipeline

Pre-compute multimodal features before training.

### 1. RadarXFormer Features V3 (YOLOWorld — Recommended)
```bash
# Fit PCA on a subset (~5 min)
CUDA_VISIBLE_DEVICES=0 python precompute_radarxformer_features_v3.py --mode fit-pca

# Extract all features (~3 hours)
CUDA_VISIBLE_DEVICES=0 python precompute_radarxformer_features_v3.py --mode extract
```

### 2. RadarXFormer Features V2 (COCO YOLOv8m — Legacy)
```bash
CUDA_VISIBLE_DEVICES=0 python precompute_radarxformer_features.py --mode all
```

### 3. Annotation & Detected Features (Baselines)
```bash
python precompute_annotation_features.py --data-root /path/to/nuscenes
python precompute_detected_features.py --data-root /path/to/nuscenes
```

---

## 🧠 Training

Train trimodal fusion models. All configs are in `configs/` and registered in `run.py`.

```bash
# V24: BERT-base + YOLOWorld features (best individual model)
python run.py --RUN train --MODEL mcan_trimodal_v24_yoloworld --GPU 0 --VERSION trimodal_yoloworld_v1

# V15: DistilBERT + Multi-head (best count accuracy)
python run.py --RUN train --MODEL mcan_trimodal_v15_bert_mh --GPU 0 --VERSION trimodal_bert_mh_v1

# V18: BERT-base baseline
python run.py --RUN train --MODEL mcan_trimodal_v18_bert_base --GPU 0 --VERSION trimodal_bert_base_v1
```

---

## 🔮 Smart Ensemble Evaluation

Run `ensemble_eval_v5.py` with the best 5 models for maximum accuracy. Automatically evaluates Oracle Ceiling, Agreement Router, Confidence-Gated, Margin, Rank Fusion, and Grid Search.

```bash
python ensemble_eval_v5.py \
    --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
             mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
             mcan_trimodal_v16_bert_deep:trimodal_bert_deep_v1:16 \
             mcan_trimodal_v18_bert_base:trimodal_bert_base_v1:15 \
             mcan_trimodal_v24_yoloworld:trimodal_yoloworld_v1:16 \
    --gpu 0
```

### Grid Search Optimal Weights (5-model, per question type)

| Type | V14 | V15 | V16 | V18 | V24 | Accuracy |
|---|---|---|---|---|---|---|
| exist | 0.01 | 0.14 | 0.34 | 0.07 | **0.44** | 84.43% |
| count | 0.03 | **0.50** | 0.33 | 0.00 | 0.14 | 22.74% |
| object | 0.07 | 0.23 | 0.01 | 0.13 | **0.57** | 50.09% |
| status | 0.25 | 0.11 | **0.28** | **0.35** | 0.01 | 57.41% |
| comparison | 0.26 | **0.36** | 0.32 | 0.04 | 0.02 | 69.43% |

---

## 📁 Project Structure

```
├── configs/                          # Model configuration YAML files (V1–V24)
├── src/
│   ├── models/mcan/                  # MCAN backbone, trimodal fusion, multi-head classifier
│   ├── datasets/nuscenes_qa.py       # Dataset loader with trimodal feature loading
│   ├── execution/                    # Train/test engines with checkpoint key remapping
│   └── configs/base_cfgs.py          # Feature paths, hyperparameter defaults
├── precompute_radarxformer_features_v3.py   # YOLOWorld + CLIP feature extraction
├── precompute_radarxformer_features.py      # Legacy COCO YOLO extraction
├── precompute_annotation_features.py        # Ground-truth annotation features
├── ensemble_eval_v2.py               # Basic ensemble with key remapping
├── ensemble_eval_v3.py               # Q-Type routed ensemble
├── ensemble_eval_v4.py               # Oracle + temperature sweep + grid search
├── ensemble_eval_v5.py               # Full advanced ensemble (recommended)
├── run.py                            # Main training/evaluation entry point
└── log/                              # Training logs for all model versions
```

---

## 📚 References

- **NuScenes Dataset:** [Paper](https://arxiv.org/pdf/2305.14836.pdf) | [Download](https://www.nuscenes.org/download)
- **MCAN (Deep Modular Co-Attention Networks):** [Paper](https://arxiv.org/abs/1906.10770)
- **YOLOWorld:** Open-vocabulary object detection
- **CLIP (ViT-B/32):** Visual feature extraction for detection crops
