# NuScenes-QA: Trimodal Fusion & Smart Ensembling

This repository provides an advanced implementation for Visual Question Answering on the **nuScenes dataset**. We push beyond standard baseline models by implementing a **Trimodal Fusion Architecture** and a **Smart Ensembling Pipeline**, achieving **60.44% accuracy** (Oracle Ceiling: 79.18%) on open-vocabulary autonomous driving QA tasks.

## 📊 Results Summary

### 🏆 Best Ensemble: 60.44% (13-Model Variable Bins)

| Strategy | Overall | Exist | Count | Object | Status | Comparison |
|---|---|---|---|---|---|---|
| **Variable Bins (13-model)** | **60.44%** | 84.73 | 25.34 | 52.53 | 59.50 | 70.60 |
| ConfBins-10 (13-model) | 60.26% | 84.61 | 25.17 | 52.26 | 59.17 | 70.44 |
| ConfBins-8 (13-model) | 60.12% | 84.61 | 24.73 | 52.12 | 59.17 | 70.33 |
| ConfBins-7 (13-model) | 60.10% | 84.60 | 24.81 | 52.14 | 58.95 | 70.26 |
| Scipy Optimized (9-model) | 59.27% | 84.49 | 23.16 | 50.91 | 57.96 | 69.81 |
| Grid Search (8-model) | 59.12% | 84.43 | 22.91 | 50.73 | 57.69 | 69.78 |
| Best Single Model (V29) | 57.78% | 84.16 | 21.28 | 48.22 | 55.76 | 68.88 |
| Oracle Ceiling (13-model) | **79.18%** | — | — | — | — | — |

### Accuracy Progression

```
57.78% ──► 58.86% ──► 59.12% ──► 59.27% ──► 59.88% ──► 60.44%
single    5-model    8-model    scipy       conf-bins   variable
model     grid       grid       9-model     5×13        bins 13
```

### Model Library (13 Models)

| Model | Config | Overall | Key Differentiator |
|---|---|---|---|
| V14 | `mcan_trimodal_v14_bert_ft` | 57.37% | BERT-4L, V2 features |
| V15 | `mcan_trimodal_v15_bert_mh` | 57.69% | DistilBERT-2L, V2 features |
| V16 | `mcan_trimodal_v16_bert_deep` | 57.58% | BERT-6L, V2 features |
| V18 | `mcan_trimodal_v18_bert_base` | 57.64% | BERT-4L, V2 features |
| V24 | `mcan_trimodal_v24_yoloworld` | 57.60% | BERT-4L, **V3 YOLOWorld** features |
| V25 | `mcan_trimodal_v25_yoloworld_seed2` | 57.65% | V24 config, seed=7777 |
| V26 | `mcan_trimodal_v26_deep_yoloworld` | 57.44% | BERT-6L, V3 features, seed=9999 |
| V27 | `mcan_trimodal_v27_distilbert_v3` | 57.50% | **DistilBERT + V3 + single-head** |
| V29 | `mcan_trimodal_v29_v2seed` | 57.78% | BERT-4L, V2 features, **seed=5555** |
| V30 | `mcan_trimodal_v30_countfocus` | 55.83% | Count-focused (high count loss weight) |
| V31 | `mcan_trimodal_v31_shallow_wide` | 57.57% | **4-layer MCAN, 4096 FFN, 2048 proj** |
| V32 | `mcan_trimodal_v32_large` | 57.56% | **768 hidden (matches BERT native)** |
| V33 | `mcan_trimodal_v33_extended` | 57.35% | 25 epochs, late LR decay |

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

### 3. Smart Ensembling Pipeline (V1–V11)
Progressively more sophisticated ensemble strategies:

| Script | Strategy | Best Accuracy |
|---|---|---|
| `ensemble_eval_v5.py` | Dirichlet grid search, agreement routing, confidence gating | 59.12% |
| `ensemble_eval_v7.py` | Scipy differential evolution (global optimizer) | 59.27% |
| `ensemble_eval_v9.py` | Confidence-binned per-type optimization | 59.46% |
| `ensemble_eval_v10.py` | Multi-bin sweep (3–5 bins) + subtype × confidence | 59.88% |
| **`ensemble_eval_v11.py`** | **Per-type variable bin count (3–12 sweep)** | **60.44%** |

### 4. Per-Type Variable Bins (Key to 60%+)
The breakthrough insight: different question types need different confidence granularity.
- **exist** (84.73%): Highly confident — 12 bins for fine-tuning edge cases
- **count** (25.34%): Huge variance — 12 bins to separate easy/hard count patterns
- **object** (52.53%): 12 bins for better hard-sample routing
- **status** (59.50%): 12 bins optimized independently
- **comparison** (70.60%): 12 bins with high-confidence regions near 100%

Each bin gets its own scipy-optimized 13-dimensional weight vector via differential evolution.

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

---

## 🧠 Training

Train trimodal fusion models. All configs are in `configs/` and registered in `run.py`.

```bash
# V29: Best individual model (BERT-base + V2 features + seed=5555)
python run.py --RUN train --MODEL mcan_trimodal_v29_v2seed --GPU 0 \
    --VERSION trimodal_v2seed_v1 --SEED 5555

# V24: BERT-base + YOLOWorld V3 features
python run.py --RUN train --MODEL mcan_trimodal_v24_yoloworld --GPU 0 \
    --VERSION trimodal_yoloworld_v1

# V32: Large model (768-dim hidden matching BERT native)
python run.py --RUN train --MODEL mcan_trimodal_v32_large --GPU 0 \
    --VERSION trimodal_large_v1
```

---

## 🔮 Smart Ensemble Evaluation

Run `ensemble_eval_v11.py` with all 13 models for maximum accuracy:

```bash
python ensemble_eval_v11.py \
    --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
             mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
             mcan_trimodal_v16_bert_deep:trimodal_bert_deep_v1:16 \
             mcan_trimodal_v18_bert_base:trimodal_bert_base_v1:15 \
             mcan_trimodal_v24_yoloworld:trimodal_yoloworld_v1:16 \
             mcan_trimodal_v25_yoloworld_seed2:trimodal_yoloworld_v2:16 \
             mcan_trimodal_v26_deep_yoloworld:trimodal_deep_yoloworld_v1:16 \
             mcan_trimodal_v27_distilbert_v3:trimodal_distilbert_v3_v1:16 \
             mcan_trimodal_v29_v2seed:trimodal_v2seed_v1:16 \
             mcan_trimodal_v30_countfocus:trimodal_countfocus_v1:10 \
             mcan_trimodal_v31_shallow_wide:trimodal_shallow_wide_v1:12 \
             mcan_trimodal_v32_large:trimodal_large_v1:12 \
             mcan_trimodal_v33_extended:trimodal_extended_v1:16 \
    --gpu 0
```

---

## 📁 Project Structure

```
├── configs/                          # Model configs (V1–V34)
├── src/
│   ├── models/mcan/                  # MCAN backbone, trimodal fusion, multi-head classifier
│   ├── datasets/nuscenes_qa.py       # Dataset loader with trimodal feature loading
│   ├── execution/                    # Train/test engines with checkpoint key remapping
│   └── configs/base_cfgs.py          # Feature paths, hyperparameter defaults
├── precompute_radarxformer_features_v3.py   # YOLOWorld + CLIP feature extraction
├── precompute_radarxformer_features.py      # Legacy COCO YOLO extraction
├── ensemble_eval_v11.py              # 🏆 Best ensemble (variable bins)
├── ensemble_eval_v10.py              # Multi-bin + subtype × confidence
├── ensemble_eval_v9.py               # Confidence-binned optimization
├── ensemble_eval_v7.py               # Scipy differential evolution
├── ensemble_eval_v5.py               # Grid search + agreement routing
├── run.py                            # Main training/evaluation entry point
└── log/                              # Training logs for all model versions
```

---

## 📚 References

- **NuScenes Dataset:** [Paper](https://arxiv.org/pdf/2305.14836.pdf) | [Download](https://www.nuscenes.org/download)
- **MCAN (Deep Modular Co-Attention Networks):** [Paper](https://arxiv.org/abs/1906.10770)
- **YOLOWorld:** Open-vocabulary object detection
- **CLIP (ViT-B/32):** Visual feature extraction for detection crops
