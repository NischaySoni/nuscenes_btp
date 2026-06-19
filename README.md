# NuScenes-QA: Trimodal Fusion & Smart Ensembling 

This repository provides an advanced implementation for Visual Question Answering on the **nuScenes dataset**. We push beyond standard baseline models by implementing a **Trimodal Fusion Architecture** and a **Smart Ensembling Pipeline**, achieving state-of-the-art results near 60% accuracy on open-vocabulary autonomous driving QA tasks.

## 🚀 Key Features & Innovations

1. **Trimodal Fusion Architecture (MCAN-based)**
   - Fuses Language (BERT-base), 2D Visual/Detection features, 3D Lidar (BEV), and Semantic Map Priors.
   - Replaced basic DistilBERT with deeply fine-tuned BERT-base models for superior language understanding.
   - Advanced multi-head routing for distinct question types (`exist`, `count`, `object`, `status`, `comparison`).

2. **YOLOWorld + CLIP RadarXFormer Features (V3)**
   - Open-vocabulary object detection using `YOLOWorld`, capturing all 23 NuScenes categories (including rare classes like trailers, construction vehicles, ambulances, and police cars which standard COCO misses).
   - Generates 48-dim features (16-dim structured priors + 32-dim PCA-compressed CLIP visual features).
   - Incorporates attention-weighted multi-radar aggregation.

3. **Smart Ensembling (V1-V5)**
   - Achieved a theoretical **Oracle Ceiling of 67.48%** using structurally diverse models.
   - Includes multiple intelligent routing strategies to close the ensemble gap:
     - **Agreement-Aware Router**: Consensus voting for majority, confidence-blend for disagreement.
     - **Q-Type Routed**: Softmax temperature weighting based on per-category model strengths.
     - **Confidence Gated**: Trusting highly confident individual models.
     - **Margin-Based & Rank Fusion (Borda)**.
     - **Dirichlet Grid Search**: Brute-force optimal weight combinations for 4+ model ensembles.

---

## 🛠️ Feature Extraction Pipeline

Before training, pre-compute the multimodal features.

### 1. RadarXFormer Features (V3 - YOLOWorld)
Extracts detection boxes, depths, velocities, and CLIP features.
```bash
# Fit PCA on a subset (Fast)
CUDA_VISIBLE_DEVICES=0 python precompute_radarxformer_features_v3.py --mode fit-pca

# Extract all features (~3 hours)
CUDA_VISIBLE_DEVICES=0 python precompute_radarxformer_features_v3.py --mode extract
```

### 2. Annotation & Detected Features (Baselines)
```bash
python precompute_annotation_features.py --data-root /path/to/nuscenes
python precompute_detected_features.py --data-root /path/to/nuscenes
```

---

## 🧠 Training Models

Train the trimodal fusion models using different configurations. Best architectures are registered in `run.py`.

```bash
# Train the ultimate V24 (BERT-base + YOLOWorld features)
python run.py --RUN train --MODEL mcan_trimodal_v24_yoloworld --GPU 0 --VERSION trimodal_yoloworld_v1

# Train V15 (Multi-head baseline)
python run.py --RUN train --MODEL mcan_trimodal_v15_bert_mh --GPU 0 --VERSION trimodal_bert_mh_v1
```

---

## 🔮 Smart Ensembling (Evaluation)

To achieve maximum accuracy, run the `ensemble_eval_v5.py` script. It automatically evaluates the Oracle Ceiling, Confidence, QType-Routed, and Grid Search strategies.

```bash
python ensemble_eval_v5.py \
    --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
             mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
             mcan_trimodal_v16_bert_deep:trimodal_bert_deep_v1:16 \
             mcan_trimodal_v18_bert_base:trimodal_bert_base_v1:15 \
             mcan_trimodal_v24_yoloworld:trimodal_yoloworld_v1:16 \
    --gpu 0
```

---

## 📊 Dataset Reference
The NuScenes dataset is a benchmark for autonomous driving tasks. 
- **Paper:** [nuScenes Paper (PDF)](https://arxiv.org/pdf/2305.14836.pdf)
- **Download:** [nuScenes Official Website](https://www.nuscenes.org/download)
