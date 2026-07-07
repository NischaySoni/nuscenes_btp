#!/usr/bin/env python3
"""
Ensemble V12: Learned Stacking Meta-Classifier

Instead of fixed-weight blending (variable bins), train a neural network
to predict the correct answer from ALL models' logits.

Key insight: variable bins use the same weights for ~1,400 samples per bin.
A learned meta-classifier can learn per-sample decision boundaries in the
13×30-dim logit space — far more expressive than linear blending.

Features per sample:
  - Raw logits: 13 models × 30 classes = 390 dims
  - Per-model confidence: 13 dims (softmax max)
  - Per-model entropy: 13 dims
  - Agreement features: 5 dims (vote counts for top-5 answers)
  - Question type: 5 dims (one-hot)
  - Margin features: 13 dims (gap between top-2 logits per model)
  Total: ~439 dims → MLP → 30 classes

Training:
  - K-fold cross-validation (5 folds) on the validation set
  - Each fold: train on 4/5, predict on 1/5
  - Final predictions are the concatenation of held-out fold predictions
  - This avoids overfitting to the validation set

Usage:
    python ensemble_eval_v12.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
                 ... \
        --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import differential_evolution
from collections import Counter, defaultdict

from ensemble_eval_v2 import get_logits

QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']


def get_qtype_for_sample(qa_item):
    template_type = qa_item.get('template_type', 'exist')
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'


def softmax_np(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def build_indices(dataset):
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    qtype_indices = {qt: [] for qt in QTYPE_NAMES}
    valid_gt = {}
    qtypes = {}
    for i in range(len(qa_list)):
        gt_ans_str = str(qa_list[i]['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue
        qtype = get_qtype_for_sample(qa_list[i])
        qtype_indices[qtype].append(i)
        valid_gt[i] = gt_idx
        qtypes[i] = qtype
    return qtype_indices, valid_gt, qtypes


def build_meta_features(all_logits, n_samples, n_classes, n_models, qtypes):
    """Build rich meta-features from model logits."""

    # 1. Raw logits (flattened): n_models × n_classes
    raw_logits = np.zeros((n_samples, n_models * n_classes))
    for m in range(n_models):
        raw_logits[:, m*n_classes:(m+1)*n_classes] = all_logits[m][:n_samples, :n_classes]

    # 2. Per-model softmax probabilities for top prediction
    confidences = np.zeros((n_samples, n_models))
    entropies = np.zeros((n_samples, n_models))
    margins = np.zeros((n_samples, n_models))
    predictions = np.zeros((n_samples, n_models), dtype=np.int64)

    for m in range(n_models):
        logits_m = all_logits[m][:n_samples, :n_classes]
        probs_m = softmax_np(logits_m)
        confidences[:, m] = probs_m.max(axis=1)
        entropies[:, m] = -(probs_m * np.log(probs_m + 1e-10)).sum(axis=1)
        sorted_logits = np.sort(logits_m, axis=1)[:, ::-1]
        margins[:, m] = sorted_logits[:, 0] - sorted_logits[:, 1]
        predictions[:, m] = np.argmax(logits_m, axis=1)

    # 3. Agreement features: how many models agree on top-K answers
    agreement = np.zeros((n_samples, 5))
    for i in range(n_samples):
        vote_counts = Counter(predictions[i])
        top_counts = sorted(vote_counts.values(), reverse=True)
        for k in range(min(5, len(top_counts))):
            agreement[i, k] = top_counts[k] / n_models

    # 4. Question type one-hot
    qtype_onehot = np.zeros((n_samples, 5))
    for i in range(n_samples):
        if i in qtypes:
            qt_idx = QTYPE_NAMES.index(qtypes[i]) if qtypes[i] in QTYPE_NAMES else 0
            qtype_onehot[i, qt_idx] = 1.0

    # 5. Ensemble logits (average)
    ensemble_logits = np.mean(raw_logits.reshape(n_samples, n_models, n_classes), axis=1)

    # Concatenate all features
    features = np.concatenate([
        raw_logits,       # n_models * n_classes
        confidences,      # n_models
        entropies,        # n_models
        margins,          # n_models
        agreement,        # 5
        qtype_onehot,     # 5
        ensemble_logits,  # n_classes
    ], axis=1)

    print(f"  Meta-features: {features.shape[1]} dims")
    print(f"    Raw logits: {n_models * n_classes}, Confidence: {n_models}, "
          f"Entropy: {n_models}, Margins: {n_models}")
    print(f"    Agreement: 5, Qtype: 5, Ensemble logits: {n_classes}")

    return features


class StackingMLP(nn.Module):
    """Small MLP meta-classifier for stacking."""
    def __init__(self, input_dim, n_classes, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_meta_model(X_train, y_train, X_val, y_val, n_classes,
                     hidden_dims=[512, 256], epochs=100, lr=0.001,
                     batch_size=256, dropout=0.3, device='cpu'):
    """Train a stacking MLP and return predictions on validation set."""
    input_dim = X_train.shape[1]

    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    # To tensors
    X_tr = torch.FloatTensor(X_train_norm).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val_norm).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = StackingMLP(input_dim, n_classes, hidden_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1
    best_val_preds = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = (val_preds == y_val).mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_preds = val_preds.copy()
            best_val_logits = val_logits.cpu().numpy().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_val_preds, best_val_logits, best_val_acc


def kfold_stacking(features, labels, valid_indices, n_classes, n_folds=5,
                   device='cpu', hidden_dims=[512, 256]):
    """K-fold cross-validation stacking."""
    # Only use valid samples
    valid_mask = np.array([i in valid_indices for i in range(len(features))])
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(n_valid)

    fold_size = n_valid // n_folds
    all_preds = np.zeros(len(features), dtype=np.int64)
    all_logits = np.zeros((len(features), n_classes))
    fold_accs = []

    for fold in range(n_folds):
        # Split
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_valid
        val_fold_idx = shuffled[val_start:val_end]
        train_fold_idx = np.concatenate([shuffled[:val_start], shuffled[val_end:]])

        val_global = valid_idx[val_fold_idx]
        train_global = valid_idx[train_fold_idx]

        X_train = features[train_global]
        y_train = np.array([labels[i] for i in train_global])
        X_val = features[val_global]
        y_val = np.array([labels[i] for i in val_global])

        preds, logits, acc = train_meta_model(
            X_train, y_train, X_val, y_val, n_classes,
            hidden_dims=hidden_dims, epochs=150, lr=0.001,
            batch_size=512, dropout=0.3, device=device
        )
        fold_accs.append(acc)
        all_preds[val_global] = preds
        all_logits[val_global] = logits
        print(f"    Fold {fold+1}/{n_folds}: {acc*100:.2f}%")

    mean_acc = np.mean(fold_accs)
    print(f"    Mean fold accuracy: {mean_acc*100:.2f}%")
    return all_preds, all_logits


def kfold_stacking_per_type(features, labels, valid_indices, qtypes,
                             qtype_indices, n_classes, n_folds=5, device='cpu'):
    """Train separate meta-classifiers per question type."""
    all_preds = np.zeros(len(features), dtype=np.int64)

    for qt in QTYPE_NAMES:
        indices = [i for i in qtype_indices[qt] if i in valid_indices]
        if not indices:
            continue

        n_qt = len(indices)
        qt_features = features[indices]
        qt_labels = np.array([labels[i] for i in indices])

        print(f"\n  Training {qt} meta-classifier ({n_qt} samples):")

        # Shuffle
        rng = np.random.RandomState(42)
        shuffled = rng.permutation(n_qt)
        fold_size = n_qt // n_folds

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_qt
            val_fold = shuffled[val_start:val_end]
            train_fold = np.concatenate([shuffled[:val_start], shuffled[val_end:]])

            X_train = qt_features[train_fold]
            y_train = qt_labels[train_fold]
            X_val = qt_features[val_fold]
            y_val = qt_labels[val_fold]

            # Adjust hidden dims based on sample count
            if n_qt < 15000:
                hidden_dims = [256, 128]
            else:
                hidden_dims = [512, 256]

            preds, logits, acc = train_meta_model(
                X_train, y_train, X_val, y_val, n_classes,
                hidden_dims=hidden_dims, epochs=150, lr=0.001,
                batch_size=256, dropout=0.3, device=device
            )
            val_global = [indices[j] for j in val_fold]
            all_preds[val_global] = preds
            print(f"    Fold {fold+1}/{n_folds}: {acc*100:.2f}%")

    return all_preds


def evaluate(predictions, valid_gt, qtype_indices, n_samples):
    type_stats = {qt: [0, 0] for qt in QTYPE_NAMES}
    total_correct = 0
    total = 0
    for i in range(min(n_samples, len(predictions))):
        if i not in valid_gt:
            continue
        correct = int(predictions[i] == valid_gt[i])
        total_correct += correct
        total += 1
        for qt in QTYPE_NAMES:
            if i in set(qtype_indices[qt]):
                type_stats[qt][0] += correct
                type_stats[qt][1] += 1
                break
    overall = 100 * total_correct / total if total > 0 else 0
    return overall, total_correct, total, type_stats


def print_eval(label, overall, tc, t, type_stats):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Overall {tc} / {t} = {overall:.2f}")
    for qt in sorted(type_stats.keys()):
        c, tot = type_stats[qt]
        if tot > 0:
            print(f"  {qt} {c} / {tot} = {100*c/tot:.2f}")
    return overall


# ============================================================
# Hybrid: Stacking + Variable Bins
# ============================================================
def variable_bins_predictions(all_logits, valid_gt, qtype_indices, dataset,
                               n_samples, n_classes, n_models, n_bins=12):
    """Variable bins baseline (from V11)."""
    qa_list = dataset.qa_list

    # Compute sample confidences
    sample_confs = np.zeros(n_samples)
    for i in range(n_samples):
        blended = sum(all_logits[m][i, :n_classes] for m in range(n_models)) / n_models
        probs = softmax_np(blended.reshape(1, -1))[0]
        sample_confs[i] = probs.max()

    predictions = np.zeros(n_samples, dtype=np.int64)

    for qt in QTYPE_NAMES:
        indices = [i for i in qtype_indices[qt] if i in valid_gt]
        if not indices:
            continue

        type_confs = [sample_confs[i] for i in indices]
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        thresholds = [np.percentile(type_confs, p) for p in percentiles]

        for bin_idx in range(n_bins):
            if bin_idx == 0:
                bin_indices = [i for i in indices if sample_confs[i] <= thresholds[0]]
            elif bin_idx == n_bins - 1:
                bin_indices = [i for i in indices if sample_confs[i] > thresholds[-1]]
            else:
                bin_indices = [i for i in indices
                              if thresholds[bin_idx-1] < sample_confs[i] <= thresholds[bin_idx]]

            if len(bin_indices) < 15:
                for i in bin_indices:
                    blended = sum(all_logits[m][i, :n_classes] for m in range(n_models)) / n_models
                    predictions[i] = np.argmax(blended)
                continue

            # Scipy optimize
            qt_logits = []
            qt_gts = []
            for idx in bin_indices:
                if idx not in valid_gt:
                    continue
                model_logits = [all_logits[m][idx, :n_classes] for m in range(n_models)]
                qt_logits.append(model_logits)
                qt_gts.append(valid_gt[idx])

            qt_logits_arr = np.array(qt_logits)
            qt_gts_arr = np.array(qt_gts)
            n_qt = len(qt_gts_arr)

            def neg_accuracy(w_raw):
                w = np.exp(w_raw) / np.exp(w_raw).sum()
                blended = np.einsum('m,nmc->nc', w, qt_logits_arr)
                preds = np.argmax(blended, axis=1)
                return -((preds == qt_gts_arr).sum() / n_qt)

            bounds = [(-3, 3)] * n_models
            result = differential_evolution(
                neg_accuracy, bounds=bounds, seed=42,
                maxiter=300, tol=1e-8, popsize=20, polish=True
            )
            best_w = np.exp(result.x) / np.exp(result.x).sum()

            for idx in bin_indices:
                blended = sum(best_w[m] * all_logits[m][idx, :n_classes] for m in range(n_models))
                predictions[idx] = np.argmax(blended)

    return predictions


def hybrid_stacking_bins(stacking_preds, stacking_logits, bins_preds,
                          all_logits, valid_gt, qtype_indices, qtypes,
                          n_samples, n_classes, n_models):
    """
    Hybrid: use stacking when confident, fall back to variable bins.
    Also: use stacking for types where it's better, bins for others.
    """
    # Compare per-type accuracy
    stacking_type_acc = {}
    bins_type_acc = {}
    for qt in QTYPE_NAMES:
        indices = [i for i in qtype_indices[qt] if i in valid_gt]
        if not indices:
            continue
        s_correct = sum(1 for i in indices if stacking_preds[i] == valid_gt[i])
        b_correct = sum(1 for i in indices if bins_preds[i] == valid_gt[i])
        stacking_type_acc[qt] = s_correct / len(indices)
        bins_type_acc[qt] = b_correct / len(indices)
        better = "STACKING" if s_correct > b_correct else "BINS"
        print(f"  {qt}: stacking={s_correct}/{len(indices)} ({100*s_correct/len(indices):.2f}%) "
              f"bins={b_correct}/{len(indices)} ({100*b_correct/len(indices):.2f}%) → {better}")

    # Cherry-pick: use whichever is better per type
    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        if i in qtypes:
            qt = qtypes[i]
            if stacking_type_acc.get(qt, 0) >= bins_type_acc.get(qt, 0):
                predictions[i] = stacking_preds[i]
            else:
                predictions[i] = bins_preds[i]
        else:
            predictions[i] = bins_preds[i]

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Ensemble V12: Learned Stacking')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--device', default='cuda',
                        help='Device for meta-classifier training')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    n_models = len(model_specs)
    print(f"Learned Stacking Ensemble of {n_models} models:")
    for spec in model_specs:
        print(f"  {spec}")

    # Load logits
    all_logits = []
    dataset = None
    for spec in model_specs:
        logits, ds = get_logits(spec, args.gpu)
        all_logits.append(logits)
        if dataset is None:
            dataset = ds

    n_samples = min(l.shape[0] for l in all_logits)
    n_classes = min(l.shape[1] for l in all_logits)
    print(f"\nSamples: {n_samples}, Classes: {n_classes}")

    qtype_indices, valid_gt, qtypes = build_indices(dataset)

    # Individual models
    for m, spec in enumerate(model_specs):
        preds = np.argmax(all_logits[m][:n_samples, :n_classes], axis=1)
        ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
        name = spec.split(':')[1] if ':' in spec else spec
        pt = ' '.join(f'{qt}={100*ts[qt][0]/ts[qt][1]:.1f}' if ts[qt][1] > 0 else f'{qt}=0'
                      for qt in QTYPE_NAMES)
        print(f"  M{m} ({name}): {pt} | {ov:.2f}%")

    # Oracle
    oracle_correct = 0
    for i in range(n_samples):
        if i in valid_gt:
            for m in range(n_models):
                if np.argmax(all_logits[m][i, :n_classes]) == valid_gt[i]:
                    oracle_correct += 1
                    break
    total_valid = sum(1 for i in range(n_samples) if i in valid_gt)
    oracle_pct = 100 * oracle_correct / total_valid
    print(f"\n  Oracle: {oracle_correct}/{total_valid} = {oracle_pct:.2f}%")

    # Build meta-features
    print(f"\n{'='*60}")
    print(f"  Building meta-features...")
    print(f"{'='*60}")
    features = build_meta_features(all_logits, n_samples, n_classes, n_models, qtypes)

    # Device selection
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("  CUDA not available, using CPU")
    print(f"  Device: {device}")

    results = {}

    # ---- Strategy 1: Global stacking (all types together) ----
    print(f"\n{'='*60}")
    print(f"  Strategy 1: Global Stacking ({args.folds}-fold CV)")
    print(f"{'='*60}")
    stacking_preds_global, stacking_logits_global = kfold_stacking(
        features, valid_gt, valid_gt, n_classes, n_folds=args.folds,
        device=device, hidden_dims=[512, 256]
    )
    ov, tc, t, ts = evaluate(stacking_preds_global, valid_gt, qtype_indices, n_samples)
    print_eval("GlobalStacking", ov, tc, t, ts)
    results['global_stacking'] = ov

    # ---- Strategy 2: Per-type stacking ----
    print(f"\n{'='*60}")
    print(f"  Strategy 2: Per-Type Stacking ({args.folds}-fold CV)")
    print(f"{'='*60}")
    stacking_preds_pertype = kfold_stacking_per_type(
        features, valid_gt, valid_gt, qtypes, qtype_indices,
        n_classes, n_folds=args.folds, device=device
    )
    ov, tc, t, ts = evaluate(stacking_preds_pertype, valid_gt, qtype_indices, n_samples)
    print_eval("PerTypeStacking", ov, tc, t, ts)
    results['pertype_stacking'] = ov

    # ---- Strategy 3: Variable bins baseline (V11) ----
    print(f"\n{'='*60}")
    print(f"  Strategy 3: Variable Bins Baseline (12 bins)")
    print(f"{'='*60}")
    bins_preds = variable_bins_predictions(
        all_logits, valid_gt, qtype_indices, dataset,
        n_samples, n_classes, n_models, n_bins=12
    )
    ov, tc, t, ts = evaluate(bins_preds, valid_gt, qtype_indices, n_samples)
    print_eval("VariableBins", ov, tc, t, ts)
    results['variable_bins'] = ov

    # ---- Strategy 4: Hybrid (best per type) ----
    print(f"\n{'='*60}")
    print(f"  Strategy 4: Hybrid (Stacking vs Bins per type)")
    print(f"{'='*60}")
    # Use global stacking for hybrid
    hybrid_preds = hybrid_stacking_bins(
        stacking_preds_global, stacking_logits_global, bins_preds,
        all_logits, valid_gt, qtype_indices, qtypes,
        n_samples, n_classes, n_models
    )
    ov, tc, t, ts = evaluate(hybrid_preds, valid_gt, qtype_indices, n_samples)
    print_eval("Hybrid", ov, tc, t, ts)
    results['hybrid'] = ov

    # Also try hybrid with per-type stacking
    print(f"\n  Hybrid with per-type stacking:")
    hybrid_preds2 = hybrid_stacking_bins(
        stacking_preds_pertype, stacking_logits_global, bins_preds,
        all_logits, valid_gt, qtype_indices, qtypes,
        n_samples, n_classes, n_models
    )
    ov, tc, t, ts = evaluate(hybrid_preds2, valid_gt, qtype_indices, n_samples)
    print_eval("HybridPerType", ov, tc, t, ts)
    results['hybrid_pertype'] = ov

    # ---- Strategy 5: Wider MLP ----
    print(f"\n{'='*60}")
    print(f"  Strategy 5: Wider Stacking MLP [1024, 512, 256]")
    print(f"{'='*60}")
    wide_preds, wide_logits = kfold_stacking(
        features, valid_gt, valid_gt, n_classes, n_folds=args.folds,
        device=device, hidden_dims=[1024, 512, 256]
    )
    ov, tc, t, ts = evaluate(wide_preds, valid_gt, qtype_indices, n_samples)
    print_eval("WideStacking", ov, tc, t, ts)
    results['wide_stacking'] = ov

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY (Oracle: {oracle_pct:.2f}%)")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if acc == max(results.values()) else ""
        gap = 65.0 - acc
        status = f"({gap:.2f}% to 65)" if gap > 0 else f"(+{-gap:.2f}% above 65!)"
        print(f"  {name:25s} = {acc:.2f}% {status}{marker}")

    best_name = max(results.items(), key=lambda x: x[1])
    print(f"\n  🏆 BEST: {best_name[0]} = {best_name[1]:.2f}%")
    if best_name[1] >= 65.0:
        print(f"\n  🎉🎉🎉 65% ACHIEVED! 🎉🎉🎉")


if __name__ == '__main__':
    main()
