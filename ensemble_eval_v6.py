#!/usr/bin/env python3
"""
Ensemble V6: Learned Stacking Meta-Classifier

Instead of fixed per-type weights, train a small model (LightGBM/LogReg)
to predict the correct answer based on all models' logit patterns.

Features per sample:
  - Each model's top-1 predicted class index
  - Each model's top-1 confidence (softmax probability)
  - Each model's logit margin (top1 - top2)
  - Each model's entropy of softmax distribution
  - Number of models agreeing on top-1
  - Question type (one-hot)
  - Whether any model predicts a count answer vs non-count

Uses 5-fold cross-validation on the val set to learn and evaluate.

Usage:
    python ensemble_eval_v6.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
                 mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
                 mcan_trimodal_v16_bert_deep:trimodal_bert_deep_v1:16 \
                 mcan_trimodal_v18_bert_base:trimodal_bert_base_v1:15 \
                 mcan_trimodal_v24_yoloworld:trimodal_yoloworld_v1:16 \
                 mcan_trimodal_v27_distilbert_v3:trimodal_distilbert_v3_v1:16 \
                 mcan_trimodal_v29_v2seed:trimodal_v2seed_v1:16 \
        --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from ensemble_eval_v2 import get_logits

QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']


def get_qtype_for_sample(qa_item):
    template_type = qa_item.get('template_type', 'exist')
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def build_meta_features(all_logits, dataset, n_samples, n_classes):
    """
    Build feature matrix for the meta-classifier.
    Each row = one sample, columns = features about all models' predictions.
    """
    n_models = len(all_logits)
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix

    features_list = []
    gt_labels = []
    valid_indices = []
    qtypes = []

    for i in range(min(n_samples, len(qa_list))):
        gt_ans_str = str(qa_list[i]['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue

        feat = []

        # Per-model features
        preds = []
        confs = []
        for m in range(n_models):
            logits_i = all_logits[m][i, :n_classes]
            probs = softmax(logits_i)

            pred = np.argmax(logits_i)
            conf = probs[pred]
            sorted_l = np.sort(logits_i)[::-1]
            margin = sorted_l[0] - sorted_l[1]
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            preds.append(pred)
            confs.append(conf)

            feat.extend([
                conf,           # Model m confidence
                margin,         # Model m margin
                entropy,        # Model m entropy
            ])

        # Agreement features
        vote_counts = Counter(preds)
        most_common_vote, most_common_count = vote_counts.most_common(1)[0]
        n_unique_preds = len(vote_counts)

        feat.extend([
            most_common_count / n_models,  # Agreement ratio
            n_unique_preds / n_models,     # Disagreement ratio
            np.std(confs),                 # Confidence spread
            np.max(confs) - np.min(confs), # Confidence range
        ])

        # Question type (one-hot)
        qtype = get_qtype_for_sample(qa_list[i])
        for qt in QTYPE_NAMES:
            feat.append(1.0 if qtype == qt else 0.0)

        features_list.append(feat)
        gt_labels.append(gt_idx)
        valid_indices.append(i)
        qtypes.append(qtype)

    X = np.array(features_list, dtype=np.float32)
    y_gt = np.array(gt_labels, dtype=np.int64)

    return X, y_gt, valid_indices, qtypes


def stacking_model_selection(X, y_gt, all_logits, valid_indices, qtypes,
                             n_samples, n_classes, n_folds=5):
    """
    Train a meta-classifier to select which model to trust per sample.

    Target: for each sample, which model index gets it right?
    If no model is right, target = -1 (excluded from training).
    """
    n_models = len(all_logits)

    # Build target: which model is correct?
    # For multi-class: predict the correct answer directly using meta-features
    print(f"\n  Building stacking meta-classifier...")
    print(f"  Features: {X.shape[1]} dims, Samples: {X.shape[0]}")

    # Strategy A: Model selection (predict which model to trust)
    model_targets = []
    for idx_in_valid, sample_i in enumerate(valid_indices):
        gt = y_gt[idx_in_valid]
        correct_models = []
        for m in range(n_models):
            pred = np.argmax(all_logits[m][sample_i, :n_classes])
            if pred == gt:
                correct_models.append(m)

        if correct_models:
            # Pick the model with highest confidence among correct ones
            best_m = max(correct_models,
                         key=lambda m: softmax(all_logits[m][sample_i, :n_classes]).max())
            model_targets.append(best_m)
        else:
            model_targets.append(-1)

    model_targets = np.array(model_targets)
    has_correct = model_targets >= 0
    print(f"  Samples where ≥1 model correct: {has_correct.sum()} ({100*has_correct.mean():.1f}%)")

    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_predictions = np.zeros(len(valid_indices), dtype=np.int64)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Only train on samples where at least one model is correct
        train_mask = has_correct[train_idx]
        X_train = X[train_idx[train_mask]]
        y_train = model_targets[train_idx[train_mask]]

        X_test = X[test_idx]

        # Train model selector
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Predict which model to trust
        selected_models = clf.predict(X_test)

        # Use selected model's prediction
        for ti, (test_i, sel_model) in enumerate(zip(test_idx, selected_models)):
            sample_i = valid_indices[test_i]
            all_predictions[test_i] = np.argmax(all_logits[sel_model][sample_i, :n_classes])

        acc = sum(all_predictions[test_idx[j]] == y_gt[test_idx[j]] for j in range(len(test_idx)))
        print(f"  Fold {fold+1}: {acc}/{len(test_idx)} = {100*acc/len(test_idx):.2f}%")

    return all_predictions


def stacking_logit_blend(X, y_gt, all_logits, valid_indices, qtypes,
                         n_samples, n_classes, n_folds=5):
    """
    Strategy B: Learn per-sample blending weights for logits.
    For each sample, predict optimal weights for combining model logits.
    """
    n_models = len(all_logits)
    print(f"\n  Building logit-blend meta-classifier...")

    # For each question type, train a separate weight predictor
    all_predictions = np.zeros(len(valid_indices), dtype=np.int64)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Per question type, find best weights on training fold
        type_weights = {}
        for qt in QTYPE_NAMES:
            qt_train = [i for i in train_idx if qtypes[i] == qt]
            if len(qt_train) < 10:
                type_weights[qt] = np.ones(n_models) / n_models
                continue

            # Grid search on training fold for this question type
            best_acc = -1
            best_w = np.ones(n_models) / n_models
            np.random.seed(42 + fold)

            for _ in range(3000):
                w = np.random.dirichlet(np.ones(n_models))
                correct = 0
                total = 0
                for i in qt_train:
                    sample_i = valid_indices[i]
                    blended = sum(w[m] * all_logits[m][sample_i, :n_classes] for m in range(n_models))
                    if np.argmax(blended) == y_gt[i]:
                        correct += 1
                    total += 1
                acc = correct / total if total > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                    best_w = w.copy()

            type_weights[qt] = best_w

        # Apply to test fold
        for ti in test_idx:
            sample_i = valid_indices[ti]
            qt = qtypes[ti]
            w = type_weights.get(qt, np.ones(n_models) / n_models)
            blended = sum(w[m] * all_logits[m][sample_i, :n_classes] for m in range(n_models))
            all_predictions[ti] = np.argmax(blended)

        acc = sum(all_predictions[test_idx[j]] == y_gt[test_idx[j]] for j in range(len(test_idx)))
        print(f"  Fold {fold+1}: {acc}/{len(test_idx)} = {100*acc/len(test_idx):.2f}%")

    return all_predictions


def evaluate(predictions_arr, y_gt, qtypes, label=""):
    """Evaluate predictions."""
    type_stats = {qt: [0, 0] for qt in QTYPE_NAMES}
    total_correct = 0
    total = len(y_gt)

    for i in range(total):
        correct = int(predictions_arr[i] == y_gt[i])
        total_correct += correct
        qt = qtypes[i]
        type_stats[qt][0] += correct
        type_stats[qt][1] += 1

    overall = 100 * total_correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Overall {total_correct} / {total} = {overall:.2f}")
    for qt in sorted(type_stats.keys()):
        c, tot = type_stats[qt]
        if tot > 0:
            print(f"  {qt} {c} / {tot} = {100*c/tot:.2f}")
    return overall


def main():
    parser = argparse.ArgumentParser(description='Ensemble V6: Learned Stacking')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    n_models = len(model_specs)
    print(f"Stacking Ensemble of {n_models} models:")
    for spec in model_specs:
        print(f"  {spec}")

    # Load all logits
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

    # Individual model eval
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix

    per_type_accs = {}
    for m, spec in enumerate(model_specs):
        correct = 0
        total = 0
        ts = {qt: [0, 0] for qt in QTYPE_NAMES}
        for i in range(min(n_samples, len(qa_list))):
            gt = ans2ix.get(str(qa_list[i]['answer']), -1)
            if gt == -1:
                continue
            pred = np.argmax(all_logits[m][i, :n_classes])
            c = int(pred == gt)
            correct += c
            total += 1
            qt = get_qtype_for_sample(qa_list[i])
            ts[qt][0] += c
            ts[qt][1] += 1

        per_type_accs[m] = {qt: 100*ts[qt][0]/ts[qt][1] if ts[qt][1] > 0 else 0
                            for qt in QTYPE_NAMES}
        name = spec.split(':')[1] if ':' in spec else spec
        pt = ' '.join(f'{qt}={per_type_accs[m][qt]:.1f}' for qt in QTYPE_NAMES)
        print(f"  Model {m} ({name}): {pt} | Overall={100*correct/total:.2f}%")

    # Build meta features
    X, y_gt, valid_indices, qtypes = build_meta_features(
        all_logits, dataset, n_samples, n_classes
    )

    results = {}

    # ---- Strategy 1: Model Selection ----
    print(f"\n{'='*60}")
    print(f"  Strategy: Learned Model Selection (LogReg)")
    print(f"{'='*60}")
    sel_preds = stacking_model_selection(
        X, y_gt, all_logits, valid_indices, qtypes, n_samples, n_classes,
        n_folds=args.folds
    )
    ov = evaluate(sel_preds, y_gt, qtypes, "Learned Model Selection")
    results['model_selection'] = ov

    # ---- Strategy 2: CV Grid Search (proper, no overfitting) ----
    print(f"\n{'='*60}")
    print(f"  Strategy: CV Grid Search (per-type weights, cross-validated)")
    print(f"{'='*60}")
    cv_preds = stacking_logit_blend(
        X, y_gt, all_logits, valid_indices, qtypes, n_samples, n_classes,
        n_folds=args.folds
    )
    ov = evaluate(cv_preds, y_gt, qtypes, "CV Grid Search")
    results['cv_grid_search'] = ov

    # ---- Strategy 3: Simple equal-weight baseline ----
    print(f"\n{'='*60}")
    print(f"  Strategy: Equal Weight Baseline")
    print(f"{'='*60}")
    eq_preds = np.zeros(len(valid_indices), dtype=np.int64)
    for vi, sample_i in enumerate(valid_indices):
        blended = sum(all_logits[m][sample_i, :n_classes] for m in range(n_models)) / n_models
        eq_preds[vi] = np.argmax(blended)
    ov = evaluate(eq_preds, y_gt, qtypes, "Equal Weight")
    results['equal_weight'] = ov

    # ---- Strategy 4: Oracle (for reference) ----
    oracle_preds = np.zeros(len(valid_indices), dtype=np.int64)
    for vi, sample_i in enumerate(valid_indices):
        gt = y_gt[vi]
        found = False
        for m in range(n_models):
            if np.argmax(all_logits[m][sample_i, :n_classes]) == gt:
                oracle_preds[vi] = gt
                found = True
                break
        if not found:
            oracle_preds[vi] = np.argmax(all_logits[0][sample_i, :n_classes])
    ov = evaluate(oracle_preds, y_gt, qtypes, "Oracle Ceiling")
    results['oracle'] = ov

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if name != 'oracle' and acc == max(v for k, v in results.items() if k != 'oracle') else ""
        if name == 'oracle':
            marker = " (ceiling)"
        print(f"  {name:25s} = {acc:.2f}%{marker}")


if __name__ == '__main__':
    main()
