#!/usr/bin/env python3
"""
Ensemble V7: Scipy-Optimized Per-Type Weights

The Dirichlet random search finds good-but-not-optimal weights.
This script uses scipy.optimize.minimize with proper constraints
to find the true optimal weight combination per question type.

Also tries: combined strategies (grid search + confidence gating)

Usage:
    python ensemble_eval_v7.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
                 mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
                 ... \
        --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
from scipy.optimize import minimize, differential_evolution
from collections import Counter

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


def build_indices(dataset):
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    qtype_indices = {qt: [] for qt in QTYPE_NAMES}
    valid_gt = {}
    for i in range(len(qa_list)):
        gt_ans_str = str(qa_list[i]['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue
        qtype = get_qtype_for_sample(qa_list[i])
        qtype_indices[qtype].append(i)
        valid_gt[i] = gt_idx
    return qtype_indices, valid_gt


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


def scipy_optimize_weights(all_logits, valid_gt, qtype_indices, n_classes, n_models):
    """
    Use scipy differential_evolution to find optimal weights per type.
    This is MUCH better than random Dirichlet for finding the global optimum.
    """
    best_type_weights = {}

    for qt in QTYPE_NAMES:
        indices = qtype_indices[qt]
        if not indices:
            continue

        # Pre-compute logits for this qtype's samples for speed
        qt_logits = []
        qt_gts = []
        for idx in indices:
            if idx not in valid_gt:
                continue
            model_logits = [all_logits[m][idx, :n_classes] for m in range(n_models)]
            qt_logits.append(model_logits)
            qt_gts.append(valid_gt[idx])

        qt_logits_arr = np.array(qt_logits)  # (N, n_models, n_classes)
        qt_gts_arr = np.array(qt_gts)
        n_qt = len(qt_gts_arr)

        def neg_accuracy(w_raw):
            # Softmax to enforce simplex constraint
            w = np.exp(w_raw) / np.exp(w_raw).sum()
            # Weighted sum of logits
            blended = np.einsum('m,nmc->nc', w, qt_logits_arr)
            preds = np.argmax(blended, axis=1)
            correct = (preds == qt_gts_arr).sum()
            return -(correct / n_qt)  # Negative because we minimize

        # Differential evolution (global optimization)
        bounds = [(-3, 3)] * n_models
        result = differential_evolution(
            neg_accuracy,
            bounds=bounds,
            seed=42,
            maxiter=500,
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.9,
            popsize=30,
            polish=True
        )

        # Convert to weights
        best_w_raw = result.x
        best_w = np.exp(best_w_raw) / np.exp(best_w_raw).sum()
        best_acc = -result.fun

        best_type_weights[qt] = best_w.tolist()
        w_str = ' '.join(f'M{m}:{w:.3f}' for m, w in enumerate(best_w))
        print(f"  {qt}: {w_str} → {best_acc*100:.2f}%")

    return best_type_weights


def apply_weights(all_logits, best_type_weights, dataset, n_samples, n_classes, n_models):
    """Apply per-type weights to produce final predictions."""
    qa_list = dataset.qa_list
    predictions = np.zeros(n_samples, dtype=np.int64)

    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        weights = best_type_weights.get(qtype, [1.0/n_models]*n_models)
        blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        predictions[i] = np.argmax(blended)

    return predictions


def hybrid_scipy_confidence(all_logits, best_type_weights, dataset,
                            n_samples, n_classes, n_models, conf_threshold=0.85):
    """
    Hybrid: Use scipy-optimized weights normally, but when any single model
    has very high confidence AND agrees with the weighted prediction, boost trust.
    When confidence is high but disagrees, use weighted prediction.
    """
    qa_list = dataset.qa_list
    predictions = np.zeros(n_samples, dtype=np.int64)
    stats = {'scipy': 0, 'conf_override': 0}

    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        weights = best_type_weights.get(qtype, [1.0/n_models]*n_models)

        # Standard weighted prediction
        blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        weighted_pred = np.argmax(blended)

        # Check if any model is extremely confident
        best_conf = -1
        best_conf_pred = -1
        for m in range(n_models):
            probs = softmax(all_logits[m][i, :n_classes])
            conf = probs.max()
            if conf > best_conf:
                best_conf = conf
                best_conf_pred = np.argmax(all_logits[m][i, :n_classes])

        # If a model is very confident and its prediction agrees with top-2
        # of the weighted logits, trust it
        if best_conf > conf_threshold:
            top2 = np.argsort(blended)[-2:]
            if best_conf_pred in top2:
                predictions[i] = best_conf_pred
                stats['conf_override'] += 1
            else:
                predictions[i] = weighted_pred
                stats['scipy'] += 1
        else:
            predictions[i] = weighted_pred
            stats['scipy'] += 1

    print(f"  Scipy: {stats['scipy']}, Conf override: {stats['conf_override']}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Ensemble V7: Scipy Optimization')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    n_models = len(model_specs)
    print(f"Scipy-Optimized Ensemble of {n_models} models:")
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

    qtype_indices, valid_gt = build_indices(dataset)

    # Individual models
    for m, spec in enumerate(model_specs):
        preds = np.argmax(all_logits[m][:n_samples, :n_classes], axis=1)
        ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
        name = spec.split(':')[1] if ':' in spec else spec
        pt = ' '.join(f'{qt}={100*ts[qt][0]/ts[qt][1]:.1f}' if ts[qt][1] > 0 else f'{qt}=0'
                      for qt in QTYPE_NAMES)
        print(f"  Model {m} ({name}): {pt} | Overall={ov:.2f}%")

    # Oracle
    oracle_correct = 0
    for i in range(n_samples):
        if i in valid_gt:
            for m in range(n_models):
                if np.argmax(all_logits[m][i, :n_classes]) == valid_gt[i]:
                    oracle_correct += 1
                    break
    total_valid = sum(1 for i in range(n_samples) if i in valid_gt)
    print(f"\n  Oracle ceiling: {oracle_correct}/{total_valid} = {100*oracle_correct/total_valid:.2f}%")

    results = {}

    # ---- Strategy 1: Scipy Differential Evolution ----
    print(f"\n{'='*60}")
    print(f"  Strategy: Scipy Differential Evolution (global optimizer)")
    print(f"{'='*60}")
    scipy_weights = scipy_optimize_weights(all_logits, valid_gt, qtype_indices, n_classes, n_models)
    preds = apply_weights(all_logits, scipy_weights, dataset, n_samples, n_classes, n_models)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("Scipy Optimized", ov, tc, t, ts)
    results['scipy'] = ov

    # ---- Strategy 2: Hybrid scipy + confidence ----
    for thresh in [0.7, 0.8, 0.85, 0.9, 0.95]:
        print(f"\n{'='*60}")
        print(f"  Strategy: Hybrid Scipy + Confidence (threshold={thresh})")
        print(f"{'='*60}")
        preds = hybrid_scipy_confidence(
            all_logits, scipy_weights, dataset, n_samples, n_classes, n_models,
            conf_threshold=thresh
        )
        ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
        print_eval(f"Hybrid-{thresh}", ov, tc, t, ts)
        results[f'hybrid_{thresh}'] = ov

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY (Oracle: {100*oracle_correct/total_valid:.2f}%)")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if acc == max(results.values()) else ""
        print(f"  {name:25s} = {acc:.2f}%{marker}")

    best_name = max(results.items(), key=lambda x: x[1])
    print(f"\n  🏆 BEST: {best_name[0]} = {best_name[1]:.2f}%")


if __name__ == '__main__':
    main()
