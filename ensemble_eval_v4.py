#!/usr/bin/env python3
"""
Ensemble V4: Advanced strategies + Oracle analysis + Temperature sweep.

Features:
  1. Oracle ceiling: what's the best possible if we always pick the right model?
  2. Temperature sweep: find optimal softmax temp for qtype_routed
  3. Confidence-weighted: weight by model confidence per sample (not just accuracy)
  4. Hybrid: combine qtype_routed logits with majority vote as tiebreaker
  5. Learned weights via grid search over per-type weight combinations

Usage:
    python ensemble_eval_v4.py \
        --models mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
                 mcan_trimodal_v18_bert_base:trimodal_bert_base_v1:15 \
                 mcan_trimodal_v24_yoloworld:trimodal_yoloworld_v1:16 \
        --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
import torch.utils.data as Data
from itertools import product

from src.models.mcan.model_cfgs import Cfgs
from src.models.model_loader import ModelLoader
from src.datasets.nuscenes_qa import NuScenes_QA

from ensemble_eval_v2 import load_config, get_logits

QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']


def get_qtype_for_sample(qa_item):
    template_type = qa_item.get('template_type', 'exist')
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'


def get_qtype_indices(dataset):
    """Build mapping: qtype -> list of sample indices."""
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    qtype_indices = {qt: [] for qt in QTYPE_NAMES}
    valid_gt = {}  # idx -> gt_idx

    for i in range(len(qa_list)):
        qa_item = qa_list[i]
        gt_ans_str = str(qa_item['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue
        qtype = get_qtype_for_sample(qa_item)
        qtype_indices[qtype].append(i)
        valid_gt[i] = gt_idx

    return qtype_indices, valid_gt


def evaluate_preds(predictions, valid_gt, qtype_indices, n_samples):
    """Evaluate predictions, return overall acc and per-type dict."""
    type_stats = {qt: {'correct': 0, 'total': 0} for qt in QTYPE_NAMES}
    total_correct = 0
    total = 0

    for i in range(min(n_samples, len(predictions))):
        if i not in valid_gt:
            continue
        gt_idx = valid_gt[i]
        correct = int(predictions[i] == gt_idx)
        total_correct += correct
        total += 1
        for qt in QTYPE_NAMES:
            if i in set(qtype_indices[qt]):
                type_stats[qt]['correct'] += correct
                type_stats[qt]['total'] += 1
                break

    overall = 100 * total_correct / total if total > 0 else 0
    per_type = {}
    for qt in QTYPE_NAMES:
        s = type_stats[qt]
        per_type[qt] = 100 * s['correct'] / s['total'] if s['total'] > 0 else 0

    return overall, per_type, total_correct, total


def print_results(label, overall, per_type, total_correct=None, total=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if total_correct is not None:
        print(f"Overall {total_correct} / {total} = {overall:.2f}")
    else:
        print(f"Overall = {overall:.2f}%")
    for qt in sorted(per_type.keys()):
        print(f"  {qt} = {per_type[qt]:.2f}%")
    return overall


# ============================================================
# Strategy 1: Oracle (theoretical ceiling)
# ============================================================
def oracle_ensemble(all_logits, valid_gt, qtype_indices, n_samples, n_classes):
    """For each sample, pick the model that gets it right (if any do)."""
    n_models = len(all_logits)
    predictions = np.zeros(n_samples, dtype=np.int64)
    any_correct = 0
    all_wrong = 0

    for i in range(n_samples):
        if i not in valid_gt:
            predictions[i] = np.argmax(all_logits[0][i, :n_classes])
            continue

        gt_idx = valid_gt[i]
        found_correct = False
        for m in range(n_models):
            pred = np.argmax(all_logits[m][i, :n_classes])
            if pred == gt_idx:
                predictions[i] = pred
                found_correct = True
                any_correct += 1
                break

        if not found_correct:
            # No model gets it right — use model 0's prediction
            predictions[i] = np.argmax(all_logits[0][i, :n_classes])
            all_wrong += 1

    print(f"  Oracle: {any_correct} samples have ≥1 correct model, {all_wrong} have none")
    return predictions


# ============================================================
# Strategy 2: Temperature sweep for qtype_routed
# ============================================================
def qtype_routed_with_temp(all_logits, per_type_accs, qtype_indices, dataset, temp):
    """qtype_routed with specific temperature."""
    qa_list = dataset.qa_list
    n_samples = all_logits[0].shape[0]
    n_classes = min(l.shape[1] for l in all_logits)
    n_models = len(all_logits)

    type_weights = {}
    for qt in QTYPE_NAMES:
        accs = np.array([per_type_accs[m][qt] for m in range(n_models)])
        exp_accs = np.exp((accs - accs.max()) * temp)
        weights = exp_accs / exp_accs.sum()
        type_weights[qt] = weights

    ensemble_logits = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        weights = type_weights.get(qtype, np.ones(n_models) / n_models)
        for m in range(n_models):
            ensemble_logits[i] += weights[m] * all_logits[m][i, :n_classes]

    return np.argmax(ensemble_logits, axis=1)


# ============================================================
# Strategy 3: Confidence-weighted per sample
# ============================================================
def confidence_weighted(all_logits, n_samples, n_classes):
    """
    Weight each model's logits by its confidence (max softmax prob).
    Models that are more confident on a particular sample get more weight.
    """
    n_models = len(all_logits)
    ensemble_logits = np.zeros((n_samples, n_classes), dtype=np.float32)

    for i in range(n_samples):
        weights = np.zeros(n_models)
        for m in range(n_models):
            logits_i = all_logits[m][i, :n_classes]
            # Softmax to get probabilities
            exp_l = np.exp(logits_i - logits_i.max())
            probs = exp_l / exp_l.sum()
            # Confidence = max probability (entropy would also work)
            weights[m] = probs.max()

        # Normalize weights
        weights = weights / weights.sum()

        for m in range(n_models):
            ensemble_logits[i] += weights[m] * all_logits[m][i, :n_classes]

    return np.argmax(ensemble_logits, axis=1)


# ============================================================
# Strategy 4: Hybrid (qtype_routed + confidence)
# ============================================================
def hybrid_ensemble(all_logits, per_type_accs, qtype_indices, dataset, temp=2.0, conf_weight=0.3):
    """
    Combine qtype routing weights with per-sample confidence.
    Final weight = (1-conf_weight) * qtype_weight + conf_weight * confidence_weight
    """
    qa_list = dataset.qa_list
    n_samples = all_logits[0].shape[0]
    n_classes = min(l.shape[1] for l in all_logits)
    n_models = len(all_logits)

    # Pre-compute qtype weights
    type_weights = {}
    for qt in QTYPE_NAMES:
        accs = np.array([per_type_accs[m][qt] for m in range(n_models)])
        exp_accs = np.exp((accs - accs.max()) * temp)
        weights = exp_accs / exp_accs.sum()
        type_weights[qt] = weights

    ensemble_logits = np.zeros((n_samples, n_classes), dtype=np.float32)

    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        qw = type_weights.get(qtype, np.ones(n_models) / n_models)

        # Per-sample confidence weights
        cw = np.zeros(n_models)
        for m in range(n_models):
            logits_i = all_logits[m][i, :n_classes]
            exp_l = np.exp(logits_i - logits_i.max())
            probs = exp_l / exp_l.sum()
            cw[m] = probs.max()
        cw = cw / cw.sum()

        # Combine
        final_w = (1 - conf_weight) * qw + conf_weight * cw
        final_w = final_w / final_w.sum()

        for m in range(n_models):
            ensemble_logits[i] += final_w[m] * all_logits[m][i, :n_classes]

    return np.argmax(ensemble_logits, axis=1)


# ============================================================
# Strategy 5: Grid search per-type weights
# ============================================================
def grid_search_weights(all_logits, valid_gt, qtype_indices, dataset, n_classes):
    """
    Brute-force grid search over per-type weight combinations.
    For 3 models, search weight triplets on a coarse grid.
    """
    qa_list = dataset.qa_list
    n_samples = all_logits[0].shape[0]
    n_models = len(all_logits)

    if n_models > 4:
        print("  Grid search skipped (too many models)")
        return None, None

    # Weight grid (for 3 models: 0.0 to 1.0 in steps of 0.1)
    step = 0.1
    weight_vals = np.arange(0.0, 1.01, step)

    # For each qtype, find best weight combination
    best_type_weights = {}
    for qt in QTYPE_NAMES:
        indices = qtype_indices[qt]
        if len(indices) == 0:
            continue

        best_acc = -1
        best_w = None

        if n_models == 2:
            for w0 in weight_vals:
                w1 = 1.0 - w0
                if w1 < -0.01:
                    continue
                weights = [w0, w1]
                correct = 0
                for i in indices:
                    if i not in valid_gt:
                        continue
                    logits_i = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
                    pred = np.argmax(logits_i)
                    if pred == valid_gt[i]:
                        correct += 1
                total_valid = sum(1 for i in indices if i in valid_gt)
                acc = correct / total_valid if total_valid > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                    best_w = list(weights)

        elif n_models == 3:
            for w0 in weight_vals:
                for w1 in weight_vals:
                    w2 = 1.0 - w0 - w1
                    if w2 < -0.01 or w2 > 1.01:
                        continue
                    weights = [w0, w1, max(0, w2)]
                    correct = 0
                    total_valid = 0
                    for i in indices:
                        if i not in valid_gt:
                            continue
                        logits_i = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
                        pred = np.argmax(logits_i)
                        total_valid += 1
                        if pred == valid_gt[i]:
                            correct += 1
                    acc = correct / total_valid if total_valid > 0 else 0
                    if acc > best_acc:
                        best_acc = acc
                        best_w = list(weights)

        elif n_models == 4:
            # Coarser grid for 4 models
            coarse_step = 0.2
            coarse_vals = np.arange(0.0, 1.01, coarse_step)
            for w0 in coarse_vals:
                for w1 in coarse_vals:
                    for w2 in coarse_vals:
                        w3 = 1.0 - w0 - w1 - w2
                        if w3 < -0.01 or w3 > 1.01:
                            continue
                        weights = [w0, w1, w2, max(0, w3)]
                        correct = 0
                        total_valid = 0
                        for i in indices:
                            if i not in valid_gt:
                                continue
                            logits_i = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
                            pred = np.argmax(logits_i)
                            total_valid += 1
                            if pred == valid_gt[i]:
                                correct += 1
                        acc = correct / total_valid if total_valid > 0 else 0
                        if acc > best_acc:
                            best_acc = acc
                            best_w = list(weights)

        if best_w is not None:
            best_type_weights[qt] = best_w
            w_str = ' '.join(f'M{m}:{w:.2f}' for m, w in enumerate(best_w))
            print(f"  {qt}: {w_str} → {best_acc*100:.2f}%")

    # Apply best weights
    ensemble_logits = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        weights = best_type_weights.get(qtype, [1.0/n_models]*n_models)
        for m in range(n_models):
            ensemble_logits[i] += weights[m] * all_logits[m][i, :n_classes]

    return np.argmax(ensemble_logits, axis=1), best_type_weights


def main():
    parser = argparse.ArgumentParser(description='Ensemble V4: Advanced')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    print(f"Advanced Ensemble of {len(model_specs)} models:")
    for spec in model_specs:
        print(f"  {spec}")

    # Get logits
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

    # Build index structures
    qtype_indices, valid_gt = get_qtype_indices(dataset)

    # Individual model evaluation
    per_type_accs = {}
    for m, (spec, logits) in enumerate(zip(model_specs, all_logits)):
        preds = np.argmax(logits[:n_samples, :n_classes], axis=1)
        overall, per_type, tc, t = evaluate_preds(preds, valid_gt, qtype_indices, n_samples)
        per_type_accs[m] = per_type
        name = spec.split(':')[1] if ':' in spec else spec
        per_type_str = ' '.join(f'{qt}={per_type[qt]:.1f}' for qt in QTYPE_NAMES)
        print(f"\n  Model {m} ({name}): {per_type_str} | Overall={overall:.2f}%")

    results = {}

    # ---- Strategy 0: Oracle ceiling ----
    print(f"\n{'='*60}")
    print(f"  Strategy: ORACLE (theoretical ceiling)")
    print(f"{'='*60}")
    oracle_preds = oracle_ensemble(all_logits, valid_gt, qtype_indices, n_samples, n_classes)
    overall, per_type, tc, t = evaluate_preds(oracle_preds, valid_gt, qtype_indices, n_samples)
    print_results("Oracle Ceiling", overall, per_type, tc, t)
    results['oracle'] = overall

    # ---- Strategy 1: Temperature sweep ----
    print(f"\n{'='*60}")
    print(f"  Strategy: Temperature Sweep (qtype_routed)")
    print(f"{'='*60}")
    best_temp_acc = 0
    best_temp = 2.0
    for temp in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        preds = qtype_routed_with_temp(all_logits, per_type_accs, qtype_indices, dataset, temp)
        overall, per_type, tc, t = evaluate_preds(preds, valid_gt, qtype_indices, n_samples)
        marker = " ★" if overall > best_temp_acc else ""
        print(f"  temp={temp:5.1f} → {overall:.2f}%{marker}")
        if overall > best_temp_acc:
            best_temp_acc = overall
            best_temp = temp
    results['temp_sweep'] = best_temp_acc
    print(f"  Best temperature: {best_temp} → {best_temp_acc:.2f}%")

    # ---- Strategy 2: Confidence-weighted ----
    print(f"\n{'='*60}")
    print(f"  Strategy: Confidence-Weighted")
    print(f"{'='*60}")
    conf_preds = confidence_weighted(all_logits, n_samples, n_classes)
    overall, per_type, tc, t = evaluate_preds(conf_preds, valid_gt, qtype_indices, n_samples)
    print_results("Confidence-Weighted", overall, per_type, tc, t)
    results['confidence'] = overall

    # ---- Strategy 3: Hybrid ----
    print(f"\n{'='*60}")
    print(f"  Strategy: Hybrid (qtype_routed + confidence)")
    print(f"{'='*60}")
    best_hybrid = 0
    best_hybrid_params = (2.0, 0.3)
    for temp in [best_temp, 2.0, 5.0]:
        for cw in [0.1, 0.2, 0.3, 0.4, 0.5]:
            preds = hybrid_ensemble(all_logits, per_type_accs, qtype_indices, dataset, temp, cw)
            overall, per_type, tc, t = evaluate_preds(preds, valid_gt, qtype_indices, n_samples)
            marker = " ★" if overall > best_hybrid else ""
            print(f"  temp={temp:.1f}, conf_w={cw:.1f} → {overall:.2f}%{marker}")
            if overall > best_hybrid:
                best_hybrid = overall
                best_hybrid_params = (temp, cw)
    results['hybrid'] = best_hybrid

    # ---- Strategy 4: Grid search ----
    print(f"\n{'='*60}")
    print(f"  Strategy: Grid Search (per-type optimal weights)")
    print(f"{'='*60}")
    grid_preds, grid_weights = grid_search_weights(all_logits, valid_gt, qtype_indices, dataset, n_classes)
    if grid_preds is not None:
        overall, per_type, tc, t = evaluate_preds(grid_preds, valid_gt, qtype_indices, n_samples)
        print_results("Grid Search", overall, per_type, tc, t)
        results['grid_search'] = overall

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if acc == max(results.values()) else ""
        print(f"  {name:20s} = {acc:.2f}%{marker}")

    best_name = max(results.items(), key=lambda x: x[1])
    print(f"\n  🏆 BEST: {best_name[0]} = {best_name[1]:.2f}%")


if __name__ == '__main__':
    main()
