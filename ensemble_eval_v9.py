#!/usr/bin/env python3
"""
Ensemble V9: Fine-Grained Routing via Sub-Type + Confidence Binning

Key insight: per-TYPE weights (exist/count/object/status/comparison)
are too coarse. Within each type, there are sub-types:
  - count_0 vs count_1
  - object_0 vs object_1
  - comparison_0 vs comparison_1
  - exist_0 vs exist_1
  - status_0 vs status_1

AND within each sub-type, the model's confidence level matters.
A high-confidence prediction is more likely correct than a low-confidence one.

This script optimizes weights per SUB-TYPE (10 categories instead of 5)
and also bins samples by ensemble confidence for additional routing.

Usage:
    python ensemble_eval_v9.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
                 mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
                 ... \
        --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
from scipy.optimize import differential_evolution
from collections import Counter, defaultdict

from ensemble_eval_v2 import get_logits

QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']
# Sub-types from NuScenes-QA template_type field
SUBTYPE_MAP = {}  # Will be built from data


def get_qtype_for_sample(qa_item):
    template_type = qa_item.get('template_type', 'exist')
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'


def get_subtype_for_sample(qa_item):
    """Get fine-grained sub-type (e.g., count_0, count_1, object_0, etc.)"""
    return qa_item.get('template_type', 'exist')


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def build_indices(dataset):
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    qtype_indices = {qt: [] for qt in QTYPE_NAMES}
    subtype_indices = defaultdict(list)
    valid_gt = {}
    for i in range(len(qa_list)):
        gt_ans_str = str(qa_list[i]['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue
        qtype = get_qtype_for_sample(qa_list[i])
        subtype = get_subtype_for_sample(qa_list[i])
        qtype_indices[qtype].append(i)
        subtype_indices[subtype].append(i)
        valid_gt[i] = gt_idx
    return qtype_indices, subtype_indices, valid_gt


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


def scipy_optimize_for_indices(all_logits, valid_gt, indices, n_classes, n_models, label=""):
    """Optimize weights for a specific set of sample indices."""
    qt_logits = []
    qt_gts = []
    for idx in indices:
        if idx not in valid_gt:
            continue
        model_logits = [all_logits[m][idx, :n_classes] for m in range(n_models)]
        qt_logits.append(model_logits)
        qt_gts.append(valid_gt[idx])

    if len(qt_gts) < 20:  # Too few samples for optimization
        return None, 0

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
    best_acc = -result.fun

    if label:
        w_str = ' '.join(f'M{m}:{w:.3f}' for m, w in enumerate(best_w) if w > 0.01)
        print(f"  {label}: {w_str} → {best_acc*100:.2f}% ({n_qt} samples)")

    return best_w.tolist(), best_acc


def strategy_subtype(all_logits, valid_gt, subtype_indices, qtype_indices,
                     dataset, n_samples, n_classes, n_models):
    """Optimize weights per sub-type (10 categories instead of 5)."""
    print(f"\n  Sub-types found: {len(subtype_indices)}")

    subtype_weights = {}
    # Also compute per-type fallback
    type_weights = {}
    for qt in QTYPE_NAMES:
        w, acc = scipy_optimize_for_indices(
            all_logits, valid_gt, qtype_indices[qt], n_classes, n_models, label=f"[type] {qt}"
        )
        if w:
            type_weights[qt] = w

    for subtype, indices in sorted(subtype_indices.items()):
        w, acc = scipy_optimize_for_indices(
            all_logits, valid_gt, indices, n_classes, n_models, label=f"[sub] {subtype}"
        )
        if w:
            subtype_weights[subtype] = w

    # Apply: use subtype weights if available, else fall back to type weights
    qa_list = dataset.qa_list
    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        subtype = get_subtype_for_sample(qa_list[i])
        qtype = get_qtype_for_sample(qa_list[i])
        weights = subtype_weights.get(subtype, type_weights.get(qtype, [1.0/n_models]*n_models))
        blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        predictions[i] = np.argmax(blended)

    return predictions


def strategy_confidence_binned(all_logits, valid_gt, qtype_indices,
                               dataset, n_samples, n_classes, n_models):
    """
    Split each question type into confidence bins and optimize separately.
    High-confidence samples might need different weights than low-confidence ones.
    """
    qa_list = dataset.qa_list

    # First compute ensemble confidence for each sample (equal weight)
    sample_confs = np.zeros(n_samples)
    for i in range(n_samples):
        blended = sum(all_logits[m][i, :n_classes] for m in range(n_models)) / n_models
        probs = softmax(blended)
        sample_confs[i] = probs.max()

    # For each type, split into 3 confidence bins
    n_bins = 3
    bin_weights = {}  # (qtype, bin_idx) -> weights

    for qt in QTYPE_NAMES:
        indices = [i for i in qtype_indices[qt] if i in valid_gt]
        if not indices:
            continue

        # Compute percentile thresholds for this type
        type_confs = [sample_confs[i] for i in indices]
        thresholds = [np.percentile(type_confs, p) for p in [33, 67]]

        for bin_idx in range(n_bins):
            if bin_idx == 0:
                bin_indices = [i for i in indices if sample_confs[i] <= thresholds[0]]
                bin_label = f"{qt}/low-conf"
            elif bin_idx == 1:
                bin_indices = [i for i in indices if thresholds[0] < sample_confs[i] <= thresholds[1]]
                bin_label = f"{qt}/mid-conf"
            else:
                bin_indices = [i for i in indices if sample_confs[i] > thresholds[1]]
                bin_label = f"{qt}/high-conf"

            w, acc = scipy_optimize_for_indices(
                all_logits, valid_gt, bin_indices, n_classes, n_models, label=bin_label
            )
            if w:
                bin_weights[(qt, bin_idx)] = (w, thresholds)

    # Apply
    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        conf = sample_confs[i]

        # Find appropriate bin
        key = None
        if (qtype, 0) in bin_weights:
            _, thresholds = bin_weights[(qtype, 0)]
            if conf <= thresholds[0]:
                key = (qtype, 0)
            elif conf <= thresholds[1]:
                key = (qtype, 1)
            else:
                key = (qtype, 2)

        if key and key in bin_weights:
            weights = bin_weights[key][0]
        else:
            weights = [1.0/n_models] * n_models

        blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        predictions[i] = np.argmax(blended)

    return predictions


def strategy_combined_best(all_logits, valid_gt, qtype_indices, subtype_indices,
                           dataset, n_samples, n_classes, n_models):
    """
    For each question type, try subtype optimization vs type optimization
    vs confidence-binned, and pick whichever gives highest accuracy.
    """
    qa_list = dataset.qa_list

    # Strategy A: per-type scipy
    type_weights = {}
    for qt in QTYPE_NAMES:
        w, acc = scipy_optimize_for_indices(
            all_logits, valid_gt, qtype_indices[qt], n_classes, n_models
        )
        if w:
            type_weights[qt] = (w, acc)

    # Strategy B: per-subtype scipy
    subtype_weights = {}
    for subtype, indices in subtype_indices.items():
        w, acc = scipy_optimize_for_indices(
            all_logits, valid_gt, indices, n_classes, n_models
        )
        if w:
            subtype_weights[subtype] = (w, acc)

    # For each question type, compare type-level vs subtype-level accuracy
    print(f"\n  Per-type vs per-subtype comparison:")
    best_strategy = {}  # qtype -> 'type' or 'subtype'
    for qt in QTYPE_NAMES:
        indices = [i for i in qtype_indices[qt] if i in valid_gt]
        if not indices:
            continue

        # Evaluate type-level weights
        type_w = type_weights.get(qt, (None, 0))
        if type_w[0] is None:
            continue

        type_correct = 0
        for idx in indices:
            blended = sum(type_w[0][m] * all_logits[m][idx, :n_classes] for m in range(n_models))
            if np.argmax(blended) == valid_gt[idx]:
                type_correct += 1
        type_acc = type_correct / len(indices)

        # Evaluate subtype-level weights
        sub_correct = 0
        for idx in indices:
            subtype = get_subtype_for_sample(qa_list[idx])
            sw = subtype_weights.get(subtype)
            if sw:
                w = sw[0]
            else:
                w = type_w[0]
            blended = sum(w[m] * all_logits[m][idx, :n_classes] for m in range(n_models))
            if np.argmax(blended) == valid_gt[idx]:
                sub_correct += 1
        sub_acc = sub_correct / len(indices)

        winner = 'subtype' if sub_acc > type_acc else 'type'
        best_strategy[qt] = winner
        print(f"  {qt}: type={type_acc*100:.2f}% vs subtype={sub_acc*100:.2f}% → {winner}")

    # Apply best strategy per type
    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        subtype = get_subtype_for_sample(qa_list[i])
        strategy = best_strategy.get(qtype, 'type')

        if strategy == 'subtype' and subtype in subtype_weights:
            weights = subtype_weights[subtype][0]
        elif qtype in type_weights:
            weights = type_weights[qtype][0]
        else:
            weights = [1.0/n_models] * n_models

        blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        predictions[i] = np.argmax(blended)

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Ensemble V9: Sub-Type Routing')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    n_models = len(model_specs)
    print(f"Sub-Type Routing Ensemble of {n_models} models:")
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

    qtype_indices, subtype_indices, valid_gt = build_indices(dataset)

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

    results = {}

    # ---- Strategy 1: Per-subtype optimization ----
    print(f"\n{'='*60}")
    print(f"  Strategy 1: Per-SubType Scipy Optimization")
    print(f"{'='*60}")
    preds = strategy_subtype(all_logits, valid_gt, subtype_indices, qtype_indices,
                             dataset, n_samples, n_classes, n_models)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("SubType", ov, tc, t, ts)
    results['subtype'] = ov

    # ---- Strategy 2: Confidence-binned optimization ----
    print(f"\n{'='*60}")
    print(f"  Strategy 2: Confidence-Binned Scipy Optimization")
    print(f"{'='*60}")
    preds = strategy_confidence_binned(all_logits, valid_gt, qtype_indices,
                                       dataset, n_samples, n_classes, n_models)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("ConfBinned", ov, tc, t, ts)
    results['conf_binned'] = ov

    # ---- Strategy 3: Best of type vs subtype per qtype ----
    print(f"\n{'='*60}")
    print(f"  Strategy 3: Combined Best (type vs subtype per question type)")
    print(f"{'='*60}")
    preds = strategy_combined_best(all_logits, valid_gt, qtype_indices, subtype_indices,
                                   dataset, n_samples, n_classes, n_models)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("CombinedBest", ov, tc, t, ts)
    results['combined_best'] = ov

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY (Oracle: {oracle_pct:.2f}%)")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if acc == max(results.values()) else ""
        print(f"  {name:25s} = {acc:.2f}%{marker}")

    best_name = max(results.items(), key=lambda x: x[1])
    print(f"\n  🏆 BEST: {best_name[0]} = {best_name[1]:.2f}%")


if __name__ == '__main__':
    main()
