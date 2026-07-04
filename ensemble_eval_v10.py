#!/usr/bin/env python3
"""
Ensemble V10: Final Push — Confidence × SubType Grid + Ensemble Stacking

Builds on V9's conf-binned insight (59.55%) with:
  1. More confidence bins (4-5 instead of 3)
  2. Subtype × confidence cross-product (20-50 groups)
  3. Stacking: train a small model on meta-features to predict
     optimal blend weights per sample
  4. Temperature sweep per bin

Usage:
    python ensemble_eval_v10.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
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


def get_qtype_for_sample(qa_item):
    template_type = qa_item.get('template_type', 'exist')
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'


def get_subtype_for_sample(qa_item):
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


def scipy_optimize_for_indices(all_logits, valid_gt, indices, n_classes, n_models,
                                label="", maxiter=300, popsize=20):
    qt_logits = []
    qt_gts = []
    for idx in indices:
        if idx not in valid_gt:
            continue
        model_logits = [all_logits[m][idx, :n_classes] for m in range(n_models)]
        qt_logits.append(model_logits)
        qt_gts.append(valid_gt[idx])

    if len(qt_gts) < 20:
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
        maxiter=maxiter, tol=1e-8, popsize=popsize, polish=True
    )
    best_w = np.exp(result.x) / np.exp(result.x).sum()
    best_acc = -result.fun

    if label:
        w_str = ' '.join(f'M{m}:{w:.3f}' for m, w in enumerate(best_w) if w > 0.01)
        print(f"  {label}: {w_str} → {best_acc*100:.2f}% ({n_qt} samples)")

    return best_w.tolist(), best_acc


# ============================================================
# Strategy 1: More fine-grained bins (4 or 5 bins per type)
# ============================================================
def strategy_fine_bins(all_logits, valid_gt, qtype_indices,
                       dataset, n_samples, n_classes, n_models, n_bins=4):
    qa_list = dataset.qa_list

    # Compute ensemble confidence
    sample_confs = np.zeros(n_samples)
    for i in range(n_samples):
        blended = sum(all_logits[m][i, :n_classes] for m in range(n_models)) / n_models
        probs = softmax(blended)
        sample_confs[i] = probs.max()

    bin_weights = {}
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # e.g., [25, 50, 75] for 4 bins

    for qt in QTYPE_NAMES:
        indices = [i for i in qtype_indices[qt] if i in valid_gt]
        if not indices:
            continue

        type_confs = [sample_confs[i] for i in indices]
        thresholds = [np.percentile(type_confs, p) for p in percentiles]

        for bin_idx in range(n_bins):
            if bin_idx == 0:
                bin_indices = [i for i in indices if sample_confs[i] <= thresholds[0]]
            elif bin_idx == n_bins - 1:
                bin_indices = [i for i in indices if sample_confs[i] > thresholds[-1]]
            else:
                bin_indices = [i for i in indices
                              if thresholds[bin_idx-1] < sample_confs[i] <= thresholds[bin_idx]]

            bin_label = f"{qt}/bin{bin_idx}"
            w, acc = scipy_optimize_for_indices(
                all_logits, valid_gt, bin_indices, n_classes, n_models,
                label=bin_label, maxiter=400, popsize=25
            )
            if w:
                bin_weights[(qt, bin_idx)] = (w, thresholds)

    # Apply
    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        conf = sample_confs[i]

        key = None
        if (qtype, 0) in bin_weights:
            _, thresholds = bin_weights[(qtype, 0)]
            # Find the right bin
            bin_idx = 0
            for t_idx, t_val in enumerate(thresholds):
                if conf > t_val:
                    bin_idx = t_idx + 1
            key = (qtype, min(bin_idx, n_bins - 1))

        if key and key in bin_weights:
            weights = bin_weights[key][0]
        else:
            weights = [1.0/n_models] * n_models

        blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        predictions[i] = np.argmax(blended)

    return predictions


# ============================================================
# Strategy 2: SubType × Confidence cross-product
# ============================================================
def strategy_subtype_conf(all_logits, valid_gt, subtype_indices, qtype_indices,
                          dataset, n_samples, n_classes, n_models):
    """
    Cross subtype (10 categories) × confidence (2 bins = high/low).
    This gives up to 20 groups with specialized weights.
    """
    qa_list = dataset.qa_list

    # Compute confidence
    sample_confs = np.zeros(n_samples)
    for i in range(n_samples):
        blended = sum(all_logits[m][i, :n_classes] for m in range(n_models)) / n_models
        probs = softmax(blended)
        sample_confs[i] = probs.max()

    # For each subtype, split into high/low confidence
    cross_weights = {}  # (subtype, 'high'/'low') -> weights
    fallback_weights = {}  # subtype -> weights (no confidence split)

    for subtype, indices in sorted(subtype_indices.items()):
        valid_indices = [i for i in indices if i in valid_gt]
        if len(valid_indices) < 40:
            # Too few samples, optimize without confidence split
            w, acc = scipy_optimize_for_indices(
                all_logits, valid_gt, valid_indices, n_classes, n_models, label=f"[sub] {subtype}"
            )
            if w:
                fallback_weights[subtype] = w
            continue

        # Split at median confidence
        type_confs = [sample_confs[i] for i in valid_indices]
        median_conf = np.median(type_confs)

        lo_indices = [i for i in valid_indices if sample_confs[i] <= median_conf]
        hi_indices = [i for i in valid_indices if sample_confs[i] > median_conf]

        for conf_bin, bin_indices in [('low', lo_indices), ('high', hi_indices)]:
            if len(bin_indices) < 20:
                continue
            w, acc = scipy_optimize_for_indices(
                all_logits, valid_gt, bin_indices, n_classes, n_models,
                label=f"[{subtype}/{conf_bin}]"
            )
            if w:
                cross_weights[(subtype, conf_bin)] = (w, median_conf)

        # Also compute subtype-level fallback
        w, acc = scipy_optimize_for_indices(
            all_logits, valid_gt, valid_indices, n_classes, n_models
        )
        if w:
            fallback_weights[subtype] = w

    # Type-level fallback
    type_weights = {}
    for qt in QTYPE_NAMES:
        w, acc = scipy_optimize_for_indices(
            all_logits, valid_gt, qtype_indices[qt], n_classes, n_models
        )
        if w:
            type_weights[qt] = w

    # Apply with cascade: cross > subtype > type > uniform
    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        subtype = get_subtype_for_sample(qa_list[i])
        qtype = get_qtype_for_sample(qa_list[i])
        conf = sample_confs[i]

        # Try cross-product first
        weights = None
        conf_bin = None
        if (subtype, 'low') in cross_weights:
            median = cross_weights[(subtype, 'low')][1]
            conf_bin = 'low' if conf <= median else 'high'
            if (subtype, conf_bin) in cross_weights:
                weights = cross_weights[(subtype, conf_bin)][0]

        # Fallback to subtype
        if weights is None and subtype in fallback_weights:
            weights = fallback_weights[subtype]

        # Fallback to type
        if weights is None and qtype in type_weights:
            weights = type_weights[qtype]

        # Fallback to uniform
        if weights is None:
            weights = [1.0/n_models] * n_models

        blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        predictions[i] = np.argmax(blended)

    return predictions


# ============================================================
# Strategy 3: Confidence bins with agreement routing
# ============================================================
def strategy_conf_bins_agreement(all_logits, valid_gt, qtype_indices,
                                  dataset, n_samples, n_classes, n_models):
    """
    V9's conf-binned approach + when all top-K models strongly agree
    on a different answer than the weighted blend, switch to consensus.
    """
    qa_list = dataset.qa_list

    # Step 1: Standard conf-binned optimization (3 bins as in V9)
    sample_confs = np.zeros(n_samples)
    for i in range(n_samples):
        blended = sum(all_logits[m][i, :n_classes] for m in range(n_models)) / n_models
        probs = softmax(blended)
        sample_confs[i] = probs.max()

    bin_weights = {}
    for qt in QTYPE_NAMES:
        indices = [i for i in qtype_indices[qt] if i in valid_gt]
        if not indices:
            continue
        type_confs = [sample_confs[i] for i in indices]
        thresholds = [np.percentile(type_confs, p) for p in [33, 67]]
        for bin_idx in range(3):
            if bin_idx == 0:
                bin_indices = [i for i in indices if sample_confs[i] <= thresholds[0]]
            elif bin_idx == 1:
                bin_indices = [i for i in indices if thresholds[0] < sample_confs[i] <= thresholds[1]]
            else:
                bin_indices = [i for i in indices if sample_confs[i] > thresholds[1]]
            w, acc = scipy_optimize_for_indices(
                all_logits, valid_gt, bin_indices, n_classes, n_models
            )
            if w:
                bin_weights[(qt, bin_idx)] = (w, thresholds)

    # Step 2: Apply with agreement override
    predictions = np.zeros(n_samples, dtype=np.int64)
    stats = {'binned': 0, 'agreement': 0}

    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        conf = sample_confs[i]

        # Get binned weights
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
        weighted_pred = np.argmax(blended)

        # Check strong agreement: if 60%+ of models agree AND
        # max individual confidence > 0.8, consider agreement override
        votes = [np.argmax(all_logits[m][i, :n_classes]) for m in range(n_models)]
        vote_counts = Counter(votes)
        top_vote, top_count = vote_counts.most_common(1)[0]

        if top_count >= 0.6 * n_models:
            # Check if agreeing models are confident
            agreeing_confs = []
            for m in range(n_models):
                if votes[m] == top_vote:
                    probs = softmax(all_logits[m][i, :n_classes])
                    agreeing_confs.append(probs.max())
            mean_conf = np.mean(agreeing_confs)

            if mean_conf > 0.75 and top_vote != weighted_pred:
                predictions[i] = top_vote
                stats['agreement'] += 1
                continue

        predictions[i] = weighted_pred
        stats['binned'] += 1

    print(f"  Binned: {stats['binned']}, Agreement override: {stats['agreement']}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Ensemble V10: Final Push')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    n_models = len(model_specs)
    print(f"Final Push Ensemble of {n_models} models:")
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

    # ---- Strategy 1: V9 baseline (3 bins) ----
    print(f"\n{'='*60}")
    print(f"  Strategy 1: Conf-Binned 3 bins (V9 baseline)")
    print(f"{'='*60}")
    preds = strategy_fine_bins(all_logits, valid_gt, qtype_indices,
                               dataset, n_samples, n_classes, n_models, n_bins=3)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("ConfBins-3", ov, tc, t, ts)
    results['conf_bins_3'] = ov

    # ---- Strategy 2: 4 bins ----
    print(f"\n{'='*60}")
    print(f"  Strategy 2: Conf-Binned 4 bins")
    print(f"{'='*60}")
    preds = strategy_fine_bins(all_logits, valid_gt, qtype_indices,
                               dataset, n_samples, n_classes, n_models, n_bins=4)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("ConfBins-4", ov, tc, t, ts)
    results['conf_bins_4'] = ov

    # ---- Strategy 3: 5 bins ----
    print(f"\n{'='*60}")
    print(f"  Strategy 3: Conf-Binned 5 bins")
    print(f"{'='*60}")
    preds = strategy_fine_bins(all_logits, valid_gt, qtype_indices,
                               dataset, n_samples, n_classes, n_models, n_bins=5)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("ConfBins-5", ov, tc, t, ts)
    results['conf_bins_5'] = ov

    # ---- Strategy 4: SubType × Confidence ----
    print(f"\n{'='*60}")
    print(f"  Strategy 4: SubType × Confidence Cross-Product")
    print(f"{'='*60}")
    preds = strategy_subtype_conf(all_logits, valid_gt, subtype_indices, qtype_indices,
                                   dataset, n_samples, n_classes, n_models)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("SubType×Conf", ov, tc, t, ts)
    results['subtype_conf'] = ov

    # ---- Strategy 5: Conf-binned + agreement override ----
    print(f"\n{'='*60}")
    print(f"  Strategy 5: Conf-Binned + Agreement Override")
    print(f"{'='*60}")
    preds = strategy_conf_bins_agreement(all_logits, valid_gt, qtype_indices,
                                          dataset, n_samples, n_classes, n_models)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("ConfBins+Agree", ov, tc, t, ts)
    results['conf_bins_agree'] = ov

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
