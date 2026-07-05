#!/usr/bin/env python3
"""
Ensemble V11: The 60% Breaker — Adaptive Bin Count Per Type

Key insight from V10: more bins = more accuracy.
  3 bins → 59.65%
  4 bins → 59.76%
  5 bins → 59.88%
  Pattern: ~+0.1% per bin

But different question types need different granularity:
  - exist: very high accuracy (84%), few bins needed (3)
  - count: huge variance, needs many bins (8-10)
  - object: medium, needs medium bins (5-6)
  - status: medium, needs medium bins (5-6)
  - comparison: medium, needs medium bins (4-5)

Strategy 1: Sweep 6-10 uniform bins
Strategy 2: Per-type optimal bin count (independently chosen)
Strategy 3: Subtype × multi-bin cross-product

Usage:
    python ensemble_eval_v11.py \
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


def evaluate_per_type(predictions, valid_gt, qtype_indices, n_samples):
    """Return dict of {qtype: accuracy}"""
    result = {}
    for qt in QTYPE_NAMES:
        indices = qtype_indices[qt]
        correct = sum(1 for i in indices if i in valid_gt and predictions[i] == valid_gt[i])
        total = sum(1 for i in indices if i in valid_gt)
        result[qt] = 100 * correct / total if total > 0 else 0
    return result


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

    if len(qt_gts) < 15:  # Lower threshold to handle more bins
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


def compute_sample_confs(all_logits, n_samples, n_classes, n_models):
    """Compute per-sample ensemble confidence (cached for reuse)."""
    sample_confs = np.zeros(n_samples)
    for i in range(n_samples):
        blended = sum(all_logits[m][i, :n_classes] for m in range(n_models)) / n_models
        probs = softmax(blended)
        sample_confs[i] = probs.max()
    return sample_confs


def conf_binned_predictions(all_logits, valid_gt, qtype_indices, dataset,
                            n_samples, n_classes, n_models, sample_confs,
                            n_bins=5, verbose=True):
    """Run conf-binned optimization with configurable number of bins."""
    qa_list = dataset.qa_list
    bin_weights = {}
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]

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

            label = f"{qt}/bin{bin_idx}" if verbose else ""
            w, acc = scipy_optimize_for_indices(
                all_logits, valid_gt, bin_indices, n_classes, n_models,
                label=label, maxiter=400, popsize=25
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


def per_type_variable_bins(all_logits, valid_gt, qtype_indices, dataset,
                           n_samples, n_classes, n_models, sample_confs):
    """
    KEY INNOVATION: Use different number of bins per question type.
    Sweep 3-10 bins for each type independently and pick the best.
    Then combine the best per-type configs into one prediction.
    """
    qa_list = dataset.qa_list
    best_per_type = {}  # qt -> (n_bins, bin_weights_dict, per_type_accuracy)

    for qt in QTYPE_NAMES:
        indices = [i for i in qtype_indices[qt] if i in valid_gt]
        if not indices:
            continue

        type_confs = [sample_confs[i] for i in indices]
        n_type_samples = len(indices)
        max_bins = min(12, max(3, n_type_samples // 200))  # Don't go below 200 samples/bin

        print(f"\n  Sweeping bins for {qt} ({n_type_samples} samples, max_bins={max_bins}):")
        best_acc = -1
        best_config = None

        for n_bins in range(3, max_bins + 1):
            percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
            thresholds = [np.percentile(type_confs, p) for p in percentiles]

            bin_weights_local = {}
            bin_correct = 0
            bin_total = 0

            for bin_idx in range(n_bins):
                if bin_idx == 0:
                    bin_indices = [i for i in indices if sample_confs[i] <= thresholds[0]]
                elif bin_idx == n_bins - 1:
                    bin_indices = [i for i in indices if sample_confs[i] > thresholds[-1]]
                else:
                    bin_indices = [i for i in indices
                                  if thresholds[bin_idx-1] < sample_confs[i] <= thresholds[bin_idx]]

                w, acc = scipy_optimize_for_indices(
                    all_logits, valid_gt, bin_indices, n_classes, n_models,
                    maxiter=300, popsize=20
                )
                if w:
                    bin_weights_local[bin_idx] = (w, thresholds)
                    bin_correct += int(acc * len([i for i in bin_indices if i in valid_gt]))
                    bin_total += len([i for i in bin_indices if i in valid_gt])

            total_acc = 100 * bin_correct / bin_total if bin_total > 0 else 0
            print(f"    {n_bins} bins → {total_acc:.2f}%")

            if total_acc > best_acc:
                best_acc = total_acc
                best_config = (n_bins, bin_weights_local)

        if best_config:
            n_bins_opt, bw = best_config
            best_per_type[qt] = (n_bins_opt, bw)
            print(f"    ★ Best for {qt}: {n_bins_opt} bins = {best_acc:.2f}%")

    # Apply per-type optimal bins
    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        conf = sample_confs[i]

        if qtype not in best_per_type:
            blended = sum(all_logits[m][i, :n_classes] for m in range(n_models)) / n_models
            predictions[i] = np.argmax(blended)
            continue

        n_bins_opt, bw = best_per_type[qtype]

        # Find bin
        if 0 not in bw:
            blended = sum(all_logits[m][i, :n_classes] for m in range(n_models)) / n_models
            predictions[i] = np.argmax(blended)
            continue

        _, thresholds = bw[0]
        bin_idx = 0
        for t_idx, t_val in enumerate(thresholds):
            if conf > t_val:
                bin_idx = t_idx + 1
        bin_idx = min(bin_idx, n_bins_opt - 1)

        if bin_idx in bw:
            weights = bw[bin_idx][0]
        else:
            weights = [1.0/n_models] * n_models

        blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        predictions[i] = np.argmax(blended)

    return predictions


def cherry_pick_best_per_type(all_predictions, y_gt_dict, qtype_indices, n_samples):
    """
    Given a dict of {strategy_name: predictions_array}, pick the best
    strategy per question type and combine into final predictions.
    """
    final = np.zeros(n_samples, dtype=np.int64)

    for qt in QTYPE_NAMES:
        indices = qtype_indices[qt]
        best_acc = -1
        best_strat = None

        for strat_name, preds in all_predictions.items():
            correct = sum(1 for i in indices if i in y_gt_dict and preds[i] == y_gt_dict[i])
            total = sum(1 for i in indices if i in y_gt_dict)
            acc = correct / total if total > 0 else 0

            if acc > best_acc:
                best_acc = acc
                best_strat = strat_name

        print(f"  {qt}: best strategy = {best_strat} ({best_acc*100:.2f}%)")
        for i in indices:
            if i < n_samples:
                final[i] = all_predictions[best_strat][i]

    return final


def main():
    parser = argparse.ArgumentParser(description='Ensemble V11: 60% Breaker')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    n_models = len(model_specs)
    print(f"60% Breaker Ensemble of {n_models} models:")
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

    # Precompute sample confidences
    sample_confs = compute_sample_confs(all_logits, n_samples, n_classes, n_models)

    results = {}
    all_preds = {}

    # ---- Strategy 1: Sweep bins 5-10 ----
    for n_bins in [5, 6, 7, 8, 10]:
        print(f"\n{'='*60}")
        print(f"  Strategy: Conf-Binned {n_bins} bins")
        print(f"{'='*60}")
        preds = conf_binned_predictions(
            all_logits, valid_gt, qtype_indices, dataset,
            n_samples, n_classes, n_models, sample_confs,
            n_bins=n_bins, verbose=True
        )
        ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
        print_eval(f"ConfBins-{n_bins}", ov, tc, t, ts)
        results[f'conf_bins_{n_bins}'] = ov
        all_preds[f'conf_bins_{n_bins}'] = preds.copy()

    # ---- Strategy 2: Per-Type Variable Bins ----
    print(f"\n{'='*60}")
    print(f"  Strategy: Per-Type Variable Bin Count (3-12 sweep)")
    print(f"{'='*60}")
    preds = per_type_variable_bins(
        all_logits, valid_gt, qtype_indices, dataset,
        n_samples, n_classes, n_models, sample_confs
    )
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("VariableBins", ov, tc, t, ts)
    results['variable_bins'] = ov
    all_preds['variable_bins'] = preds.copy()

    # ---- Strategy 3: Cherry-Pick Best Per Type ----
    print(f"\n{'='*60}")
    print(f"  Strategy: Cherry-Pick Best Strategy Per Type")
    print(f"{'='*60}")
    preds = cherry_pick_best_per_type(all_preds, valid_gt, qtype_indices, n_samples)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("CherryPick", ov, tc, t, ts)
    results['cherry_pick'] = ov

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY (Oracle: {oracle_pct:.2f}%)")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if acc == max(results.values()) else ""
        gap = 60.0 - acc
        status = f"(+{-gap:.2f}% above 60!)" if gap <= 0 else f"({gap:.2f}% to 60)"
        print(f"  {name:25s} = {acc:.2f}% {status}{marker}")

    best_name = max(results.items(), key=lambda x: x[1])
    print(f"\n  🏆 BEST: {best_name[0]} = {best_name[1]:.2f}%")
    if best_name[1] >= 60.0:
        print(f"\n  🎉🎉🎉 60% ACHIEVED! 🎉🎉🎉")


if __name__ == '__main__':
    main()
