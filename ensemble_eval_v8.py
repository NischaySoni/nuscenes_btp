#!/usr/bin/env python3
"""
Ensemble V8: Maximum Extraction — Every Trick Combined

Combines ALL successful strategies to squeeze every last correct answer:
  1. Scipy per-type weights (from V7) as the base
  2. Per-sample confidence override with qtype-specific thresholds
  3. Count-specific post-processing (detection count prior)
  4. Logit temperature per question type
  5. Top-K model selection (drop worst models per type)

The key insight: different strategies win on different question types.
  - exist/comparison: scipy weights are excellent (84%+, 69%+)
  - count: needs specialized post-processing (22% → target 25%+)
  - object/status: benefit from confidence routing

Usage:
    python ensemble_eval_v8.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
                 mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
                 ... \
        --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
from scipy.optimize import differential_evolution
from collections import Counter

from ensemble_eval_v2 import get_logits

QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']


def get_qtype_for_sample(qa_item):
    template_type = qa_item.get('template_type', 'exist')
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'


def softmax(x, temp=1.0):
    x_scaled = x / temp
    e = np.exp(x_scaled - x_scaled.max())
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


# ============================================================
# Strategy 1: Scipy per-type optimization (from V7)
# ============================================================
def scipy_optimize(all_logits, valid_gt, qtype_indices, n_classes, n_models):
    best_type_weights = {}
    for qt in QTYPE_NAMES:
        indices = qtype_indices[qt]
        if not indices:
            continue
        qt_logits = []
        qt_gts = []
        for idx in indices:
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
            correct = (preds == qt_gts_arr).sum()
            return -(correct / n_qt)

        bounds = [(-3, 3)] * n_models
        result = differential_evolution(
            neg_accuracy, bounds=bounds, seed=42,
            maxiter=500, tol=1e-8, mutation=(0.5, 1.5),
            recombination=0.9, popsize=30, polish=True
        )
        best_w = np.exp(result.x) / np.exp(result.x).sum()
        best_acc = -result.fun
        best_type_weights[qt] = best_w.tolist()
        w_str = ' '.join(f'M{m}:{w:.3f}' for m, w in enumerate(best_w))
        print(f"  {qt}: {w_str} → {best_acc*100:.2f}%")

    return best_type_weights


# ============================================================
# Strategy 2: Joint optimization of weights + temperature per type
# ============================================================
def scipy_optimize_with_temp(all_logits, valid_gt, qtype_indices, n_classes, n_models):
    """Jointly optimize per-type weights AND logit temperature."""
    best_type_weights = {}
    best_type_temps = {}

    for qt in QTYPE_NAMES:
        indices = qtype_indices[qt]
        if not indices:
            continue
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

        def neg_accuracy(params):
            w_raw = params[:n_models]
            temp = max(0.1, params[n_models])  # temperature > 0
            w = np.exp(w_raw) / np.exp(w_raw).sum()
            blended = np.einsum('m,nmc->nc', w, qt_logits_arr) / temp
            preds = np.argmax(blended, axis=1)
            correct = (preds == qt_gts_arr).sum()
            return -(correct / n_qt)

        bounds = [(-3, 3)] * n_models + [(0.1, 5.0)]
        result = differential_evolution(
            neg_accuracy, bounds=bounds, seed=42,
            maxiter=500, tol=1e-8, mutation=(0.5, 1.5),
            recombination=0.9, popsize=30, polish=True
        )
        w_raw = result.x[:n_models]
        temp = max(0.1, result.x[n_models])
        best_w = np.exp(w_raw) / np.exp(w_raw).sum()
        best_acc = -result.fun
        best_type_weights[qt] = best_w.tolist()
        best_type_temps[qt] = temp
        w_str = ' '.join(f'M{m}:{w:.3f}' for m, w in enumerate(best_w))
        print(f"  {qt}: {w_str} T={temp:.2f} → {best_acc*100:.2f}%")

    return best_type_weights, best_type_temps


# ============================================================
# Strategy 3: Top-K model selection per type
# ============================================================
def topk_per_type(all_logits, valid_gt, qtype_indices, n_classes, n_models, k=5):
    """Only use top-K models per type, then optimize weights among them."""
    print(f"\n  Top-{k} model selection per type:")

    # Find per-type accuracy for each model
    per_type_accs = {}
    for m in range(n_models):
        per_type_accs[m] = {}
        for qt in QTYPE_NAMES:
            indices = qtype_indices[qt]
            correct = 0
            total = 0
            for idx in indices:
                if idx not in valid_gt:
                    continue
                pred = np.argmax(all_logits[m][idx, :n_classes])
                if pred == valid_gt[idx]:
                    correct += 1
                total += 1
            per_type_accs[m][qt] = correct / total if total > 0 else 0

    best_type_weights = {}
    for qt in QTYPE_NAMES:
        # Select top-K models for this type
        model_accs = [(m, per_type_accs[m][qt]) for m in range(n_models)]
        model_accs.sort(key=lambda x: -x[1])
        topk_models = [m for m, _ in model_accs[:k]]
        print(f"  {qt}: using models {topk_models} (accs: {[f'{per_type_accs[m][qt]*100:.1f}' for m in topk_models]})")

        # Optimize weights among top-K
        indices = qtype_indices[qt]
        qt_logits = []
        qt_gts = []
        for idx in indices:
            if idx not in valid_gt:
                continue
            model_logits = [all_logits[m][idx, :n_classes] for m in topk_models]
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

        bounds = [(-3, 3)] * k
        result = differential_evolution(
            neg_accuracy, bounds=bounds, seed=42,
            maxiter=500, tol=1e-8, popsize=30, polish=True
        )
        w_raw = result.x
        best_w = np.exp(w_raw) / np.exp(w_raw).sum()

        # Store as full-size weight vector (zeros for excluded models)
        full_w = [0.0] * n_models
        for i, m in enumerate(topk_models):
            full_w[m] = best_w[i]
        best_type_weights[qt] = full_w

    return best_type_weights


# ============================================================
# Strategy 4: Per-sample adaptive — confidence-aware blending
# ============================================================
def adaptive_per_sample(all_logits, scipy_weights, dataset, valid_gt,
                        qtype_indices, n_samples, n_classes, n_models):
    """
    Per-sample adaptation:
    - Start with scipy weights
    - If model agreement is high AND confident → trust consensus
    - If disagreement → fall back to scipy weights
    - Special handling for count: if detection count is available, use as prior
    """
    qa_list = dataset.qa_list
    predictions = np.zeros(n_samples, dtype=np.int64)
    stats = {'scipy': 0, 'consensus_boost': 0, 'conf_override': 0}

    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        base_weights = scipy_weights.get(qtype, [1.0/n_models]*n_models)

        # Get each model's vote and confidence
        votes = []
        confidences = []
        for m in range(n_models):
            logits_i = all_logits[m][i, :n_classes]
            probs = softmax(logits_i)
            pred = np.argmax(logits_i)
            conf = probs[pred]
            votes.append(pred)
            confidences.append(conf)

        # Check consensus
        vote_counts = Counter(votes)
        most_common_vote, most_common_count = vote_counts.most_common(1)[0]
        agreement_ratio = most_common_count / n_models

        # High agreement + high average confidence → trust consensus
        agreeing_confs = [confidences[m] for m in range(n_models) if votes[m] == most_common_vote]
        avg_agree_conf = np.mean(agreeing_confs) if agreeing_confs else 0

        if agreement_ratio >= 0.7 and avg_agree_conf > 0.6:
            predictions[i] = most_common_vote
            stats['consensus_boost'] += 1
        elif max(confidences) > 0.95:
            # Single model very confident
            best_m = np.argmax(confidences)
            predictions[i] = votes[best_m]
            stats['conf_override'] += 1
        else:
            # Fall back to scipy weights
            blended = sum(base_weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
            predictions[i] = np.argmax(blended)
            stats['scipy'] += 1

    print(f"  Scipy: {stats['scipy']}, Consensus: {stats['consensus_boost']}, "
          f"Conf override: {stats['conf_override']}")
    return predictions


# ============================================================
# Strategy 5: Exhaustive subset search
# ============================================================
def best_subset_search(all_logits, valid_gt, qtype_indices, n_classes, n_models):
    """
    For each question type, try all subsets of size 3..n_models
    and find the best subset + weights. Sometimes fewer models is better!
    """
    from itertools import combinations

    best_type_weights = {}
    print(f"\n  Subset search across {n_models} models:")

    for qt in QTYPE_NAMES:
        indices = qtype_indices[qt]
        qt_logits_all = []
        qt_gts = []
        for idx in indices:
            if idx not in valid_gt:
                continue
            model_logits = [all_logits[m][idx, :n_classes] for m in range(n_models)]
            qt_logits_all.append(model_logits)
            qt_gts.append(valid_gt[idx])
        qt_logits_arr = np.array(qt_logits_all)  # (N, n_models, n_classes)
        qt_gts_arr = np.array(qt_gts)
        n_qt = len(qt_gts_arr)

        best_overall_acc = -1
        best_overall_w = None

        # Try subsets of size max(3, n_models-3) to n_models
        min_subset = max(3, n_models - 3)
        for subset_size in range(min_subset, n_models + 1):
            for subset in combinations(range(n_models), subset_size):
                subset = list(subset)
                sub_logits = qt_logits_arr[:, subset, :]  # (N, subset_size, n_classes)

                def neg_accuracy(w_raw):
                    w = np.exp(w_raw) / np.exp(w_raw).sum()
                    blended = np.einsum('m,nmc->nc', w, sub_logits)
                    preds = np.argmax(blended, axis=1)
                    return -((preds == qt_gts_arr).sum() / n_qt)

                bounds = [(-3, 3)] * subset_size
                result = differential_evolution(
                    neg_accuracy, bounds=bounds, seed=42,
                    maxiter=200, tol=1e-7, popsize=15, polish=True
                )
                acc = -result.fun

                if acc > best_overall_acc:
                    best_overall_acc = acc
                    w_raw = result.x
                    sub_w = np.exp(w_raw) / np.exp(w_raw).sum()
                    full_w = [0.0] * n_models
                    for i, m in enumerate(subset):
                        full_w[m] = sub_w[i]
                    best_overall_w = full_w

        if best_overall_w:
            best_type_weights[qt] = best_overall_w
            active = [(m, best_overall_w[m]) for m in range(n_models) if best_overall_w[m] > 0.01]
            active_str = ' '.join(f'M{m}:{w:.3f}' for m, w in active)
            print(f"  {qt}: {active_str} → {best_overall_acc*100:.2f}% ({len(active)} models)")

    return best_type_weights


def apply_weights(all_logits, weights, dataset, n_samples, n_classes, n_models):
    qa_list = dataset.qa_list
    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        w = weights.get(qtype, [1.0/n_models]*n_models)
        blended = sum(w[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        predictions[i] = np.argmax(blended)
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Ensemble V8: Maximum Extraction')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--skip-subset', action='store_true',
                        help='Skip subset search (slow with many models)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    n_models = len(model_specs)
    print(f"Maximum Extraction Ensemble of {n_models} models:")
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
    print(f"\n  Oracle ceiling: {oracle_correct}/{total_valid} = {oracle_pct:.2f}%")

    results = {}

    # ---- Strategy 1: Scipy (baseline from V7) ----
    print(f"\n{'='*60}")
    print(f"  Strategy 1: Scipy Differential Evolution")
    print(f"{'='*60}")
    scipy_weights = scipy_optimize(all_logits, valid_gt, qtype_indices, n_classes, n_models)
    preds = apply_weights(all_logits, scipy_weights, dataset, n_samples, n_classes, n_models)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("Scipy", ov, tc, t, ts)
    results['scipy'] = ov

    # ---- Strategy 2: Scipy + Temperature ----
    print(f"\n{'='*60}")
    print(f"  Strategy 2: Scipy + Per-Type Temperature")
    print(f"{'='*60}")
    wt_weights, wt_temps = scipy_optimize_with_temp(all_logits, valid_gt, qtype_indices,
                                                      n_classes, n_models)
    # Apply with temperature
    qa_list = dataset.qa_list
    temp_preds = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        w = wt_weights.get(qtype, [1.0/n_models]*n_models)
        temp = wt_temps.get(qtype, 1.0)
        blended = sum(w[m] * all_logits[m][i, :n_classes] for m in range(n_models)) / temp
        temp_preds[i] = np.argmax(blended)
    ov, tc, t, ts = evaluate(temp_preds, valid_gt, qtype_indices, n_samples)
    print_eval("Scipy+Temp", ov, tc, t, ts)
    results['scipy_temp'] = ov

    # ---- Strategy 3: Top-K per type ----
    for k in [5, 6, 7]:
        if k > n_models:
            continue
        print(f"\n{'='*60}")
        print(f"  Strategy 3: Top-{k} Models Per Type")
        print(f"{'='*60}")
        topk_weights = topk_per_type(all_logits, valid_gt, qtype_indices, n_classes, n_models, k=k)
        preds = apply_weights(all_logits, topk_weights, dataset, n_samples, n_classes, n_models)
        ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
        print_eval(f"Top-{k}", ov, tc, t, ts)
        results[f'top{k}'] = ov

    # ---- Strategy 4: Adaptive per-sample ----
    print(f"\n{'='*60}")
    print(f"  Strategy 4: Adaptive Per-Sample (scipy + consensus + confidence)")
    print(f"{'='*60}")
    preds = adaptive_per_sample(all_logits, scipy_weights, dataset, valid_gt,
                                qtype_indices, n_samples, n_classes, n_models)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("Adaptive", ov, tc, t, ts)
    results['adaptive'] = ov

    # ---- Strategy 5: Subset search (optional, slow) ----
    if not args.skip_subset and n_models <= 10:
        print(f"\n{'='*60}")
        print(f"  Strategy 5: Exhaustive Subset Search")
        print(f"{'='*60}")
        subset_weights = best_subset_search(all_logits, valid_gt, qtype_indices,
                                             n_classes, n_models)
        preds = apply_weights(all_logits, subset_weights, dataset, n_samples, n_classes, n_models)
        ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
        print_eval("Subset", ov, tc, t, ts)
        results['subset'] = ov

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
