#!/usr/bin/env python3
"""
Ensemble V5: Agreement-Aware Router + More Models

Key insight from V4 oracle analysis: 67.48% ceiling with just 3 models.
The gap from 58.4% → 67.5% means ~9% of samples have the right answer
in at least one model but we're picking the wrong one.

Strategy: Per-sample routing based on model agreement & confidence.
  - When 2+ models agree → use the consensus answer (high reliability)
  - When all disagree → use confidence-weighted selection
  - Also: run with MORE models to raise the oracle ceiling

Usage:
    python ensemble_eval_v5.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
                 mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
                 mcan_trimodal_v16_bert_deep:trimodal_bert_deep_v1:16 \
                 mcan_trimodal_v18_bert_base:trimodal_bert_base_v1:15 \
                 mcan_trimodal_v24_yoloworld:trimodal_yoloworld_v1:16 \
        --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
from collections import Counter

from ensemble_eval_v2 import get_logits

QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']


def get_qtype_for_sample(qa_item):
    template_type = qa_item.get('template_type', 'exist')
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'


def build_indices(dataset):
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    qtype_indices = {qt: set() for qt in QTYPE_NAMES}
    valid_gt = {}
    for i in range(len(qa_list)):
        gt_ans_str = str(qa_list[i]['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue
        qtype = get_qtype_for_sample(qa_list[i])
        qtype_indices[qtype].add(i)
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
            if i in qtype_indices[qt]:
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


def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


# ============================================================
# Strategy: Agreement-Aware Router
# ============================================================
def agreement_router(all_logits, per_type_accs, dataset, n_samples, n_classes,
                     temp=2.0, conf_boost=1.5):
    """
    Smart per-sample routing:
    1. Each model votes (argmax of its logits)
    2. If majority agrees → use that answer
    3. If no majority → blend logits with confidence-boosted weights
    """
    n_models = len(all_logits)
    qa_list = dataset.qa_list
    predictions = np.zeros(n_samples, dtype=np.int64)

    # Pre-compute qtype weights
    type_weights = {}
    for qt in QTYPE_NAMES:
        accs = np.array([per_type_accs[m][qt] for m in range(n_models)])
        exp_accs = np.exp((accs - accs.max()) * temp)
        weights = exp_accs / exp_accs.sum()
        type_weights[qt] = weights

    stats = {'agree': 0, 'disagree': 0}

    for i in range(min(n_samples, len(qa_list))):
        # Get each model's vote
        votes = []
        confidences = []
        for m in range(n_models):
            logits_i = all_logits[m][i, :n_classes]
            pred = np.argmax(logits_i)
            probs = softmax(logits_i)
            conf = probs[pred]
            votes.append(pred)
            confidences.append(conf)

        # Check for majority agreement
        vote_counts = Counter(votes)
        most_common_vote, most_common_count = vote_counts.most_common(1)[0]

        if most_common_count > n_models / 2:
            # Majority agrees → use consensus
            predictions[i] = most_common_vote
            stats['agree'] += 1
        else:
            # No majority → use confidence-boosted qtype routing
            qtype = get_qtype_for_sample(qa_list[i])
            qw = type_weights.get(qtype, np.ones(n_models) / n_models)

            # Boost weights by per-sample confidence
            conf_arr = np.array(confidences)
            conf_weights = conf_arr / conf_arr.sum()

            # Combine: qtype routing + confidence boost
            final_w = (1 - 0.3) * qw + 0.3 * conf_weights
            final_w = final_w / final_w.sum()

            blended = np.zeros(n_classes, dtype=np.float32)
            for m in range(n_models):
                blended += final_w[m] * all_logits[m][i, :n_classes]
            predictions[i] = np.argmax(blended)
            stats['disagree'] += 1

    print(f"  Agreement: {stats['agree']} ({100*stats['agree']/n_samples:.1f}%)")
    print(f"  Disagreement: {stats['disagree']} ({100*stats['disagree']/n_samples:.1f}%)")
    return predictions


# ============================================================
# Strategy: Confidence-Gated Router
# ============================================================
def confidence_gated(all_logits, per_type_accs, dataset, n_samples, n_classes,
                     conf_threshold=0.7):
    """
    If ANY model has confidence > threshold, trust that model alone.
    Otherwise, fall back to qtype_routed weighted average.
    """
    n_models = len(all_logits)
    qa_list = dataset.qa_list
    predictions = np.zeros(n_samples, dtype=np.int64)

    type_weights = {}
    for qt in QTYPE_NAMES:
        accs = np.array([per_type_accs[m][qt] for m in range(n_models)])
        exp_accs = np.exp((accs - accs.max()) * 2.0)
        weights = exp_accs / exp_accs.sum()
        type_weights[qt] = weights

    stats = {'gated': 0, 'blended': 0}

    for i in range(min(n_samples, len(qa_list))):
        # Check if any model is very confident
        best_conf = -1
        best_model = -1
        for m in range(n_models):
            logits_i = all_logits[m][i, :n_classes]
            probs = softmax(logits_i)
            conf = probs.max()
            if conf > best_conf:
                best_conf = conf
                best_model = m

        if best_conf > conf_threshold:
            predictions[i] = np.argmax(all_logits[best_model][i, :n_classes])
            stats['gated'] += 1
        else:
            qtype = get_qtype_for_sample(qa_list[i])
            weights = type_weights.get(qtype, np.ones(n_models) / n_models)
            blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
            predictions[i] = np.argmax(blended)
            stats['blended'] += 1

    print(f"  Gated (conf>{conf_threshold}): {stats['gated']} ({100*stats['gated']/n_samples:.1f}%)")
    print(f"  Blended: {stats['blended']} ({100*stats['blended']/n_samples:.1f}%)")
    return predictions


# ============================================================
# Strategy: Margin-Based Router
# ============================================================
def margin_router(all_logits, per_type_accs, dataset, n_samples, n_classes):
    """
    For each sample, pick the model with the largest margin
    between its top-1 and top-2 predictions (most decisive model).
    """
    n_models = len(all_logits)
    predictions = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        best_margin = -1e9
        best_pred = 0
        for m in range(n_models):
            logits_i = all_logits[m][i, :n_classes]
            sorted_l = np.sort(logits_i)[::-1]
            margin = sorted_l[0] - sorted_l[1]
            if margin > best_margin:
                best_margin = margin
                best_pred = np.argmax(logits_i)
        predictions[i] = best_pred

    return predictions


# ============================================================
# Strategy: Rank-Based Fusion
# ============================================================
def rank_fusion(all_logits, n_samples, n_classes):
    """
    Borda count / rank fusion: rank all answers by each model,
    then sum ranks. More robust to logit scale differences.
    """
    n_models = len(all_logits)
    rank_scores = np.zeros((n_samples, n_classes), dtype=np.float32)

    for m in range(n_models):
        for i in range(n_samples):
            logits_i = all_logits[m][i, :n_classes]
            # Rank: higher logit = higher rank score
            order = np.argsort(logits_i)
            for rank, idx in enumerate(order):
                rank_scores[i, idx] += rank

    return np.argmax(rank_scores, axis=1)


def main():
    parser = argparse.ArgumentParser(description='Ensemble V5: Agreement Router')
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

    qtype_indices, valid_gt = build_indices(dataset)

    # Individual models
    per_type_accs = {}
    for m, (spec, logits) in enumerate(zip(model_specs, all_logits)):
        preds = np.argmax(logits[:n_samples, :n_classes], axis=1)
        overall, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
        per_type_accs[m] = {}
        for qt in QTYPE_NAMES:
            per_type_accs[m][qt] = 100 * ts[qt][0] / ts[qt][1] if ts[qt][1] > 0 else 0
        name = spec.split(':')[1] if ':' in spec else spec
        pt = ' '.join(f'{qt}={per_type_accs[m][qt]:.1f}' for qt in QTYPE_NAMES)
        print(f"  Model {m} ({name}): {pt} | Overall={overall:.2f}%")

    # Oracle
    oracle_preds = np.zeros(n_samples, dtype=np.int64)
    oracle_correct = 0
    for i in range(n_samples):
        if i in valid_gt:
            for m in range(len(all_logits)):
                if np.argmax(all_logits[m][i, :n_classes]) == valid_gt[i]:
                    oracle_preds[i] = valid_gt[i]
                    oracle_correct += 1
                    break
            else:
                oracle_preds[i] = np.argmax(all_logits[0][i, :n_classes])
    oracle_ov, _, _, oracle_ts = evaluate(oracle_preds, valid_gt, qtype_indices, n_samples)
    print_eval("Oracle Ceiling", oracle_ov, oracle_correct,
               sum(1 for i in range(n_samples) if i in valid_gt), oracle_ts)

    results = {}

    # ---- Strategy 1: Agreement Router ----
    print(f"\n{'='*60}")
    print("  Strategy: Agreement Router")
    print(f"{'='*60}")
    preds = agreement_router(all_logits, per_type_accs, dataset, n_samples, n_classes)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("Agreement Router", ov, tc, t, ts)
    results['agreement'] = ov

    # ---- Strategy 2: Confidence-Gated ----
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"\n{'='*60}")
        print(f"  Strategy: Confidence-Gated (threshold={thresh})")
        print(f"{'='*60}")
        preds = confidence_gated(all_logits, per_type_accs, dataset, n_samples, n_classes,
                                 conf_threshold=thresh)
        ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
        print_eval(f"Conf-Gated-{thresh}", ov, tc, t, ts)
        key = f'conf_gated_{thresh}'
        results[key] = ov

    # ---- Strategy 3: Margin Router ----
    print(f"\n{'='*60}")
    print("  Strategy: Margin Router")
    print(f"{'='*60}")
    preds = margin_router(all_logits, per_type_accs, dataset, n_samples, n_classes)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("Margin Router", ov, tc, t, ts)
    results['margin'] = ov

    # ---- Strategy 4: Rank Fusion (Borda) ----
    print(f"\n{'='*60}")
    print("  Strategy: Rank Fusion (Borda Count)")
    print(f"{'='*60}")
    preds = rank_fusion(all_logits, n_samples, n_classes)
    ov, tc, t, ts = evaluate(preds, valid_gt, qtype_indices, n_samples)
    print_eval("Rank Fusion", ov, tc, t, ts)
    results['rank_fusion'] = ov

    # ---- Strategy 5: Grid Search (from V4, reimplemented) ----
    print(f"\n{'='*60}")
    print("  Strategy: Grid Search (per-type optimal weights)")
    print(f"{'='*60}")
    n_models = len(all_logits)
    qa_list = dataset.qa_list
    step = 0.1
    wvals = np.arange(0.0, 1.01, step)

    best_type_weights = {}
    for qt in QTYPE_NAMES:
        indices = list(qtype_indices[qt])
        if not indices:
            continue
        best_acc = -1
        best_w = None

        if n_models <= 3:
            for w0 in wvals:
                for w1 in wvals:
                    rest = 1.0 - w0 - w1
                    if n_models == 3 and (rest < -0.01 or rest > 1.01):
                        continue
                    if n_models == 2:
                        weights = [w0, 1.0 - w0]
                        if weights[1] < -0.01:
                            continue
                    else:
                        weights = [w0, w1, max(0, rest)]

                    correct = 0
                    total_valid = 0
                    for idx in indices:
                        if idx not in valid_gt:
                            continue
                        logits_i = sum(weights[m] * all_logits[m][idx, :n_classes] for m in range(n_models))
                        if np.argmax(logits_i) == valid_gt[idx]:
                            correct += 1
                        total_valid += 1
                    acc = correct / total_valid if total_valid > 0 else 0
                    if acc > best_acc:
                        best_acc = acc
                        best_w = list(weights)

                    if n_models == 2:
                        break  # inner loop not needed for 2 models
        else:
            # For 4+ models: random search
            np.random.seed(42)
            for _ in range(5000):
                raw = np.random.dirichlet(np.ones(n_models))
                weights = raw.tolist()
                correct = 0
                total_valid = 0
                for idx in indices:
                    if idx not in valid_gt:
                        continue
                    logits_i = sum(weights[m] * all_logits[m][idx, :n_classes] for m in range(n_models))
                    if np.argmax(logits_i) == valid_gt[idx]:
                        correct += 1
                    total_valid += 1
                acc = correct / total_valid if total_valid > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                    best_w = list(weights)

        if best_w:
            best_type_weights[qt] = best_w
            w_str = ' '.join(f'M{m}:{w:.2f}' for m, w in enumerate(best_w))
            print(f"  {qt}: {w_str} → {best_acc*100:.2f}%")

    # Apply best weights
    grid_preds = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        weights = best_type_weights.get(qtype, [1.0/n_models]*n_models)
        blended = sum(weights[m] * all_logits[m][i, :n_classes] for m in range(n_models))
        grid_preds[i] = np.argmax(blended)
    ov, tc, t, ts = evaluate(grid_preds, valid_gt, qtype_indices, n_samples)
    print_eval("Grid Search", ov, tc, t, ts)
    results['grid_search'] = ov

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY (Oracle ceiling: {oracle_ov:.2f}%)")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if acc == max(results.values()) else ""
        gap = oracle_ov - acc
        print(f"  {name:25s} = {acc:.2f}%  (gap to oracle: {gap:.2f}%){marker}")

    best_name = max(results.items(), key=lambda x: x[1])
    print(f"\n  🏆 BEST: {best_name[0]} = {best_name[1]:.2f}%")


if __name__ == '__main__':
    main()
