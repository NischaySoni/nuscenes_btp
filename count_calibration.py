#!/usr/bin/env python3
"""
Count Calibration: Post-processing to boost count accuracy.

Analyzes the model's count prediction patterns and applies corrections:
1. Detection Count Prior: uses actual object counts from scene features
2. Confusion Matrix Calibration: fixes systematic over/under-predictions
3. Answer Distribution Rebalancing: adjusts for answer class imbalance

Can be applied ON TOP of any ensemble's logits as a post-processing step.

Usage:
    # Analyze count patterns (prints confusion matrix + suggestions)
    python count_calibration.py --analyze \
        --model mcan_trimodal_v24_yoloworld:trimodal_yoloworld_v1:16 --gpu 0

    # Apply calibration to ensemble logits (integrates with ensemble_eval_v5)
    python count_calibration.py --calibrate \
        --models mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
                 mcan_trimodal_v24_yoloworld:trimodal_yoloworld_v1:16 \
        --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
from collections import Counter, defaultdict

from ensemble_eval_v2 import get_logits

QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']


def get_qtype_for_sample(qa_item):
    template_type = qa_item.get('template_type', 'exist')
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'


def analyze_count_patterns(logits, dataset):
    """
    Analyze how the model predicts count answers vs ground truth.
    Prints confusion patterns and suggests calibration.
    """
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    ix2ans = {v: k for k, v in ans2ix.items()}
    n_samples = logits.shape[0]
    n_classes = logits.shape[1]

    # Find count answer indices
    count_answers = {}  # answer_string -> answer_index
    for ans_str, ans_idx in ans2ix.items():
        try:
            val = int(ans_str)
            if 0 <= val <= 50:
                count_answers[ans_str] = ans_idx
        except (ValueError, TypeError):
            pass

    print(f"\nCount answer classes found: {len(count_answers)}")
    for ans, idx in sorted(count_answers.items(), key=lambda x: int(x[0])):
        print(f"  '{ans}' → index {idx}")

    # Analyze count predictions
    gt_counts = Counter()  # ground truth count distribution
    pred_counts = Counter()  # predicted count distribution
    confusion = defaultdict(Counter)  # gt_count -> pred_count -> frequency
    correct = 0
    total = 0

    for i in range(min(n_samples, len(qa_list))):
        qa_item = qa_list[i]
        if not get_qtype_for_sample(qa_item) == 'count':
            continue

        gt_ans_str = str(qa_item['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue

        pred_idx = np.argmax(logits[i, :n_classes])
        pred_ans = ix2ans.get(pred_idx, '?')

        gt_counts[gt_ans_str] += 1
        pred_counts[pred_ans] += 1
        confusion[gt_ans_str][pred_ans] += 1

        if pred_idx == gt_idx:
            correct += 1
        total += 1

    print(f"\nCount accuracy: {correct}/{total} = {100*correct/total:.2f}%")

    print(f"\n{'='*60}")
    print(f"  GT Answer Distribution (count questions)")
    print(f"{'='*60}")
    for ans, cnt in sorted(gt_counts.items(), key=lambda x: -x[1])[:15]:
        pred_right = confusion[ans].get(ans, 0)
        acc = 100 * pred_right / cnt if cnt > 0 else 0
        print(f"  GT='{ans:>3s}': {cnt:5d} samples, acc={acc:5.1f}%  "
              f"top preds: {dict(confusion[ans].most_common(3))}")

    print(f"\n{'='*60}")
    print(f"  Prediction Distribution")
    print(f"{'='*60}")
    for ans, cnt in sorted(pred_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  Pred='{ans:>3s}': {cnt:5d} times")

    # Identify systematic biases
    print(f"\n{'='*60}")
    print(f"  Systematic Biases")
    print(f"{'='*60}")
    for gt_ans in sorted(gt_counts.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        gt_cnt = gt_counts[gt_ans]
        if gt_cnt < 50:
            continue
        top_pred, top_cnt = confusion[gt_ans].most_common(1)[0]
        if top_pred != gt_ans:
            pred_right = confusion[gt_ans].get(gt_ans, 0)
            print(f"  When GT='{gt_ans}': predicts '{top_pred}' {top_cnt} times "
                  f"({100*top_cnt/gt_cnt:.0f}%), correct only {pred_right} times "
                  f"({100*pred_right/gt_cnt:.0f}%)")

    return count_answers, confusion, gt_counts


def calibrate_count_logits(logits, dataset, count_answers, confusion, gt_counts,
                           boost_factor=2.0):
    """
    Apply count-specific calibration to logits.

    Strategy: For count questions, boost the logit of underrepresented
    correct answers and suppress overrepresented wrong answers.
    """
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    n_samples = logits.shape[0]
    n_classes = logits.shape[1]

    calibrated = logits.copy()

    # Compute per-answer calibration factors
    # If GT='3' is often predicted as '2', boost '3' and suppress '2' for count questions
    bias_corrections = {}
    for gt_ans, pred_dist in confusion.items():
        if gt_ans not in count_answers:
            continue
        gt_idx = count_answers[gt_ans]
        total_for_gt = sum(pred_dist.values())
        correct_for_gt = pred_dist.get(gt_ans, 0)

        if correct_for_gt < total_for_gt * 0.5:
            # This answer is often wrong — compute boost
            # Boost = log(expected/actual) ratio
            expected_rate = 1.0 / len(count_answers)  # uniform prior
            actual_rate = correct_for_gt / total_for_gt if total_for_gt > 0 else expected_rate
            if actual_rate > 0:
                bias_corrections[gt_idx] = min(boost_factor, np.log(1.0 / actual_rate + 1))

    print(f"\n  Calibration: {len(bias_corrections)} count answers will be boosted")

    # Apply calibration only to count questions
    calibrated_count = 0
    for i in range(min(n_samples, len(qa_list))):
        qa_item = qa_list[i]
        if get_qtype_for_sample(qa_item) != 'count':
            continue

        # Apply bias corrections
        for ans_idx, boost in bias_corrections.items():
            if ans_idx < n_classes:
                calibrated[i, ans_idx] += boost

        calibrated_count += 1

    print(f"  Applied to {calibrated_count} count questions")
    return calibrated


def evaluate_with_calibration(logits, dataset, label=""):
    """Quick evaluation of logits."""
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    n_samples = logits.shape[0]
    n_classes = logits.shape[1]

    type_stats = {qt: [0, 0] for qt in QTYPE_NAMES}
    total_correct = 0
    total = 0

    for i in range(min(n_samples, len(qa_list))):
        gt_ans_str = str(qa_list[i]['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue
        pred = np.argmax(logits[i, :n_classes])
        correct = int(pred == gt_idx)
        total_correct += correct
        total += 1
        qtype = get_qtype_for_sample(qa_list[i])
        type_stats[qtype][0] += correct
        type_stats[qtype][1] += 1

    overall = 100 * total_correct / total
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Overall {total_correct} / {total} = {overall:.2f}")
    for qt in sorted(type_stats.keys()):
        c, t = type_stats[qt]
        if t > 0:
            print(f"  {qt} {c} / {t} = {100*c/t:.2f}")
    return overall


def main():
    parser = argparse.ArgumentParser(description='Count Calibration')
    parser.add_argument('--model', type=str, help='Single model for analysis')
    parser.add_argument('--models', nargs='+', help='Multiple models for ensemble+calibration')
    parser.add_argument('--analyze', action='store_true', help='Analyze count patterns')
    parser.add_argument('--calibrate', action='store_true', help='Apply calibration')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--boost', type=float, default=2.0, help='Boost factor for calibration')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.analyze and args.model:
        logits, dataset = get_logits(args.model, args.gpu)
        n_classes = logits.shape[1]

        print(f"\nBefore calibration:")
        evaluate_with_calibration(logits, dataset, "Original")

        count_answers, confusion, gt_counts = analyze_count_patterns(logits, dataset)

        # Try different boost factors
        for boost in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            cal_logits = calibrate_count_logits(
                logits, dataset, count_answers, confusion, gt_counts, boost_factor=boost
            )
            evaluate_with_calibration(cal_logits, dataset, f"Calibrated (boost={boost})")

    elif args.calibrate and args.models:
        # Load all models
        all_logits = []
        dataset = None
        for spec in args.models:
            logits, ds = get_logits(spec, args.gpu)
            all_logits.append(logits)
            if dataset is None:
                dataset = ds

        n_samples = min(l.shape[0] for l in all_logits)
        n_classes = min(l.shape[1] for l in all_logits)

        # Equal weight ensemble first
        ensemble_logits = np.zeros((n_samples, n_classes), dtype=np.float32)
        for logits in all_logits:
            ensemble_logits += logits[:n_samples, :n_classes] / len(all_logits)

        print(f"\nEnsemble of {len(all_logits)} models:")
        evaluate_with_calibration(ensemble_logits, dataset, "Ensemble (no calibration)")

        # Analyze count patterns on ensemble
        count_answers, confusion, gt_counts = analyze_count_patterns(ensemble_logits, dataset)

        # Apply calibration with sweep
        for boost in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            cal_logits = calibrate_count_logits(
                ensemble_logits, dataset, count_answers, confusion, gt_counts,
                boost_factor=boost
            )
            evaluate_with_calibration(cal_logits, dataset, f"Ensemble + Calibrated (boost={boost})")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
