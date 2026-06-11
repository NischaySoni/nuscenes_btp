#!/usr/bin/env python3
"""
Smart Ensemble V3: Question-Type-Routed Ensemble + Majority Voting.

Instead of averaging logits equally for all questions, this script:
  1. Routes each question to the best model(s) for that question type
  2. Supports multiple fusion strategies: weighted_avg, qtype_routed, majority_vote
  3. Automatically learns optimal per-type weights from model accuracies

Usage:
    python ensemble_eval_v3.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
                 mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
                 mcan_trimodal_v18_bert_base:trimodal_bert_base_v1:15 \
        --strategy qtype_routed \
        --gpu 0
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from src.models.mcan.model_cfgs import Cfgs
from src.models.model_loader import ModelLoader
from src.datasets.nuscenes_qa import NuScenes_QA
from src.execution.result_eval import Eval

# Reuse model loading infrastructure from ensemble_eval_v2
from ensemble_eval_v2 import load_config, get_logits


QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']

def get_qtype_for_sample(qa_item):
    """Extract base question type for a QA item."""
    template_type = qa_item.get('template_type', 'exist')
    # Strip hop suffix
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'  # fallback


def evaluate_per_type(dataset, predictions):
    """Evaluate predictions and return per-type accuracy dict."""
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix

    type_stats = {qt: {'correct': 0, 'total': 0} for qt in QTYPE_NAMES}
    total_correct = 0
    total = 0

    for i in range(min(len(predictions), len(qa_list))):
        pred_idx = predictions[i]
        qa_item = qa_list[i]
        gt_ans_str = str(qa_item['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue

        qtype = get_qtype_for_sample(qa_item)
        correct = int(pred_idx == gt_idx)
        total_correct += correct
        total += 1
        type_stats[qtype]['correct'] += correct
        type_stats[qtype]['total'] += 1

    return total_correct, total, type_stats


def print_results(label, total_correct, total, type_stats):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Overall {total_correct} / {total} = {100*total_correct/total:.2f}")
    for qt in sorted(type_stats.keys()):
        s = type_stats[qt]
        if s['total'] > 0:
            print(f"{qt} {s['correct']} / {s['total']} = {100*s['correct']/s['total']:.2f}")
    return 100 * total_correct / total


def strategy_qtype_routed(all_logits, per_type_accs, dataset, softmax_temp=2.0):
    """
    Question-type-routed ensemble: for each question type, weight models
    proportional to their accuracy on that type (softmax-weighted).
    
    This gives more influence to the model that's best at each type.
    """
    qa_list = dataset.qa_list
    n_samples = all_logits[0].shape[0]
    n_classes = all_logits[0].shape[1]
    n_models = len(all_logits)

    # Compute per-type weights via softmax over accuracies
    type_weights = {}
    for qt in QTYPE_NAMES:
        accs = np.array([per_type_accs[m][qt] for m in range(n_models)])
        # Softmax with temperature: lower temp = more peaked weights
        exp_accs = np.exp((accs - accs.max()) * softmax_temp)
        weights = exp_accs / exp_accs.sum()
        type_weights[qt] = weights
        
        # Print the weights
        weight_strs = [f"M{m}:{w:.3f}" for m, w in enumerate(weights)]
        print(f"  {qt}: {' '.join(weight_strs)}")

    # Build ensemble logits sample by sample
    ensemble_logits = np.zeros((n_samples, n_classes), dtype=np.float32)

    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        weights = type_weights.get(qtype, np.ones(n_models) / n_models)
        for m in range(n_models):
            ensemble_logits[i] += weights[m] * all_logits[m][i, :n_classes]

    return np.argmax(ensemble_logits, axis=1)


def strategy_majority_vote(all_logits, dataset):
    """
    Majority voting: each model votes for its predicted answer.
    Ties are broken by the model with highest confidence (max logit).
    """
    n_samples = all_logits[0].shape[0]
    n_models = len(all_logits)
    n_classes = min(l.shape[1] for l in all_logits)

    predictions = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        votes = {}
        for m in range(n_models):
            pred = np.argmax(all_logits[m][i, :n_classes])
            conf = all_logits[m][i, pred]
            if pred not in votes:
                votes[pred] = {'count': 0, 'max_conf': -1e9}
            votes[pred]['count'] += 1
            votes[pred]['max_conf'] = max(votes[pred]['max_conf'], conf)

        # Sort by vote count, then by max confidence
        best = max(votes.items(), key=lambda x: (x[1]['count'], x[1]['max_conf']))
        predictions[i] = best[0]

    return predictions


def strategy_qtype_best_model(all_logits, per_type_accs, dataset):
    """
    Pure routing: for each question type, use ONLY the single best model.
    No averaging — just picks the model with highest accuracy for that type.
    """
    qa_list = dataset.qa_list
    n_samples = all_logits[0].shape[0]
    n_classes = min(l.shape[1] for l in all_logits)
    n_models = len(all_logits)

    # Find best model per type
    best_model = {}
    for qt in QTYPE_NAMES:
        accs = [per_type_accs[m][qt] for m in range(n_models)]
        best_m = int(np.argmax(accs))
        best_model[qt] = best_m
        print(f"  {qt}: Model {best_m} (acc={accs[best_m]:.2f}%)")

    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        m = best_model.get(qtype, 0)
        predictions[i] = np.argmax(all_logits[m][i, :n_classes])

    return predictions


def strategy_top2_per_type(all_logits, per_type_accs, dataset):
    """
    For each question type, average logits from the TOP 2 models only.
    Ignores weaker models that would dilute the signal.
    """
    qa_list = dataset.qa_list
    n_samples = all_logits[0].shape[0]
    n_classes = min(l.shape[1] for l in all_logits)
    n_models = len(all_logits)

    # Find top-2 models per type
    top2 = {}
    for qt in QTYPE_NAMES:
        accs = [(per_type_accs[m][qt], m) for m in range(n_models)]
        accs.sort(reverse=True)
        top2[qt] = [accs[0][1], accs[1][1]]
        print(f"  {qt}: Model {accs[0][1]} ({accs[0][0]:.2f}%) + Model {accs[1][1]} ({accs[1][0]:.2f}%)")

    ensemble_logits = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(min(n_samples, len(qa_list))):
        qtype = get_qtype_for_sample(qa_list[i])
        m1, m2 = top2.get(qtype, [0, 1])
        ensemble_logits[i] = 0.5 * all_logits[m1][i, :n_classes] + 0.5 * all_logits[m2][i, :n_classes]

    return np.argmax(ensemble_logits, axis=1)


def main():
    parser = argparse.ArgumentParser(description='Smart Ensemble V3')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model specs: CONFIG:VERSION:EPOCH')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    parser.add_argument('--strategy', default='all',
                        choices=['all', 'qtype_routed', 'majority_vote',
                                 'best_model', 'top2', 'weighted_avg'],
                        help='Ensemble strategy (default: run all)')
    parser.add_argument('--weights', nargs='+', type=float, default=None)
    parser.add_argument('--temp', type=float, default=2.0,
                        help='Softmax temperature for qtype_routed (lower=more peaked)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    print(f"Smart Ensemble of {len(model_specs)} models:")
    for spec in model_specs:
        print(f"  {spec}")

    # Get logits from each model
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

    # First: evaluate each model individually and get per-type accuracies
    per_type_accs = {}  # model_idx -> {qtype: accuracy}
    for m, (spec, logits) in enumerate(zip(model_specs, all_logits)):
        preds = np.argmax(logits[:n_samples, :n_classes], axis=1)
        tc, t, ts = evaluate_per_type(dataset, preds)
        per_type_accs[m] = {}
        print(f"\n  Model {m} ({spec.split(':')[1]}):", end='')
        for qt in QTYPE_NAMES:
            if ts[qt]['total'] > 0:
                acc = 100 * ts[qt]['correct'] / ts[qt]['total']
            else:
                acc = 0
            per_type_accs[m][qt] = acc
            print(f" {qt}={acc:.1f}", end='')
        print(f" | Overall={100*tc/t:.2f}%")

    strategies_to_run = []
    if args.strategy == 'all':
        strategies_to_run = ['weighted_avg', 'qtype_routed', 'majority_vote',
                             'best_model', 'top2']
    else:
        strategies_to_run = [args.strategy]

    best_acc = 0
    best_strategy = ''

    for strategy in strategies_to_run:
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy}")
        print(f"{'='*60}")

        if strategy == 'weighted_avg':
            weights = args.weights
            if weights is None:
                weights = [1.0 / len(model_specs)] * len(model_specs)
            else:
                total = sum(weights)
                weights = [w / total for w in weights]
            ensemble_logits = np.zeros((n_samples, n_classes), dtype=np.float32)
            for logits, w in zip(all_logits, weights):
                ensemble_logits += w * logits[:n_samples, :n_classes]
            preds = np.argmax(ensemble_logits, axis=1)

        elif strategy == 'qtype_routed':
            preds = strategy_qtype_routed(all_logits, per_type_accs, dataset,
                                          softmax_temp=args.temp)

        elif strategy == 'majority_vote':
            preds = strategy_majority_vote(all_logits, dataset)

        elif strategy == 'best_model':
            preds = strategy_qtype_best_model(all_logits, per_type_accs, dataset)

        elif strategy == 'top2':
            preds = strategy_top2_per_type(all_logits, per_type_accs, dataset)

        tc, t, ts = evaluate_per_type(dataset, preds)
        acc = print_results(f"{strategy} Results", tc, t, ts)

        if acc > best_acc:
            best_acc = acc
            best_strategy = strategy

    print(f"\n{'='*60}")
    print(f"  BEST STRATEGY: {best_strategy} = {best_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
