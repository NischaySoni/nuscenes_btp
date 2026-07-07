#!/usr/bin/env python3
"""
Ensemble V12 Final Evaluation — Matching Official Eval Format

Trains per-type stacking MLPs on full val set (5-fold CV),
reports results in the SAME format as the training evaluation:
  - Overall, exist, exist_0, exist_1, count, count_0, count_1, etc.

This is the canonical result to report.

Usage:
    python ensemble_eval_v12_final.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
                 ... \
        --gpu 0
"""

import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter, defaultdict

from ensemble_eval_v2 import get_logits

QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']


def get_qtype_for_sample(qa_item):
    template_type = qa_item.get('template_type', 'exist')
    for qtype in QTYPE_NAMES:
        if template_type.startswith(qtype):
            return qtype
    return 'exist'


def softmax_np(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def build_meta_features(all_logits, n_samples, n_classes, n_models, qtypes):
    """Build rich meta-features from model logits."""
    raw_logits = np.zeros((n_samples, n_models * n_classes))
    for m in range(n_models):
        raw_logits[:, m*n_classes:(m+1)*n_classes] = all_logits[m][:n_samples, :n_classes]

    confidences = np.zeros((n_samples, n_models))
    entropies = np.zeros((n_samples, n_models))
    margins = np.zeros((n_samples, n_models))
    predictions = np.zeros((n_samples, n_models), dtype=np.int64)

    for m in range(n_models):
        logits_m = all_logits[m][:n_samples, :n_classes]
        probs_m = softmax_np(logits_m)
        confidences[:, m] = probs_m.max(axis=1)
        entropies[:, m] = -(probs_m * np.log(probs_m + 1e-10)).sum(axis=1)
        sorted_logits = np.sort(logits_m, axis=1)[:, ::-1]
        margins[:, m] = sorted_logits[:, 0] - sorted_logits[:, 1]
        predictions[:, m] = np.argmax(logits_m, axis=1)

    agreement = np.zeros((n_samples, 5))
    for i in range(n_samples):
        vote_counts = Counter(predictions[i])
        top_counts = sorted(vote_counts.values(), reverse=True)
        for k in range(min(5, len(top_counts))):
            agreement[i, k] = top_counts[k] / n_models

    qtype_onehot = np.zeros((n_samples, 5))
    for i in range(n_samples):
        if i in qtypes:
            qt_idx = QTYPE_NAMES.index(qtypes[i]) if qtypes[i] in QTYPE_NAMES else 0
            qtype_onehot[i, qt_idx] = 1.0

    ensemble_logits = np.mean(raw_logits.reshape(n_samples, n_models, n_classes), axis=1)

    features = np.concatenate([
        raw_logits, confidences, entropies, margins,
        agreement, qtype_onehot, ensemble_logits,
    ], axis=1)

    return features


class StackingMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_meta_model(X_train, y_train, X_val, y_val, n_classes,
                     hidden_dims=[512, 256], epochs=150, lr=0.001,
                     batch_size=256, dropout=0.3, device='cpu'):
    """Train a stacking MLP and return predictions on validation set."""
    input_dim = X_train.shape[1]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    X_tr = torch.FloatTensor(X_train_norm).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val_norm).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = StackingMLP(input_dim, n_classes, hidden_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1
    best_val_preds = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = (val_preds == y_val).mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_preds = val_preds.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_val_preds, best_val_acc


def official_eval(predictions_ix, dataset, log_file=None, result_file=None):
    """
    Evaluate predictions using the SAME logic as src/execution/result_eval.py:
      - Uses template_type for question type
      - Uses num_hop (0/1) for zero-hop / one-hop subtypes
      - Compares answer STRINGS (not indices)
    """
    qa_list = dataset.qa_list
    ix2ans = dataset.ix2ans
    ans2ix = dataset.ans2ix

    correct_by_q_type = defaultdict(list)

    for i in range(min(len(predictions_ix), len(qa_list))):
        true_answer = str(qa_list[i]['answer'])
        pred_ix = int(predictions_ix[i])
        predicted_answer = ix2ans.get(str(pred_ix), str(pred_ix))

        correct = 1 if true_answer == predicted_answer else 0
        correct_by_q_type['Overall'].append(correct)

        q_type = qa_list[i].get('template_type', 'unknown')
        num_hop = qa_list[i].get('num_hop', 0)
        sub_q_type = q_type + '_' + str(num_hop)

        correct_by_q_type[q_type].append(correct)
        correct_by_q_type[sub_q_type].append(correct)

    # Print in same format as result_eval.py
    q_dict = {}
    for q_type, vals in sorted(correct_by_q_type.items()):
        vals = np.asarray(vals)
        q_dict[q_type] = [int(vals.sum()), int(vals.shape[0])]

    print(f"\n{'='*60}")
    for q_type in sorted(q_dict.keys()):
        val, tol = q_dict[q_type]
        print(f"{q_type} : {val} / {tol} = {100.0 * val / tol:.2f}")

    # Write to log file
    if log_file:
        with open(log_file, 'a+') as lf:
            lf.write('=' * 60 + '\n')
            lf.write('Ensemble Stacking Evaluation\n')
            lf.write('=' * 60 + '\n')
            for q_type in sorted(q_dict.keys()):
                val, tol = q_dict[q_type]
                lf.write(f"{q_type} : {val} / {tol} = {100.0 * val / tol:.2f}\n")
            lf.write("\n")
        print(f"\nLog written to: {log_file}")

    # Write result file (same format as official)
    if result_file:
        with open(result_file, 'w') as rf:
            for i in range(min(len(predictions_ix), len(qa_list))):
                token = qa_list[i].get('sample_token', '')
                question = qa_list[i].get('question', '')
                pred_ix = int(predictions_ix[i])
                pred_ans = ix2ans.get(str(pred_ix), str(pred_ix))
                true_ans = str(qa_list[i]['answer'])
                rf.write(f"{token}    {question}    {pred_ans}    {true_ans}\n")
        print(f"Result file written to: {result_file}")

    return q_dict


def main():
    parser = argparse.ArgumentParser(description='Ensemble V12 Final Evaluation')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--log-file', default='outputs/log/log_ensemble_stacking.txt')
    parser.add_argument('--result-file', default='outputs/result/result_ensemble_stacking.txt')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model_specs = args.models
    n_models = len(model_specs)
    print(f"Per-Type Stacking Evaluation ({n_models} models, {args.folds}-fold CV)")
    for spec in model_specs:
        print(f"  {spec}")

    # ============================================================
    # Load all model logits on val set
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Loading model logits on val set")
    print(f"{'='*60}")
    all_logits = []
    dataset = None
    for spec in model_specs:
        logits, ds = get_logits(spec, args.gpu)
        all_logits.append(logits)
        if dataset is None:
            dataset = ds

    n_samples = min(l.shape[0] for l in all_logits)
    n_classes = min(l.shape[1] for l in all_logits)
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    print(f"  Samples: {n_samples}, Classes: {n_classes}")

    # Build indices
    qtype_indices = {qt: [] for qt in QTYPE_NAMES}
    valid_gt = {}
    qtypes = {}
    for i in range(len(qa_list)):
        gt_ans_str = str(qa_list[i]['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx == -1:
            continue
        qtype = get_qtype_for_sample(qa_list[i])
        qtype_indices[qtype].append(i)
        valid_gt[i] = gt_idx
        qtypes[i] = qtype

    print(f"  Valid samples: {len(valid_gt)}")
    for qt in QTYPE_NAMES:
        print(f"    {qt}: {len(qtype_indices[qt])}")

    # ============================================================
    # Build meta-features
    # ============================================================
    features = build_meta_features(all_logits, n_samples, n_classes, n_models, qtypes)
    print(f"  Meta-features: {features.shape[1]} dims")

    # ============================================================
    # Individual model baselines (official eval format)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Individual Model Baselines (official format)")
    print(f"{'='*60}")
    for m, spec in enumerate(model_specs):
        preds = np.argmax(all_logits[m][:n_samples, :n_classes], axis=1)
        name = spec.split(':')[1] if ':' in spec else spec
        # Quick overall
        correct = sum(1 for i in range(n_samples)
                      if i in valid_gt and preds[i] == valid_gt[i])
        total = len(valid_gt)
        print(f"  M{m} ({name}): {correct}/{total} = {100*correct/total:.2f}%")

    # Oracle
    oracle_correct = 0
    for i in range(n_samples):
        if i in valid_gt:
            for m in range(n_models):
                if np.argmax(all_logits[m][i, :n_classes]) == valid_gt[i]:
                    oracle_correct += 1
                    break
    print(f"\n  Oracle: {oracle_correct}/{len(valid_gt)} = "
          f"{100*oracle_correct/len(valid_gt):.2f}%")

    # ============================================================
    # Per-Type Stacking with K-fold CV
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Per-Type Stacking ({args.folds}-fold CV)")
    print(f"{'='*60}")

    n_folds = args.folds
    all_preds = np.zeros(n_samples, dtype=np.int64)

    for qt in QTYPE_NAMES:
        indices = [i for i in qtype_indices[qt] if i in valid_gt]
        if not indices:
            continue

        n_qt = len(indices)
        qt_features = features[indices]
        qt_labels = np.array([valid_gt[i] for i in indices])

        if n_qt < 10000:
            hidden_dims = [256, 128]
        else:
            hidden_dims = [512, 256]

        print(f"\n  Training {qt} meta-classifier ({n_qt} samples, "
              f"hidden={hidden_dims}):")

        rng = np.random.RandomState(42)
        shuffled = rng.permutation(n_qt)
        fold_size = n_qt // n_folds

        fold_accs = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_qt
            val_fold = shuffled[val_start:val_end]
            train_fold = np.concatenate([shuffled[:val_start], shuffled[val_end:]])

            X_train = qt_features[train_fold]
            y_train = qt_labels[train_fold]
            X_val = qt_features[val_fold]
            y_val = qt_labels[val_fold]

            preds, acc = train_meta_model(
                X_train, y_train, X_val, y_val, n_classes,
                hidden_dims=hidden_dims, epochs=150, lr=0.001,
                batch_size=256, dropout=0.3, device=device
            )
            fold_accs.append(acc)

            # Write predictions back to global array
            val_global = [indices[j] for j in val_fold]
            all_preds[val_global] = preds
            print(f"    Fold {fold+1}/{n_folds}: {acc*100:.2f}%")

        print(f"    Mean: {np.mean(fold_accs)*100:.2f}%")

    # ============================================================
    # Official Evaluation (matching result_eval.py format)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  OFFICIAL EVALUATION — Per-Type Stacking")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)

    q_dict = official_eval(
        all_preds, dataset,
        log_file=args.log_file,
        result_file=args.result_file
    )

    # ============================================================
    # Summary comparison
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*60}")

    # Best individual model overall
    best_single = 0
    best_name = ""
    for m, spec in enumerate(model_specs):
        preds = np.argmax(all_logits[m][:n_samples, :n_classes], axis=1)
        correct = sum(1 for i in range(n_samples)
                      if i in valid_gt and preds[i] == valid_gt[i])
        acc = 100 * correct / len(valid_gt)
        name = spec.split(':')[1] if ':' in spec else spec
        if acc > best_single:
            best_single = acc
            best_name = name

    stacking_overall = q_dict.get('Overall', [0, 1])
    stacking_pct = 100 * stacking_overall[0] / stacking_overall[1]

    print(f"  Best single model ({best_name}): {best_single:.2f}%")
    print(f"  Per-Type Stacking:                {stacking_pct:.2f}%")
    print(f"  Oracle:                           {100*oracle_correct/len(valid_gt):.2f}%")
    print(f"  Improvement over best single:     +{stacking_pct - best_single:.2f}%")


if __name__ == '__main__':
    main()
