#!/usr/bin/env python3
"""
Ensemble V12 Test: Train stacking on val set → predict on test set.

Pipeline:
  1. Load all model logits on the VALIDATION set (train the stacking MLP)
  2. Load all model logits on the TEST set (apply the trained MLP)
  3. Train per-type stacking MLPs on full val set
  4. Evaluate on test set (if labels available) or save predictions

Usage:
    python ensemble_eval_v12_test.py \
        --models mcan_trimodal_v14_bert_ft:trimodal_bert_ft_v1:16 \
                 mcan_trimodal_v15_bert_mh:trimodal_bert_mh_v1:22 \
                 ... \
        --gpu 0
        
    # If test questions file is at a different path:
    --test-json /path/to/NuScenes_test_questions.json
"""

import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import differential_evolution
from collections import Counter, defaultdict

# Import the existing logit loader
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


def build_indices(dataset):
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
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
    return qtype_indices, valid_gt, qtypes


def build_test_indices(dataset):
    """Build indices for test set — answers may not be available."""
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix
    qtype_indices = {qt: [] for qt in QTYPE_NAMES}
    valid_gt = {}  # May be empty if no test labels
    qtypes = {}
    for i in range(len(qa_list)):
        qtype = get_qtype_for_sample(qa_list[i])
        qtype_indices[qtype].append(i)
        qtypes[i] = qtype
        # Try to get ground truth (for evaluation if available)
        gt_ans_str = str(qa_list[i].get('answer', ''))
        gt_idx = ans2ix.get(gt_ans_str, -1)
        if gt_idx >= 0:
            valid_gt[i] = gt_idx
    has_labels = len(valid_gt) > 0
    return qtype_indices, valid_gt, qtypes, has_labels


def build_meta_features(all_logits, n_samples, n_classes, n_models, qtypes):
    """Build rich meta-features from model logits."""
    # 1. Raw logits
    raw_logits = np.zeros((n_samples, n_models * n_classes))
    for m in range(n_models):
        raw_logits[:, m*n_classes:(m+1)*n_classes] = all_logits[m][:n_samples, :n_classes]

    # 2. Per-model stats
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

    # 3. Agreement features
    agreement = np.zeros((n_samples, 5))
    for i in range(n_samples):
        vote_counts = Counter(predictions[i])
        top_counts = sorted(vote_counts.values(), reverse=True)
        for k in range(min(5, len(top_counts))):
            agreement[i, k] = top_counts[k] / n_models

    # 4. Question type one-hot
    qtype_onehot = np.zeros((n_samples, 5))
    for i in range(n_samples):
        if i in qtypes:
            qt_idx = QTYPE_NAMES.index(qtypes[i]) if qtypes[i] in QTYPE_NAMES else 0
            qtype_onehot[i, qt_idx] = 1.0

    # 5. Ensemble logits
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


def train_stacking_model(X_train, y_train, n_classes, hidden_dims=[512, 256],
                         epochs=150, lr=0.001, batch_size=256, dropout=0.3,
                         device='cpu'):
    """Train a stacking MLP on FULL training data. Returns model + normalization stats."""
    input_dim = X_train.shape[1]

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train_norm = (X_train - mean) / std

    X_tr = torch.FloatTensor(X_train_norm).to(device)
    y_tr = torch.LongTensor(y_train).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = StackingMLP(input_dim, n_classes, hidden_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = total_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch+1}, best loss: {best_loss:.4f}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    return model, mean, std


def predict_with_model(model, X_test, mean, std, device='cpu'):
    """Apply trained stacking model to test features."""
    X_test_norm = (X_test - mean) / std
    X_t = torch.FloatTensor(X_test_norm).to(device)

    model.eval()
    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = 1024
        all_preds = []
        for start in range(0, len(X_t), batch_size):
            end = min(start + batch_size, len(X_t))
            logits = model(X_t[start:end])
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds)


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


def get_logits_for_split(model_spec, gpu_id, split='test'):
    """
    Modified version of get_logits that can run on train/val/test split.
    """
    import yaml
    from src.models.mcan.model_cfgs import Cfgs
    from src.models.model_loader import ModelLoader
    from src.datasets.nuscenes_qa import NuScenes_QA

    parts = model_spec.split(':')
    if len(parts) == 3:
        config_name, version, epoch = parts
        config_file = f'configs/{config_name}.yaml'
    elif len(parts) == 2:
        version, epoch = parts
        from ensemble_eval_v2 import find_config_for_version
        config_file = find_config_for_version(version)
    else:
        raise ValueError(f"Invalid model spec: {model_spec}")

    epoch = int(epoch)

    __C = Cfgs()
    with open(config_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    for key, val in cfg_dict.items():
        setattr(__C, key, val)

    __C.GPU = str(gpu_id)
    __C.RUN_MODE = 'val'  # Use val mode for loading

    # KEY: set the split to load
    if split == 'test':
        # Load test questions instead of val
        __C.SPLIT = {'train': 'train', 'val': 'test', 'test': 'test'}
        # If test QA file exists separately:
        test_qa = __C.RAW_PATH.get('test', None)
        if test_qa is None:
            # Try standard naming convention
            base_dir = os.path.dirname(__C.RAW_PATH['val'])
            test_path = os.path.join(base_dir, 'NuScenes_test_questions.json')
            if os.path.exists(test_path):
                __C.RAW_PATH['test'] = test_path
            else:
                print(f"  WARNING: No test questions file found at {test_path}")
                print(f"  Using val split as test (holdout evaluation)")
                __C.SPLIT = {'train': 'train', 'val': 'val', 'test': 'val'}
    elif split == 'train':
        __C.SPLIT = {'train': 'train', 'val': 'train', 'test': 'train'}
    else:  # val
        __C.SPLIT = {'train': 'train', 'val': 'val', 'test': 'val'}

    __C.N_GPU = 1
    __C.DEVICES = [int(gpu_id)]
    __C.PIN_MEM = True
    __C.NUM_WORKERS = 4

    __C.FEAT_SIZE = {
        'OBJ_FEAT_SIZE': __C.OBJ_FEAT_SIZE if hasattr(__C, 'OBJ_FEAT_SIZE') else [80, 69],
    }
    if hasattr(__C, 'BBOX_FEAT_SIZE'):
        __C.FEAT_SIZE['BBOX_FEAT_SIZE'] = __C.BBOX_FEAT_SIZE

    if hasattr(__C, 'GRAD_ACCU_STEPS') and __C.GRAD_ACCU_STEPS > 1:
        __C.SUB_BATCH_SIZE = __C.BATCH_SIZE // __C.GRAD_ACCU_STEPS
    else:
        __C.SUB_BATCH_SIZE = __C.BATCH_SIZE
    __C.EVAL_BATCH_SIZE = max(1, __C.SUB_BATCH_SIZE // 2)

    dataset = NuScenes_QA(__C)

    ckpt_path = f'./outputs/ckpts/ckpt_{version}/epoch{epoch}.pkl'
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

    net = ModelLoader(__C).Net(
        __C, dataset.pretrained_emb, dataset.token_size, dataset.ans_size
    )

    remapped = {}
    for k, v in state_dict.items():
        new_k = k.replace('distilbert.', 'bert_model.')
        remapped[new_k] = v
    state_dict = remapped

    net.cuda()
    net.eval()
    net.load_state_dict(state_dict)

    dataloader = Data.DataLoader(
        dataset, batch_size=__C.EVAL_BATCH_SIZE,
        shuffle=False, num_workers=__C.NUM_WORKERS, pin_memory=__C.PIN_MEM
    )

    use_multi_head = getattr(__C, 'USE_MULTI_HEAD', False)

    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 6:
                obj_feat, _, bbox_feat, ques_ix, ans, qtype = batch
            elif len(batch) == 5:
                obj_feat, bbox_feat, ques_ix, ans, qtype = batch
            else:
                obj_feat, bbox_feat, ques_ix, ans = batch

            obj_feat = obj_feat.cuda()
            bbox_feat = bbox_feat.cuda()
            ques_ix = ques_ix.cuda()

            pred = net(obj_feat, bbox_feat, ques_ix)

            if use_multi_head and isinstance(pred, dict):
                from src.datasets.answer_head_mapping import (
                    QTYPE_NAMES as QT_NAMES, HEAD_ANSWERS, build_global_to_local
                )
                ans2ix = dataset.ans2ix
                _, local_to_global = build_global_to_local(ans2ix)
                batch_logits = torch.zeros(obj_feat.size(0), len(ans2ix)).cuda()
                for qi, qname in enumerate(QT_NAMES):
                    if qname in pred:
                        head_logits = pred[qname]
                        for local_idx in range(head_logits.size(1)):
                            global_idx = local_to_global.get((qname, local_idx), None)
                            if global_idx is not None:
                                batch_logits[:, global_idx] = torch.max(
                                    batch_logits[:, global_idx], head_logits[:, local_idx]
                                )
                all_logits.append(batch_logits.cpu().numpy())
            else:
                all_logits.append(pred.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    del net
    torch.cuda.empty_cache()
    return logits, dataset


def main():
    parser = argparse.ArgumentParser(description='Ensemble V12 Test: Val→Test Stacking')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--test-split', default='train',
                        help='Split to use as test set: "test" or "train" (for holdout eval)')
    parser.add_argument('--save-predictions', default='outputs/test_predictions.json',
                        help='Path to save test predictions')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model_specs = args.models
    n_models = len(model_specs)
    print(f"Val→Test Stacking Pipeline ({n_models} models)")
    print(f"Test split: {args.test_split}")

    # ============================================================
    # Step 1: Load VAL logits (for training the stacking MLP)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Step 1: Loading VAL logits (training data for stacking)")
    print(f"{'='*60}")
    val_logits = []
    val_dataset = None
    for spec in model_specs:
        logits, ds = get_logits(spec, args.gpu)
        val_logits.append(logits)
        if val_dataset is None:
            val_dataset = ds

    n_val = min(l.shape[0] for l in val_logits)
    n_classes = min(l.shape[1] for l in val_logits)
    print(f"  Val samples: {n_val}, Classes: {n_classes}")

    val_qtype_indices, val_gt, val_qtypes = build_indices(val_dataset)
    val_features = build_meta_features(val_logits, n_val, n_classes, n_models, val_qtypes)
    print(f"  Val features shape: {val_features.shape}")

    # ============================================================
    # Step 2: Load TEST logits
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Step 2: Loading TEST logits ({args.test_split} split)")
    print(f"{'='*60}")
    test_logits = []
    test_dataset = None
    for spec in model_specs:
        print(f"  Loading {spec} on {args.test_split}...")
        logits, ds = get_logits_for_split(spec, args.gpu, split=args.test_split)
        test_logits.append(logits)
        if test_dataset is None:
            test_dataset = ds

    n_test = min(l.shape[0] for l in test_logits)
    print(f"  Test samples: {n_test}")

    test_qtype_indices, test_gt, test_qtypes, test_has_labels = build_test_indices(test_dataset)
    test_features = build_meta_features(test_logits, n_test, n_classes, n_models, test_qtypes)
    print(f"  Test features shape: {test_features.shape}")
    print(f"  Test has labels: {test_has_labels} ({len(test_gt)} labeled samples)")

    # ============================================================
    # Step 3: Train per-type stacking MLPs on FULL val set
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Step 3: Training per-type stacking MLPs on full val set")
    print(f"{'='*60}")

    test_predictions = np.zeros(n_test, dtype=np.int64)

    for qt in QTYPE_NAMES:
        val_indices = [i for i in val_qtype_indices[qt] if i in val_gt]
        test_indices = [i for i in test_qtype_indices[qt]]

        if not val_indices or not test_indices:
            print(f"  {qt}: skipped (no data)")
            continue

        X_train = val_features[val_indices]
        y_train = np.array([val_gt[i] for i in val_indices])
        X_test = test_features[test_indices]

        n_qt_train = len(val_indices)
        n_qt_test = len(test_indices)

        if n_qt_train < 10000:
            hidden_dims = [256, 128]
        else:
            hidden_dims = [512, 256]

        print(f"\n  Training {qt} MLP ({n_qt_train} train → {n_qt_test} test):")
        model, mean, std = train_stacking_model(
            X_train, y_train, n_classes,
            hidden_dims=hidden_dims, epochs=150, lr=0.001,
            batch_size=256, dropout=0.3, device=device
        )

        # Predict on test
        preds = predict_with_model(model, X_test, mean, std, device=device)
        for j, test_idx in enumerate(test_indices):
            test_predictions[test_idx] = preds[j]

        # Quick train-set accuracy (sanity check)
        train_preds = predict_with_model(model, X_train, mean, std, device=device)
        train_acc = (train_preds == y_train).mean()
        print(f"    Train accuracy (sanity): {train_acc*100:.2f}%")

        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Step 4: Evaluate on test set (if labels available)
    # ============================================================
    if test_has_labels:
        print(f"\n{'='*60}")
        print(f"  Step 4: TEST SET EVALUATION")
        print(f"{'='*60}")

        ov, tc, t, ts = evaluate(test_predictions, test_gt, test_qtype_indices, n_test)
        print_eval("Per-Type Stacking (TEST)", ov, tc, t, ts)

        # Also compute individual model baselines on test
        print(f"\n  Individual model baselines on test:")
        for m, spec in enumerate(model_specs):
            preds = np.argmax(test_logits[m][:n_test, :n_classes], axis=1)
            ov_m, tc_m, t_m, ts_m = evaluate(preds, test_gt, test_qtype_indices, n_test)
            name = spec.split(':')[1] if ':' in spec else spec
            print(f"    M{m} ({name}): {ov_m:.2f}%")

        # Oracle
        oracle_correct = 0
        for i in range(n_test):
            if i in test_gt:
                for m in range(n_models):
                    if np.argmax(test_logits[m][i, :n_classes]) == test_gt[i]:
                        oracle_correct += 1
                        break
        total_valid = sum(1 for i in range(n_test) if i in test_gt)
        print(f"\n  Oracle ceiling (test): {oracle_correct}/{total_valid} = "
              f"{100*oracle_correct/total_valid:.2f}%")
    else:
        print(f"\n  No test labels — saving predictions only")

    # ============================================================
    # Step 5: Save predictions
    # ============================================================
    ix2ans = {v: k for k, v in val_dataset.ans2ix.items()}
    predictions_list = []
    qa_list = test_dataset.qa_list
    for i in range(min(n_test, len(qa_list))):
        pred_idx = int(test_predictions[i])
        pred_ans = ix2ans.get(pred_idx, str(pred_idx))
        entry = {
            'question_id': qa_list[i].get('question_id', i),
            'question': qa_list[i].get('question', ''),
            'predicted_answer': pred_ans,
            'question_type': get_qtype_for_sample(qa_list[i]),
        }
        if test_has_labels:
            entry['ground_truth'] = str(qa_list[i].get('answer', ''))
            entry['correct'] = (pred_ans == entry['ground_truth'])
        predictions_list.append(entry)

    os.makedirs(os.path.dirname(args.save_predictions) or '.', exist_ok=True)
    with open(args.save_predictions, 'w') as f:
        json.dump(predictions_list, f, indent=2)
    print(f"\n  Predictions saved to: {args.save_predictions}")
    print(f"  Total predictions: {len(predictions_list)}")


if __name__ == '__main__':
    main()
