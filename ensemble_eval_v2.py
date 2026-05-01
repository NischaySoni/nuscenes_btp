#!/usr/bin/env python3
"""
Ensemble Evaluation: Average logits from multiple trained models.
Typically gives +1-2% over best single model.

Usage:
    python ensemble_eval_v2.py \
        --models trimodal_v4_finetune:24 5900402:6 \
        --gpu 0
    
    Each model spec is VERSION:EPOCH
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from src.configs.base_cfgs import Cfgs
from src.models.model_loader import ModelLoader
from src.datasets.nuscenes_qa import NuScenes_QA
from src.execution.result_eval import Eval


def load_config(model_name, config_file):
    """Load config from YAML and return Cfgs object."""
    __C = Cfgs()
    # Load YAML config
    import yaml
    with open(config_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    for key, val in cfg_dict.items():
        if hasattr(__C, key):
            setattr(__C, key, val)
        else:
            setattr(__C, key, val)
    return __C


def find_config_for_version(version):
    """Try to find the config file used for a given training version."""
    # Check log file for clues
    log_file = f'./outputs/log/log_run_{version}.txt'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
        # Look for model config hints
        if 'TRIMODAL_FUSION' in content:
            if 'v4' in version or 'finetune' in version:
                return 'configs/mcan_trimodal_v4.yaml'
            elif 'v3' in version:
                return 'configs/mcan_trimodal_v3.yaml'
            elif 'v2' in version:
                return 'configs/mcan_trimodal_v2.yaml'
            else:
                return 'configs/mcan_trimodal_fusion.yaml'
        elif 'CENTERPOINT_ONLY' in content:
            return 'configs/mcan_centerpoint.yaml'
        elif 'CENTERPOINT_FUSION' in content:
            return 'configs/mcan_centerpoint_fusion.yaml'
        elif 'RADARXF_FUSION' in content:
            return 'configs/mcan_radarxf_fusion.yaml'
    return None


@torch.no_grad()
def get_logits(model_spec, gpu_id):
    """
    Run a single model and return raw logits for all val samples.
    model_spec: "VERSION:EPOCH" or "CONFIG_NAME:VERSION:EPOCH"
    """
    parts = model_spec.split(':')
    if len(parts) == 3:
        config_name, version, epoch = parts
        config_file = f'configs/{config_name}.yaml'
    elif len(parts) == 2:
        version, epoch = parts
        config_file = find_config_for_version(version)
        if config_file is None:
            print(f"ERROR: Cannot find config for version '{version}'")
            print("Use format CONFIG_NAME:VERSION:EPOCH")
            sys.exit(1)
    else:
        print(f"ERROR: Invalid model spec '{model_spec}'")
        print("Use VERSION:EPOCH or CONFIG_NAME:VERSION:EPOCH")
        sys.exit(1)

    epoch = int(epoch)
    print(f"\n{'='*60}")
    print(f"Loading model: version={version}, epoch={epoch}")
    print(f"Config: {config_file}")
    print(f"{'='*60}")

    # Build config
    __C = Cfgs()
    import yaml
    with open(config_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    for key, val in cfg_dict.items():
        setattr(__C, key, val)

    __C.GPU = str(gpu_id)
    __C.RUN_MODE = 'val'
    __C.SPLIT = {'train': 'train', 'val': 'val', 'test': 'val'}
    __C.N_GPU = 1
    __C.DEVICES = [int(gpu_id)]
    __C.PIN_MEM = True
    __C.NUM_WORKERS = 4

    # Process config
    __C.FEAT_SIZE = {
        'OBJ_FEAT_SIZE': __C.OBJ_FEAT_SIZE if hasattr(__C, 'OBJ_FEAT_SIZE') else [80, 69],
    }
    if hasattr(__C, 'BBOX_FEAT_SIZE'):
        __C.FEAT_SIZE['BBOX_FEAT_SIZE'] = __C.BBOX_FEAT_SIZE

    # Batch sizes
    if hasattr(__C, 'GRAD_ACCU_STEPS') and __C.GRAD_ACCU_STEPS > 1:
        __C.SUB_BATCH_SIZE = __C.BATCH_SIZE // __C.GRAD_ACCU_STEPS
    else:
        __C.SUB_BATCH_SIZE = __C.BATCH_SIZE
    __C.EVAL_BATCH_SIZE = max(1, __C.SUB_BATCH_SIZE // 2)

    # Load dataset
    dataset = NuScenes_QA(__C)

    # Load checkpoint
    ckpt_path = f'./outputs/ckpts/ckpt_{version}/epoch{epoch}.pkl'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

    # Build model
    net = ModelLoader(__C).Net(
        __C,
        dataset.pretrained_emb,
        dataset.token_size,
        dataset.ans_size
    )
    net.cuda()
    net.eval()
    net.load_state_dict(state_dict)

    # Run inference
    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM
    )

    all_logits = []
    for step, batch in enumerate(dataloader):
        if len(batch) == 5:
            obj_feat, bbox_feat, ques_ix, ans, qtype = batch
        else:
            obj_feat, bbox_feat, ques_ix, ans = batch

        obj_feat = obj_feat.cuda()
        bbox_feat = bbox_feat.cuda()
        ques_ix = ques_ix.cuda()

        logits = net(obj_feat, bbox_feat, ques_ix)
        all_logits.append(logits.cpu().numpy())

        if step % 500 == 0:
            print(f"\r  Inference: [{step}/{len(dataloader)}]", end='')

    print(f"\r  Inference: [{len(dataloader)}/{len(dataloader)}] Done!")
    all_logits = np.concatenate(all_logits, axis=0)
    print(f"  Logits shape: {all_logits.shape}")

    return all_logits, dataset


def evaluate(dataset, predictions, label="Ensemble"):
    """Evaluate predictions against ground truth."""
    qa_list = dataset.qa_list
    ans2ix = dataset.ans2ix

    # Category-wise evaluation
    categories = {}
    total_correct = 0
    total = 0

    for i in range(min(len(predictions), len(qa_list))):
        pred_idx = predictions[i]
        qa_item = qa_list[i]
        gt_ans_str = str(qa_item['answer'])
        gt_idx = ans2ix.get(gt_ans_str, -1)

        if gt_idx == -1:
            continue

        template_type = qa_item.get('template_type', 'unknown')

        # Parse hop level
        hop = '0'
        if '_' in template_type:
            parts = template_type.rsplit('_', 1)
            if parts[1] in ('0', '1'):
                template_type = parts[0]
                hop = parts[1]

        # Base category
        base_cat = template_type.split('_')[0] if '_' in template_type else template_type

        correct = int(pred_idx == gt_idx)
        total_correct += correct
        total += 1

        if base_cat not in categories:
            categories[base_cat] = {'correct': 0, 'total': 0, 'hops': {}}
        categories[base_cat]['correct'] += correct
        categories[base_cat]['total'] += 1

        hop_key = f"{base_cat}_{hop}"
        if hop_key not in categories[base_cat]['hops']:
            categories[base_cat]['hops'][hop_key] = {'correct': 0, 'total': 0}
        categories[base_cat]['hops'][hop_key]['correct'] += correct
        categories[base_cat]['hops'][hop_key]['total'] += 1

    print(f"\n{'='*60}")
    print(f"  {label} Results")
    print(f"{'='*60}")
    print(f"Overall {total_correct} / {total} = {100*total_correct/total:.2f}")

    for cat in sorted(categories.keys()):
        c = categories[cat]
        acc = 100 * c['correct'] / c['total']
        print(f"{cat} {c['correct']} / {c['total']} = {acc:.2f}")
        for hop_key in sorted(c['hops'].keys()):
            h = c['hops'][hop_key]
            hacc = 100 * h['correct'] / h['total']
            print(f"{hop_key} {h['correct']} / {h['total']} = {hacc:.2f}")

    return 100 * total_correct / total


def main():
    parser = argparse.ArgumentParser(description='Ensemble Evaluation')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model specs: VERSION:EPOCH or CONFIG:VERSION:EPOCH')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Weights for each model (default: equal)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_specs = args.models
    weights = args.weights
    if weights is None:
        weights = [1.0 / len(model_specs)] * len(model_specs)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    print(f"Ensemble of {len(model_specs)} models:")
    for spec, w in zip(model_specs, weights):
        print(f"  {spec} (weight={w:.3f})")

    # Get logits from each model
    all_model_logits = []
    dataset = None
    for spec in model_specs:
        logits, ds = get_logits(spec, args.gpu)
        all_model_logits.append(logits)
        if dataset is None:
            dataset = ds

    # Ensure all logits have same shape
    min_samples = min(l.shape[0] for l in all_model_logits)
    min_classes = min(l.shape[1] for l in all_model_logits)

    print(f"\nEnsembling {len(all_model_logits)} models...")
    print(f"  Samples: {min_samples}, Classes: {min_classes}")

    # Average logits (weighted)
    ensemble_logits = np.zeros((min_samples, min_classes), dtype=np.float32)
    for logits, w in zip(all_model_logits, weights):
        ensemble_logits += w * logits[:min_samples, :min_classes]

    # Get predictions
    ensemble_preds = np.argmax(ensemble_logits, axis=1)

    # Also evaluate individual models
    print("\n" + "="*60)
    print("Individual Model Results:")
    print("="*60)
    for i, (spec, logits) in enumerate(zip(model_specs, all_model_logits)):
        preds = np.argmax(logits[:min_samples, :min_classes], axis=1)
        acc = evaluate(dataset, preds, label=f"Model {i+1}: {spec}")

    # Evaluate ensemble
    ensemble_acc = evaluate(dataset, ensemble_preds, label="ENSEMBLE")

    print(f"\n{'='*60}")
    print(f"  FINAL ENSEMBLE ACCURACY: {ensemble_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
