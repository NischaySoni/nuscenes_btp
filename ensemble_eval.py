#!/usr/bin/env python3
"""
Ensemble Evaluation: Average logits from V1 and V3 models.

V1 is strong on object (46.89%) and comparison (67.71%)
V3 is strong on count (21.33%), status (53.46%), exist (82.26%)
Ensembling should combine the best of both → 56.5-57%+

Usage:
    python ensemble_eval.py \
        --ckpt1 ./outputs/ckpts/ckpt_trimodal_v1/epoch20.pkl \
        --ckpt2 ./outputs/ckpts/ckpt_trimodal_v3_qtype/epoch19.pkl \
        --config1 configs/mcan_trimodal_fusion.yaml \
        --config2 configs/mcan_trimodal_v3.yaml
"""

import os, sys, json, torch, yaml, argparse
import numpy as np
import torch.utils.data as Data

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.model_loader import CfgLoader, ModelLoader
from src.datasets.nuscenes_qa import NuScenes_QA
from src.execution.result_eval import Eval


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble evaluation of two trimodal models')
    parser.add_argument('--ckpt1', required=True, help='Path to V1 checkpoint')
    parser.add_argument('--ckpt2', required=True, help='Path to V3 checkpoint')
    parser.add_argument('--config1', default='configs/mcan_trimodal_fusion.yaml',
                        help='Config for model 1')
    parser.add_argument('--config2', default='configs/mcan_trimodal_v3.yaml',
                        help='Config for model 2')
    parser.add_argument('--weight1', type=float, default=0.5, help='Weight for model 1')
    parser.add_argument('--weight2', type=float, default=0.5, help='Weight for model 2')
    parser.add_argument('--GPU', type=int, default=0, help='GPU id')
    return parser.parse_args()


def load_model(__C, ckpt_path):
    """Load a trained model from checkpoint."""
    print(f"\n  Loading model from {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    
    return state_dict


@torch.no_grad()
def get_logits(net, dataloader, data_size, eval_batch_size):
    """Run inference and collect raw logits (before argmax)."""
    all_logits = []
    
    for step, batch in enumerate(dataloader):
        print(f"\r  Inference: [step {step}/{int(data_size / eval_batch_size)}]", end='          ')
        
        if len(batch) == 5:
            obj_feat, bbox_feat, ques_ix, ans, qtype = batch
        else:
            obj_feat, bbox_feat, ques_ix, ans = batch
        
        obj_feat = obj_feat.cuda()
        bbox_feat = bbox_feat.cuda()
        ques_ix = ques_ix.cuda()
        
        pred = net(obj_feat, bbox_feat, ques_ix)
        all_logits.append(pred.cpu().numpy())
    
    print('')
    return np.concatenate(all_logits, axis=0)


def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)
    
    # ---- Load config 1 (used for dataset loading) ----
    with open(args.config1, 'r') as f:
        yaml_dict1 = yaml.load(f, Loader=yaml.FullLoader)
    __C1 = CfgLoader(yaml_dict1['MODEL_USE']).load()
    __C1.add_args(yaml_dict1)
    __C1.RUN_MODE = 'val'
    __C1.GPU = '0'
    __C1.DEVICES = [0]
    __C1.N_GPU = 1
    __C1.proc()
    
    # ---- Load config 2 ----
    with open(args.config2, 'r') as f:
        yaml_dict2 = yaml.load(f, Loader=yaml.FullLoader)
    __C2 = CfgLoader(yaml_dict2['MODEL_USE']).load()
    __C2.add_args(yaml_dict2)
    __C2.RUN_MODE = 'val'
    __C2.GPU = '0'
    __C2.DEVICES = [0]
    __C2.N_GPU = 1
    __C2.proc()
    
    # ---- Load dataset (same for both models) ----
    print("\nLoading validation dataset...")
    dataset = NuScenes_QA(__C1)
    print(f"  Val dataset size: {dataset.data_size}")
    
    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C1.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=__C1.NUM_WORKERS,
        pin_memory=__C1.PIN_MEM
    )
    
    # ---- Build and load Model 1 ----
    print(f"\n{'='*60}")
    print(f"  MODEL 1: {args.config1}")
    print(f"{'='*60}")
    net1 = ModelLoader(__C1).Net(
        __C1, dataset.pretrained_emb, dataset.token_size, dataset.ans_size
    )
    net1.cuda()
    net1.eval()
    state_dict1 = load_model(__C1, args.ckpt1)
    net1.load_state_dict(state_dict1)
    
    print("  Running inference for Model 1...")
    logits1 = get_logits(net1, dataloader, dataset.data_size, __C1.EVAL_BATCH_SIZE)
    del net1  # Free GPU memory
    torch.cuda.empty_cache()
    
    # ---- Build and load Model 2 ----
    print(f"\n{'='*60}")
    print(f"  MODEL 2: {args.config2}")
    print(f"{'='*60}")
    net2 = ModelLoader(__C2).Net(
        __C2, dataset.pretrained_emb, dataset.token_size, dataset.ans_size
    )
    net2.cuda()
    net2.eval()
    state_dict2 = load_model(__C2, args.ckpt2)
    net2.load_state_dict(state_dict2)
    
    print("  Running inference for Model 2...")
    logits2 = get_logits(net2, dataloader, dataset.data_size, __C2.EVAL_BATCH_SIZE)
    del net2
    torch.cuda.empty_cache()
    
    # ---- Ensemble: weighted average of logits ----
    w1, w2 = args.weight1, args.weight2
    total = w1 + w2
    w1, w2 = w1 / total, w2 / total
    
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE: w1={w1:.2f}, w2={w2:.2f}")
    print(f"{'='*60}")
    
    # Softmax before averaging (probability-space ensemble)
    from scipy.special import softmax
    probs1 = softmax(logits1, axis=1)
    probs2 = softmax(logits2, axis=1)
    ensemble_probs = w1 * probs1 + w2 * probs2
    
    ans_ix_list = np.argmax(ensemble_probs, axis=1)
    
    # Pad to match eval expectations
    expected_len = int(np.ceil(dataset.data_size / __C1.EVAL_BATCH_SIZE)) * __C1.EVAL_BATCH_SIZE
    if len(ans_ix_list) < expected_len:
        ans_ix_list = np.pad(ans_ix_list, (0, expected_len - len(ans_ix_list)),
                             mode='constant', constant_values=-1)
    
    # ---- Evaluate ensemble ----
    log_file = __C1.LOG_PATH + '/log_run_ensemble_v1_v3.txt'
    result_file = __C1.RESULT_PATH + '/result_run_ensemble_v1_v3.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    Eval(__C1, dataset, ans_ix_list, log_file, result_file)
    
    # ---- Also try different weight ratios ----
    print(f"\n{'='*60}")
    print("  SWEEP: Trying different weight ratios...")
    print(f"{'='*60}")
    
    for w1_try in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w2_try = 1.0 - w1_try
        ensemble_try = w1_try * probs1 + w2_try * probs2
        preds_try = np.argmax(ensemble_try, axis=1)
        
        if len(preds_try) < expected_len:
            preds_try = np.pad(preds_try, (0, expected_len - len(preds_try)),
                               mode='constant', constant_values=-1)
        
        log_file_try = __C1.LOG_PATH + f'/log_run_ensemble_w{int(w1_try*10)}_{int(w2_try*10)}.txt'
        print(f"\n  --- w1={w1_try:.1f}, w2={w2_try:.1f} ---")
        Eval(__C1, dataset, preds_try, log_file_try, None)
    
    print("\n  Done! Check ./outputs/log/ for all ensemble results.")


if __name__ == '__main__':
    main()
