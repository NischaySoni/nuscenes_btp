# ------------------------------------------------------------------
# NuScenes-QA — Test/Evaluation Engine (v2)
# Fixed: results list accumulation bug, handles 5-tuple dataset
# ------------------------------------------------------------------

import os, json, torch, pickle
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

from src.models.model_loader import ModelLoader
from src.execution.result_eval import Eval
from src.datasets.answer_head_mapping import (
    QTYPE_NAMES, QTYPE_TO_IDX, HEAD_ANSWERS, HEAD_SIZES,
    build_global_to_local
)


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------
@torch.no_grad()
def test_engine(__C, dataset, state_dict=None, save_eval_result=False):

    # FIX: Use a local results list instead of module-level global
    # The old code had a module-level `results = []` that was never
    # cleared between evaluation calls, causing accumulation across epochs.
    results = []

    # ---------------- Load checkpoint ----------------

    if __C.CKPT_PATH is not None:
        print('Warning: using CKPT_PATH')
        path = __C.CKPT_PATH
    else:
        path = __C.CKPTS_PATH + \
               '/ckpt_' + __C.CKPT_VERSION + \
               '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

    if state_dict is None:

        print('Loading ckpt from:', path)

        state_dict = torch.load(path)['state_dict']

        print('Checkpoint loaded')

        if __C.N_GPU > 1:
            state_dict = ckpt_proc(state_dict)

    # ---------------- Model ----------------

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size
    )

    net.cuda()
    net.eval()

    if __C.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)

    net.load_state_dict(state_dict)

    # ---------------- DataLoader ----------------

    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM
    )

    ans_ix_list = []

    os.makedirs("./outputs/attention", exist_ok=True)

    # Multi-head setup
    use_multi_head = getattr(__C, 'USE_MULTI_HEAD', False)
    if use_multi_head:
        _, local_to_global = build_global_to_local(dataset.ans2ix)
        print(f"  [Eval] Multi-head mode: routing predictions through per-type heads")

    # ---------------- Evaluation Loop ----------------

    for step, batch in enumerate(dataloader):

        print("\rEvaluation: [step %4d/%4d]" % (
            step,
            int(data_size / __C.EVAL_BATCH_SIZE),
        ), end='          ')

        # Handle both 5-tuple and 4-tuple
        if len(batch) == 5:
            obj_feat_iter, bbox_feat_iter, ques_ix_iter, ans_iter, qtype_iter = batch
            qtype_iter = qtype_iter.cuda()
        else:
            obj_feat_iter, bbox_feat_iter, ques_ix_iter, ans_iter = batch
            qtype_iter = None

        obj_feat_iter = obj_feat_iter.cuda()
        bbox_feat_iter = bbox_feat_iter.cuda()
        ques_ix_iter = ques_ix_iter.cuda()

        pred = net(
            obj_feat_iter,
            bbox_feat_iter,
            ques_ix_iter
        )

        # ---- Multi-head prediction decoding ----
        if use_multi_head and isinstance(pred, dict) and qtype_iter is not None:
            # pred is dict: {qtype_name: (B, n_head_classes)}
            batch_size_actual = obj_feat_iter.size(0)
            pred_idx = torch.zeros(batch_size_actual, dtype=torch.long, device='cuda')
            # Also build a fake "global logits" tensor for backward compat
            ans_size = dataset.ans_size
            pred_global_logits = torch.full((batch_size_actual, ans_size), -1e9, device='cuda')

            for qi, qname in enumerate(QTYPE_NAMES):
                qmask = (qtype_iter == qi)
                if qmask.any():
                    head_logits = pred[qname][qmask]  # (n_matched, head_size)
                    local_preds = head_logits.argmax(dim=1)  # (n_matched,)
                    # Map local head predictions back to global answer indices
                    for j, idx_val in enumerate(qmask.nonzero(as_tuple=True)[0]):
                        local_pred = int(local_preds[j].item())
                        key = (qi, local_pred)
                        if key in local_to_global:
                            global_pred = local_to_global[key]
                        else:
                            global_pred = 0  # fallback
                        pred_idx[idx_val] = global_pred
                        pred_global_logits[idx_val, global_pred] = 1.0  # mark as chosen

        else:
            # Standard single-head prediction
            pred_idx = torch.argmax(pred, dim=1)
            pred_global_logits = pred

        # ------------------------------------------------
        # Save predictions for analysis
        # ------------------------------------------------

        for i in range(len(pred_idx)):

            q_index = step * __C.EVAL_BATCH_SIZE + i

            if q_index < len(dataset.qa_list):
                qa_item = dataset.qa_list[q_index]
                # Get ground truth answer index from answer dict (not from batch tensor which is 0 in val)
                gt_ans_str = str(qa_item['answer'])
                gt_idx = dataset.ans2ix.get(gt_ans_str, -1)

                results.append({
                    "question": qa_item["question"],
                    "gt_answer": gt_ans_str,
                    "gt": gt_idx,
                    "pred": int(pred_idx[i]),
                    "qtype": qa_item.get('template_type', 'unknown'),
                })

        # ------------------------------------------------
        # Save attention maps (try both fusion and single mode)
        # ------------------------------------------------

        try:
            # Try fusion mode attention
            actual_net = net.module if hasattr(net, 'module') else net
            if hasattr(actual_net, 'attflat_bev'):
                att = actual_net.attflat_bev.last_attention.cpu().numpy()
            elif hasattr(actual_net, 'attflat_img'):
                att = actual_net.attflat_img.last_attention.cpu().numpy()
            else:
                att = None

            if att is not None:
                np.save(
                    f"./outputs/attention/att_step_{step}.npy",
                    att
                )
        except:
            pass

        # ------------------------------------------------
        # Standard evaluation
        # ------------------------------------------------

        pred_np = pred_global_logits.cpu().data.numpy() if use_multi_head else pred.cpu().data.numpy()

        pred_argmax = np.argmax(pred_np, axis=1)

        if pred_argmax.shape[0] != __C.EVAL_BATCH_SIZE:

            pred_argmax = np.pad(
                pred_argmax,
                (0, __C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                mode='constant',
                constant_values=-1
            )

        ans_ix_list.append(pred_argmax)

    print('')

    ans_ix_list = np.array(ans_ix_list).reshape(-1)

    # ---------------- Evaluation ----------------

    if save_eval_result:

        result_eval_file = __C.RESULT_PATH + \
            '/result_run_' + __C.CKPT_VERSION + \
            '_epoch' + str(__C.CKPT_EPOCH) + '.txt'

    else:

        result_eval_file = None

    if __C.RUN_MODE not in ['train']:

        log_file = __C.LOG_PATH + '/log_run_' + __C.CKPT_VERSION + '.txt'

    else:

        log_file = __C.LOG_PATH + '/log_run_' + __C.VERSION + '.txt'

    Eval(__C, dataset, ans_ix_list, log_file, result_eval_file)

    # ---------------- Save analysis ----------------

    with open("prediction_analysis.json", "w") as f:

        json.dump(results, f, indent=2)

    print("\nSaved prediction_analysis.json (%d predictions)" % len(results))


# -------------------------------------------------------
# Fix for multi-GPU checkpoints
# -------------------------------------------------------

def ckpt_proc(state_dict):

    state_dict_new = {}

    for key in state_dict:

        state_dict_new['module.' + key] = state_dict[key]

    return state_dict_new
