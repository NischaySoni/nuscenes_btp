# ------------------------------------------------------------------
# NuScenes-QA — Training Engine (v2)
# Fixes: correct count loss targeting, label smoothing, grad clipping
# ------------------------------------------------------------------
print(">>> train_engine.py started <<<", flush=True)

import os, torch, datetime, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from src.models.model_loader import ModelLoader
from src.utils.optim import get_optim, adjust_lr
from src.execution.test_engine import test_engine, ckpt_proc
from src.ops.detection import DetectionModule
from src.ops.radar_fusion import RadarImageFusion


def train_engine(__C, dataset, dataset_eval=None):

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    # Initialize Main Network
    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size
    )
    net.cuda()
    net.train()

    # --- Initialize Online Detection Modules ---
    use_online_detection = getattr(__C, 'USE_YOLO_DETECTION', False)
    detection_module = None
    if use_online_detection:
        print(">>> Using Online YOLO Detection & Radar Fusion <<<")
        detection_module = DetectionModule(__C).cuda()
        for param in detection_module.parameters():
            param.requires_grad = False
        if hasattr(detection_module, 'yolo_model') and hasattr(detection_module.yolo_model, 'model'):
            detection_module.yolo_model.model.eval()
    # -------------------------------------------

    # --- Initialize Teacher Model for KD ---
    teacher_net = None
    use_kd = (getattr(__C, 'USE_KD', 'False') == 'True')
    if use_kd:
        print(">>> Setting up Knowledge Distillation Teacher Model <<<")
        teacher_net = ModelLoader(__C).Net(__C, pretrained_emb, token_size, ans_size)
        teacher_net.cuda()
        teacher_net.eval()
        
        teacher_ckpt_path = getattr(__C, 'TEACHER_CKPT', None)
        if teacher_ckpt_path and os.path.exists(teacher_ckpt_path):
            print(f"Loading Teacher Checkpoint from {teacher_ckpt_path}")
            teacher_ckpt = torch.load(teacher_ckpt_path)
            if __C.N_GPU > 1:
                teacher_net.load_state_dict(ckpt_proc(teacher_ckpt['state_dict']))
            else:
                teacher_net.load_state_dict(teacher_ckpt['state_dict'])
        else:
            print(f"WARNING: Teacher Checkpoint NOT FOUND at {teacher_ckpt_path}! KD will use random teacher.")
            
        for param in teacher_net.parameters():
            param.requires_grad = False
    # ---------------------------------------

    if __C.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)
        if detection_module is not None:
            detection_module = nn.DataParallel(detection_module, device_ids=__C.DEVICES)
        if teacher_net is not None:
            teacher_net = nn.DataParallel(teacher_net, device_ids=__C.DEVICES)

    # Define Loss Function — with label smoothing for fusion/annot/detected mode
    vis_feat = getattr(__C, 'VISUAL_FEATURE', 'bev')
    is_fusion = (vis_feat == 'fusion')
    label_smoothing = getattr(__C, 'LABEL_SMOOTHING', 0.0) if vis_feat in ('fusion', 'annot', 'detected') else 0.0

    if __C.LOSS_FUNC == 'ce':
        loss_fn = nn.CrossEntropyLoss(
            reduction=__C.LOSS_REDUCTION,
            label_smoothing=label_smoothing
        ).cuda()
        print(f"  [Loss] CrossEntropyLoss(reduction={__C.LOSS_REDUCTION}, "
              f"label_smoothing={label_smoothing})")
    else:
        loss_fn = eval('torch.nn.' + __C.LOSS_FUNC_NAME_DICT[__C.LOSS_FUNC] +
                       "(reduction='" + __C.LOSS_REDUCTION + "').cuda()")

    # Load checkpoint if resume training
    ckpt_dir = __C.CKPTS_PATH + '/ckpt_' + __C.VERSION
    if getattr(__C, 'FINETUNE_FROM', None) is not None:
        path = __C.FINETUNE_FROM
        print(' ========== Fine-Tuning from:', path)
        if os.path.exists(path):
            ckpt = torch.load(path)
            if __C.N_GPU > 1:
                net.load_state_dict(ckpt_proc(ckpt['state_dict']), strict=False)
            else:
                net.load_state_dict(ckpt['state_dict'], strict=False)
            print('Successfully loaded weights! Initializing fresh optimizer for Epoch 0.')
            
            # --- FREEZE REASONING BACKBONE ---
            if getattr(__C, 'FREEZE_BACKBONE', False):
                print(' => FREEZING logic backbone mapping (embedding, lstm, backbone). Only training Visual Adapter & Classifier!')
                for name, param in net.named_parameters():
                    # We only want to train 'annot_adapter' and 'proj'/'proj_norm' (classifier)
                    if 'annot_adapter' not in name and 'proj' not in name:
                        param.requires_grad = False
        else:
            print(f'WARNING: Fine-tune checkpoint {path} not found. Starting from scratch.')

        os.makedirs(ckpt_dir, exist_ok=True)
        optim = get_optim(__C, net, data_size)
        start_epoch = 0

    elif __C.RESUME:
        print(' ========== Resume training')
        if getattr(__C, 'CKPT_PATH', None) is not None:
            print('Warning: Now using CKPT_PATH args, CKPT_VERSION and CKPT_EPOCH will not work')
            path = __C.CKPT_PATH
        elif hasattr(__C, 'CKPT_EPOCH') and __C.CKPT_EPOCH is not None:
            path = ckpt_dir + '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'
        else:
            latest_path = ckpt_dir + '/latest.pkl'
            if os.path.exists(latest_path):
                path = latest_path
                print('Auto-resuming from latest.pkl')
            else:
                print('WARNING: No checkpoint found to resume from. Starting fresh.')
                path = None

        if path is not None and os.path.exists(path):
            print('Loading ckpt from {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')

            if __C.N_GPU > 1:
                net.load_state_dict(ckpt_proc(ckpt['state_dict']))
            else:
                net.load_state_dict(ckpt['state_dict'])

            start_epoch = ckpt['epoch']
            optim = get_optim(__C, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / __C.BATCH_SIZE * start_epoch)
            optim.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            optim = get_optim(__C, net, data_size)
            start_epoch = 0

        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        optim = get_optim(__C, net, data_size)
        start_epoch = 0

    loss_sum = 0
    named_params = list(net.named_parameters())
    grad_norm = np.zeros(len(named_params))

    # --- Early stopping ---
    early_stop_patience = getattr(__C, 'EARLY_STOP_PATIENCE', 5)
    best_eval_loss = float('inf')
    epochs_without_improvement = 0

    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.BATCH_SIZE,
        shuffle=True,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM,
        drop_last=True
    )

    logfile_path = __C.LOG_PATH + '/log_run_' + __C.VERSION + '.txt'
    with open(logfile_path, 'a+') as logfile:
        logfile.write(str(__C))

    # Count loss config
    count_loss_weight = getattr(__C, 'COUNT_LOSS_WEIGHT', 0.3)
    print(f"  [Config] count_loss_weight={count_loss_weight}, "
          f"grad_clip={__C.GRAD_NORM_CLIP}, fusion={is_fusion}")

    # Training Loop
    for epoch in range(start_epoch, __C.MAX_EPOCH):

        with open(logfile_path, 'a+') as logfile:
            logfile.write('=====================================\nnowTime: ' +
                          datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')

        if epoch in __C.LR_DECAY_LIST:
            adjust_lr(optim, __C.LR_DECAY_R)

        time_start = time.time()

        # Tracking for fusion diagnostics
        gate_sum = 0.0
        gate_count = 0
        count_loss_sum = 0.0
        count_loss_steps = 0

        for step, batch in enumerate(dataloader):

            optim.zero_grad()

            # --- Unpack Tensors ---
            teacher_feat_iter = None
            if len(batch) == 6:
                obj_feat_iter, teacher_feat_iter, bbox_feat_iter, ques_ix_iter, ans_iter, qtype_iter = batch
                qtype_iter = qtype_iter.cuda()
                teacher_feat_iter = teacher_feat_iter.cuda()
            elif len(batch) == 5:
                obj_feat_iter, bbox_feat_iter, ques_ix_iter, ans_iter, qtype_iter = batch
                qtype_iter = qtype_iter.cuda()
            else:
                # Backward compatibility with 4-tuple
                obj_feat_iter, bbox_feat_iter, ques_ix_iter, ans_iter = batch
                qtype_iter = None

            # Transfer input data to GPU
            obj_feat_iter = obj_feat_iter.cuda()
            bbox_feat_iter = bbox_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()
            ans_iter = ans_iter.cuda()

            # --- Online Detection Logic ---
            if detection_module is not None:
                with torch.no_grad():
                    obj_feat_iter, bbox_feat_iter = detection_module(obj_feat_iter, bbox_feat_iter)
            # ------------------------------

            loss_tmp = 0
            for accu_step in range(__C.GRAD_ACCU_STEPS):

                # Slicing for Gradient Accumulation
                sub_obj_feat_iter = obj_feat_iter[accu_step * __C.SUB_BATCH_SIZE : (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_bbox_feat_iter = bbox_feat_iter[accu_step * __C.SUB_BATCH_SIZE : (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ques_ix_iter = ques_ix_iter[accu_step * __C.SUB_BATCH_SIZE : (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ans_iter = ans_iter[accu_step * __C.SUB_BATCH_SIZE : (accu_step + 1) * __C.SUB_BATCH_SIZE]
                if qtype_iter is not None:
                    sub_qtype_iter = qtype_iter[accu_step * __C.SUB_BATCH_SIZE : (accu_step + 1) * __C.SUB_BATCH_SIZE]
                else:
                    sub_qtype_iter = None

                # Forward Pass
                pred = net(sub_obj_feat_iter, sub_bbox_feat_iter, sub_ques_ix_iter)

                # Prepare loss items
                loss_item = [pred, sub_ans_iter]
                loss_nonlinear_list = __C.LOSS_FUNC_NONLINEAR[__C.LOSS_FUNC]
                for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                    if loss_nonlinear == 'flat':
                        loss_item[item_ix] = loss_item[item_ix].view(-1)
                    elif loss_nonlinear:
                        loss_item[item_ix] = eval('F.' + loss_nonlinear + '(loss_item[item_ix], dim=1)')

                # Primary Loss (with label smoothing if enabled)
                loss = loss_fn(loss_item[0], loss_item[1])

                # --- Knowledge Distillation Loss ---
                if teacher_net is not None and teacher_feat_iter is not None:
                    # Slice teacher features
                    sub_teacher_feat_iter = teacher_feat_iter[accu_step * __C.SUB_BATCH_SIZE : (accu_step + 1) * __C.SUB_BATCH_SIZE]
                    
                    with torch.no_grad():
                        teacher_pred = teacher_net(sub_teacher_feat_iter, sub_bbox_feat_iter, sub_ques_ix_iter)
                    
                    # KL Divergence Loss: T=2.0
                    T = 2.0
                    student_log_probs = F.log_softmax(pred / T, dim=1)
                    teacher_probs = F.softmax(teacher_pred / T, dim=1)
                    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='sum') * (T * T)
                    
                    # Scale down CE and add KD logic (alpha=0.2, beta=0.8)
                    loss = 0.2 * loss + 0.8 * kd_loss

                # ================================================
                # CORRECT Count Loss — uses question type, not answer index
                # ================================================
                actual_net = net.module if hasattr(net, 'module') else net

                if sub_qtype_iter is not None and is_fusion:
                    # count questions have qtype == 1
                    count_mask = (sub_qtype_iter == 1)

                    if count_mask.any():
                        pred_logits = loss_item[0] if loss_item[0].dim() > 1 else pred
                        true_ans = loss_item[1].view(-1)

                        # 1) Soft expected-value count loss (differentiable)
                        count_probs = torch.softmax(pred_logits[:, :11], dim=1)  # (B, 11)
                        count_indices = torch.arange(11, dtype=torch.float32, device=pred_logits.device)
                        expected_count = (count_probs * count_indices.unsqueeze(0)).sum(dim=1)
                        true_count = true_ans.float()
                        soft_count_loss = F.smooth_l1_loss(
                            expected_count[count_mask],
                            true_count[count_mask].clamp(0, 10)
                        )
                        loss = loss + count_loss_weight * soft_count_loss

                        # 2) Auxiliary count head loss (fusion model)
                        if hasattr(actual_net, '_count_logits') and actual_net._count_logits is not None:
                            count_logits = actual_net._count_logits
                            count_targets = true_ans[count_mask].clamp(0, 10)
                            count_ce_loss = F.cross_entropy(
                                count_logits[count_mask],
                                count_targets
                            )
                            loss = loss + count_loss_weight * count_ce_loss
                            count_loss_sum += count_ce_loss.item()
                            count_loss_steps += 1


                # Track fusion gate values for diagnostics
                if hasattr(actual_net, '_fusion_gate_mean'):
                    gate_sum += actual_net._fusion_gate_mean
                    gate_count += 1
                # ================================================

                if __C.LOSS_REDUCTION == 'mean':
                    loss /= __C.GRAD_ACCU_STEPS

                loss.backward()

                loss_tmp += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                loss_sum += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS

            # Progress Bar
            if dataset_eval is not None:
                mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['val']
            else:
                mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['test']

            # Enhanced progress with fusion diagnostics
            extra_info = ""
            if is_fusion and gate_count > 0:
                avg_gate = gate_sum / gate_count
                extra_info = f", Gate: {avg_gate:.3f}"
            if count_loss_steps > 0:
                avg_closs = count_loss_sum / count_loss_steps
                extra_info += f", CntL: {avg_closs:.3f}"

            print("\r[Version %s][Dataset %s][Epoch %2d][Step %4d/%4d][%s] Loss: %.4f, Lr: %.2e%s" % (
                __C.VERSION, __C.MODEL_USE, epoch + 1, step, int(data_size / __C.BATCH_SIZE),
                mode_str, loss_tmp / __C.SUB_BATCH_SIZE, optim._rate, extra_info), end='          ')

            # Gradient clipping — always clip for fusion mode
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(net.parameters(), __C.GRAD_NORM_CLIP)

            for name in range(len(named_params)):
                norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                    if named_params[name][1].grad is not None else 0
                grad_norm[name] += norm_v * __C.GRAD_ACCU_STEPS

            optim.step()

        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {}s'.format(int(elapse_time)))
        epoch_finish = epoch + 1

        # Checkpoint Saving
        if __C.N_GPU > 1:
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()

        state = {
            'state_dict': state_dict,
            'optimizer': optim.optimizer.state_dict(),
            'lr_base': optim.lr_base,
            'epoch': epoch_finish
        }
        # Save epoch-specific checkpoint
        torch.save(state, ckpt_dir + '/epoch' + str(epoch_finish) + '.pkl')
        # Always overwrite latest.pkl so we can resume after interrupts
        torch.save(state, ckpt_dir + '/latest.pkl')
        print(f'  [Checkpoint saved: epoch{epoch_finish}.pkl + latest.pkl]')

        # Logging
        epoch_avg_loss = loss_sum / data_size
        with open(logfile_path, 'a+') as logfile:
            logfile.write('Epoch: ' + str(epoch_finish) +
                          ', Loss: ' + str(epoch_avg_loss) +
                          ', Lr: ' + str(optim._rate) + '\n' +
                          'Elapsed time: ' + str(int(elapse_time)) +
                          ', Speed(s/batch): ' + str(elapse_time / (step + 1)) + '\n\n')

        # Eval after frequency
        if __C.EVAL_FREQUENCY != 0 and epoch % __C.EVAL_FREQUENCY == 0:
            test_engine(__C, dataset_eval, state_dict=state_dict, save_eval_result=False)

        # --- Early stopping check ---
        if epoch_avg_loss < best_eval_loss:
            best_eval_loss = epoch_avg_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(state, ckpt_dir + '/best.pkl')
            print(f'  [Best model updated at epoch {epoch_finish}, loss={epoch_avg_loss:.6f}]')
        else:
            epochs_without_improvement += 1
            print(f'  [No improvement for {epochs_without_improvement}/{early_stop_patience} epochs]')

        if epochs_without_improvement >= early_stop_patience:
            print(f'\n>>> Early stopping triggered after {epoch_finish} epochs (no improvement for {early_stop_patience} epochs) <<<')
            break

        loss_sum = 0
        grad_norm = np.zeros(len(named_params))
