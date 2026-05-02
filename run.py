# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------
import yaml
import argparse
from src.models.model_loader import CfgLoader
from src.execution.exec import Execution

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--RUN', dest='RUN_MODE',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      type=str, required=True)

    parser.add_argument('--MODEL', dest='MODEL',
                      choices=[
                           'mcan_small',
                           'mcan_small_yolo',
                           'mcan_small_fusion',
                           'mcan_annotation',
                           'mcan_detected',
                           'mcan_radarxformer',
                           'mcan_radarxf_fusion',
                           'mcan_trimodal_fusion',
                           'mcan_trimodal_v2',
                           'mcan_trimodal_v3',
                           'mcan_trimodal_v4',
                           'mcan_trimodal_v5',
                           'mcan_trimodal_v6',
                           'mcan_centerpoint_fusion',
                           'mcan_centerpoint',
                           'mcan_large',
                           'ban_4',
                           'ban_8',
                           'mfb',
                           'mfh',
                           'mem',
                           'butd',
                           'mmnasnet'
                           ]
                        ,
                      help='{'
                           'mcan_small,'
                           'mcan_large,'
                           'ban_4,'
                           'ban_8,'
                           'mfb,'
                           'mfh,'
                           'butd,'
                           'mmnasnet,'
                           '}'
                        ,
                      type=str, required=True)
    
    parser.add_argument('--VIS_FEAT', dest='VISUAL_FEATURE',
                         choices=[
                            'BEVDet',
                            'CenterPoint',
                            'MSMDFusion',
                             'bev',
                             'yolo',
                             'fusion',
                             'annot',
                             'detected',
                             'radarxf',
                             'radarxf_fusion',
                             'trimodal_fusion',
                             'centerpoint_fusion',
                             'centerpoint_only'
                            ],
                         help='{'
                              'BEVDet,'
                              'CenterPoint,'
                              'MSMDFusion,'
                              'bev,'
                              'yolo,'
                              '}',
                         type=str, required=False)

    parser.add_argument('--EVAL_FREQ', dest='EVAL_FREQUENCY',
                      help='number of epochs between each evaluation',
                      type=int)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                      help='batch size in training',
                      type=int)

    parser.add_argument('--GPU', dest='GPU',
                      help="gpu choose, eg.'0, 1, 2, ...'",
                      type=str)

    parser.add_argument('--SEED', dest='SEED',
                      help='fix random seed',
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                      help='version control',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                      choices=['True', 'False'],
                      help='True: use checkpoint to resume training,'
                           'False: start training with random init',
                      type=str)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead, it will override'
                           'CKPT_VERSION and CKPT_EPOCH',
                      type=str)

    parser.add_argument('--FINETUNE_FROM', dest='FINETUNE_FROM',
                      help='Path to a checkpoint (.pkl) to fine-tune from. '
                           'This loads the model weights but starts train at '
                           'Epoch 0 with a fresh optimizer.',
                      type=str)

    parser.add_argument('--USE_KD', dest='USE_KD',
                      choices=['True', 'False'],
                      help='True: Use Knowledge Distillation (Teacher-Student).',
                      type=str)

    parser.add_argument('--TEACHER_CKPT', dest='TEACHER_CKPT',
                      help='Path to the perfectly trained Teacher checkpoint.',
                      type=str)
                      
    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                      help='split batch to reduce gpu memory usage',
                      type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                      choices=['True', 'False'],
                      help='True: use pin memory, False: not use pin memory',
                      type=str)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg_file = 'configs/{}.yaml'.format(args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    __C = CfgLoader(yaml_dict['MODEL_USE']).load()
    args = __C.str_to_bool(args)
    args_dict = __C.parse_to_dict(args)
    

    # Only override YAML with CLI args that were explicitly set (not None)
    args_dict_clean = {k: v for k, v in args_dict.items() if v is not None}
    args_dict = {**yaml_dict, **args_dict_clean}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)
    
    execution = Execution(__C)
    execution.run(__C.RUN_MODE)