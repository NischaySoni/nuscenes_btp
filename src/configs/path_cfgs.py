# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

import os

class PATH:
    def __init__(self):
        self.init_path()


    def init_path(self):

        self.DATA_ROOT = './data'


        self.FEATS_PATH = {
            'BEVDet': {
                'train': self.DATA_ROOT + '/features/BEVDet',
                'val': self.DATA_ROOT + '/features/BEVDet'},
            'CenterPoint': {
                'train': self.DATA_ROOT + '/features/CenterPoint',
                'val': self.DATA_ROOT + '/features/CenterPoint'},
            'MSMDFusion': {
                'train': self.DATA_ROOT + '/features/MSMDFusion',
                'val': self.DATA_ROOT + '/features/MSMDFusion'},
            'bev': {
                'train': '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/bev_features',
                'val':   '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/bev_features'},
            'yolo': {
                'train': '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/yolo_features_rich',
                'val':   '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/yolo_features_rich'},
            'fusion': {
                'bev': {
                    'train': '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/bev_features',
                    'val':   '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/bev_features'},
                'yolo': {
                    'train': '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/yolo_features_rich',
                    'val':   '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/yolo_features_rich'}},
            'annot': {
                'train': '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/annotation_features',
                'val':   '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/annotation_features'},
            'detected': {
                'train': '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/detected_features',
                'val':   '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/detected_features'},
            'radarxf': {
                'train': '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/radarxf_features_v2',
                'val':   '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/radarxf_features_v2'},
        }
        self.VISUAL_FEATURE = 'bev'


        self.RAW_PATH = {
            'train': self.DATA_ROOT + '/questions' + '/NuScenes_train_questions.json',
            'val': self.DATA_ROOT + '/questions' + '/NuScenes_val_questions.json'}


        self.SPLIT = {
            'train': 'train',
            'val': 'val',
        }


        self.LOG_PATH = './outputs/log'
        self.CKPTS_PATH = './outputs/ckpts'
        self.RESULT_PATH = './outputs/result'


        if 'log' not in os.listdir('./outputs'):
            os.mkdir('./outputs/log')

        if 'ckpts' not in os.listdir('./outputs'):
            os.mkdir('./outputs/ckpts')
        
        if 'result' not in os.listdir('./outputs'):
            os.mkdir('./outputs/result')


    def check_path(self, vis_feat):
        print('Checking Data Path ........')

        if vis_feat == 'fusion':
            # Fusion mode: nested structure with bev and yolo sub-keys
            for sub_feat in ['bev', 'yolo']:
                for split in self.FEATS_PATH['fusion'][sub_feat]:
                    p = self.FEATS_PATH['fusion'][sub_feat][split]
                    if not os.path.exists(p):
                        print(p, 'NOT EXIST')
                        exit(-1)
        else:
            for item in self.FEATS_PATH[vis_feat]:
                if not os.path.exists(self.FEATS_PATH[vis_feat][item]):
                    print(self.FEATS_PATH[vis_feat][item], 'NOT EXIST')
                    exit(-1)

        for item in self.RAW_PATH:
            if not os.path.exists(self.RAW_PATH[item]):
                print(self.RAW_PATH[item], 'NOT EXIST')
                exit(-1)

        print('Data Path Checking Finished!')
        print('')