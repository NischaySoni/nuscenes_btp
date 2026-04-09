# ------------------------------------------------------------------
# Modified NuScenes-QA Dataset Loader
# Supports: BEV, YOLO, Fusion (BEV+YOLO), and Annotation (annot)
# Returns question_type for type-aware training losses
# ------------------------------------------------------------------

import os
import json
import re
import glob
import numpy as np
import torch
import torch.utils.data as Data
import en_core_web_lg
from pathlib import Path


# Question type mapping
QTYPE_MAP = {
    'exist': 0,
    'count': 1,
    'object': 2,
    'status': 3,
    'comparison': 4,
}


class NuScenes_QA(Data.Dataset):
    def __init__(self, __C):
        super(NuScenes_QA).__init__()
        self.__C = __C

        # --------------------------
        # ---- Load QA JSON ----
        # --------------------------
        qa_dict = {
            'train': json.load(open(__C.RAW_PATH['train'], 'r')),
            'val': json.load(open(__C.RAW_PATH['val'], 'r'))
        }

        split = __C.SPLIT[__C.RUN_MODE]
        self.qa_list = qa_dict[split]['questions']
        self.data_size = len(self.qa_list)

        print(split, 'dataset size:', self.data_size)

        # --------------------------
        # ---- Feature paths ----
        # --------------------------
        self.is_fusion = (__C.VISUAL_FEATURE == 'fusion')
        self.is_annot = (__C.VISUAL_FEATURE in ('annot', 'detected', 'radarxf'))

        if self.is_fusion:
            # Fusion mode: load from both BEV and YOLO directories
            bev_dir = __C.FEATS_PATH['fusion']['bev'][split]
            yolo_dir = __C.FEATS_PATH['fusion']['yolo'][split]

            self.stk2bevpath = {
                os.path.basename(p).split('.')[0]: p
                for p in glob.glob(bev_dir + '/*.npy')
            }
            self.stk2yolopath = {
                os.path.basename(p).split('.')[0]: p
                for p in glob.glob(yolo_dir + '/*.npy')
            }

            # Find common tokens (samples available in BOTH feature sets)
            common = set(self.stk2bevpath.keys()) & set(self.stk2yolopath.keys())
            bev_only = set(self.stk2bevpath.keys()) - common
            yolo_only = set(self.stk2yolopath.keys()) - common

            print(f'  [Fusion] BEV features: {len(self.stk2bevpath)}, '
                  f'YOLO features: {len(self.stk2yolopath)}, '
                  f'Common: {len(common)}')
            if bev_only:
                print(f'  [Fusion] {len(bev_only)} samples have BEV only (will zero-pad YOLO)')
            if yolo_only:
                print(f'  [Fusion] {len(yolo_only)} samples have YOLO only (will zero-pad BEV)')

        else:
            # Single-feature mode: BEV, YOLO, or Annotation
            feat_dir = __C.FEATS_PATH[__C.VISUAL_FEATURE][split]
            self.stk2featpath = {
                os.path.basename(p).split('.')[0]: p
                for p in glob.glob(feat_dir + '/*.npy')
            }
            print(f'  [{__C.VISUAL_FEATURE}] Loaded {len(self.stk2featpath)} feature files')

        # --------------------------
        # ---- Tokenization ----
        # --------------------------
        self.token2ix, self.pretrained_emb = self.tokenize(qa_dict)
        self.token_size = len(self.token2ix)

        # --------------------------
        # ---- Answer dict ----
        # --------------------------
        self.ans2ix, self.ix2ans = json.load(
            open('./src/datasets/answer_dict.json', 'r')
        )
        self.ans_size = len(self.ans2ix)

        print('Data Loading Finished!\n')


    # ------------------------------------------------
    # ---------------- Tokenization ------------------
    # ------------------------------------------------
    def tokenize(self, qa_dict):
        token2ix = {'PAD': 0, 'UNK': 1, 'CLS': 2}
        spacy_tool = en_core_web_lg.load()

        pretrained_emb = [
            spacy_tool('PAD').vector,
            spacy_tool('UNK').vector,
            spacy_tool('CLS').vector
        ]

        ques_list = []
        for split in qa_dict:
            for item in qa_dict[split]['questions']:
                ques_list.append(item['question'])

        for ques in ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques.lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token2ix:
                    token2ix[word] = len(token2ix)
                    pretrained_emb.append(spacy_tool(word).vector)

        return token2ix, np.array(pretrained_emb)


    # ------------------------------------------------
    # ---------------- Dataset API -------------------
    # ------------------------------------------------
    def __len__(self):
        return self.data_size


    def __getitem__(self, idx):
        ques_ix, ans, scene_token, qtype_ix = self.load_ques_ans(idx)

        if self.is_fusion:
            bev_shape = tuple(self.__C.FEAT_SIZE['OBJ_FEAT_SIZE']) if 'OBJ_FEAT_SIZE' in self.__C.FEAT_SIZE else (80, 69)
            yolo_shape = tuple(self.__C.FEAT_SIZE['BBOX_FEAT_SIZE']) if 'BBOX_FEAT_SIZE' in self.__C.FEAT_SIZE else (80, 13)

            bev_feat = self._load_feat_safe(scene_token, 'bev', bev_shape)
            yolo_feat = self._load_feat_safe(scene_token, 'yolo', yolo_shape)

            return (
                torch.from_numpy(bev_feat),
                torch.from_numpy(yolo_feat),
                torch.from_numpy(ques_ix),
                torch.from_numpy(ans),
                torch.tensor(qtype_ix, dtype=torch.long),
            )
        else:
            obj_feat = self.load_obj_feats(scene_token)
            obj_shape = tuple(self.__C.FEAT_SIZE['OBJ_FEAT_SIZE']) if 'OBJ_FEAT_SIZE' in self.__C.FEAT_SIZE else (80, 69)
            bbox_feat = np.zeros((obj_shape[0], 4), dtype=np.float32)

            # --- Knowledge Distillation ---
            if getattr(self.__C, 'USE_KD', 'False') == 'True' and self.__C.RUN_MODE == 'train':
                teacher_feat = self._load_feat_safe(scene_token, 'annot', obj_shape)
                return (
                    torch.from_numpy(obj_feat),
                    torch.from_numpy(teacher_feat), # Extra tensor for KD
                    torch.from_numpy(bbox_feat),
                    torch.from_numpy(ques_ix),
                    torch.from_numpy(ans),
                    torch.tensor(qtype_ix, dtype=torch.long),
                )

            return (
                torch.from_numpy(obj_feat),
                torch.from_numpy(bbox_feat),
                torch.from_numpy(ques_ix),
                torch.from_numpy(ans),
                torch.tensor(qtype_ix, dtype=torch.long),
            )


    # Columns in YOLO features that are categorical (not continuous)
    # dim 9 = class_id (0-79 COCO class), dim 12 = radar_match (0/1 flag)
    YOLO_CATEGORICAL_DIMS = {9, 12}

    # RadarXFormer feature layout constants
    RADARXF_STRUCT_DIM = 16
    RADARXF_CATEGORICAL_DIMS = {0, 1}  # category_id, attribute_id

    def _load_feat_safe(self, scene_token, feat_type, expected_shape):
        """Load a feature file, zero-padding if missing."""
        if self.is_fusion:
            if feat_type == 'bev':
                path_map = self.stk2bevpath
            else:
                path_map = self.stk2yolopath
            
            if scene_token in path_map:
                feat = np.load(path_map[scene_token], mmap_mode='r').astype(np.float32)
        else:
            # Single or KD Mode
            if feat_type == 'annot' and getattr(self.__C, 'USE_KD', 'False') == 'True':
                # Dynamically looking up the KD teacher feature
                split = self.__C.SPLIT.get(self.__C.RUN_MODE, 'train')
                
                # Try path configs first, fallback to base cfgs
                if hasattr(self.__C, 'FEATS_PATH') and 'annot' in self.__C.FEATS_PATH:
                    annot_dir = self.__C.FEATS_PATH['annot'][split]
                else:
                    annot_dir = '/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/annotation_features'

                feat_path = os.path.join(annot_dir, f"{scene_token}.npy")
                if os.path.exists(feat_path):
                    feat = np.load(feat_path, mmap_mode='r').astype(np.float32)
                else:
                    return np.zeros(expected_shape, dtype=np.float32)
            else:
                path_map = getattr(self, 'stk2featpath', {})
                if scene_token in path_map:
                    feat = np.load(path_map[scene_token], mmap_mode='r').astype(np.float32)
                else:
                    return np.zeros(expected_shape, dtype=np.float32)

        if 'feat' in locals():

            if feat_type == 'yolo':
                # YOLO: per-column normalization, preserving categorical dims
                for col in range(feat.shape[1]):
                    if col in self.YOLO_CATEGORICAL_DIMS:
                        continue
                    col_data = feat[:, col]
                    col_std = col_data.std()
                    if col_std > 1e-6:
                        feat[:, col] = (col_data - col_data.mean()) / col_std

                for col in range(feat.shape[1]):
                    if col not in self.YOLO_CATEGORICAL_DIMS:
                        feat[:, col] = np.clip(feat[:, col], -5.0, 5.0)

            elif self.__C.VISUAL_FEATURE == 'radarxf':
                # RadarXFormer: separate normalization for structured vs CLIP dims
                struct_dim = self.RADARXF_STRUCT_DIM

                # Identify valid rows to preserve the zero-masking for empty slots!
                valid_mask = np.abs(feat[:, :struct_dim]).sum(axis=1) > 0

                # Standardize CLIP features (dims 16+) ONLY for valid rows
                if valid_mask.any() and feat.shape[1] > struct_dim:
                    clip_part = feat[valid_mask, struct_dim:]
                    clip_std = clip_part.std()
                    if clip_std > 1e-6:
                        feat[valid_mask, struct_dim:] = (clip_part - clip_part.mean()) / clip_std
                    feat[valid_mask, struct_dim:] = np.clip(feat[valid_mask, struct_dim:], -5.0, 5.0)

                # Ensure empty padding slots remain STRICTLY ZERO to keep attention masks working
                feat[~valid_mask, :] = 0.0

            else:
                # BEV / annot / detected: global z-score normalization
                feat = (feat - feat.mean()) / (feat.std() + 1e-6)
                feat = np.clip(feat, -5.0, 5.0)

            # Shape validation
            if feat.shape != expected_shape:
                result = np.zeros(expected_shape, dtype=np.float32)
                r = min(feat.shape[0], expected_shape[0])
                c = min(feat.shape[1], expected_shape[1])
                result[:r, :c] = feat[:r, :c]
                return result

            return feat
        else:
            # Missing feature → return zeros
            return np.zeros(expected_shape, dtype=np.float32)


    # ------------------------------------------------
    # ---------------- Load QA -----------------------
    # ------------------------------------------------
    def load_ques_ans(self, idx):
        item = self.qa_list[idx]

        ques = item['question']
        scene_token = item['sample_token']

        ques_ix = self.proc_ques(ques, max_token=30)

        ans = np.zeros(1, np.int64)
        if self.__C.RUN_MODE == 'train':
            ans[0] = self.ans2ix[str(item['answer'])]

        # Extract question type from template_type field
        template_type = item.get('template_type', 'exist')
        qtype_ix = QTYPE_MAP.get(template_type, 0)

        return ques_ix, ans, scene_token, qtype_ix


    # ------------------------------------------------
    # ------------- Load Features (single mode) ------
    # ------------------------------------------------
    def load_obj_feats(self, scene_token):
        if scene_token not in self.stk2featpath:
            # Missing feature → return zeros
            obj_shape = tuple(self.__C.FEAT_SIZE['OBJ_FEAT_SIZE']) if 'OBJ_FEAT_SIZE' in self.__C.FEAT_SIZE else (80, 69)
            return np.zeros(obj_shape, dtype=np.float32)

        feat_path = self.stk2featpath[scene_token]
        obj_feat = np.load(feat_path, mmap_mode='r').astype(np.float32)

        if self.is_annot:
            if self.__C.VISUAL_FEATURE == 'radarxf':
                struct_dim = self.RADARXF_STRUCT_DIM
                valid_mask = np.abs(obj_feat[:, :struct_dim]).sum(axis=1) > 0

                if valid_mask.any() and obj_feat.shape[1] > struct_dim:
                    clip_part = obj_feat[valid_mask, struct_dim:]
                    clip_std = clip_part.std()
                    if clip_std > 1e-6:
                        obj_feat[valid_mask, struct_dim:] = (clip_part - clip_part.mean()) / clip_std
                    obj_feat[valid_mask, struct_dim:] = np.clip(obj_feat[valid_mask, struct_dim:], -5.0, 5.0)

                obj_feat[~valid_mask, :] = 0.0
            else:
                # Annotation/Detected features are PRE-NORMALIZED in extraction script
                pass
        else:
            # BEV/YOLO: global z-score normalization
            obj_feat = (obj_feat - obj_feat.mean()) / (obj_feat.std() + 1e-6)

        # Dynamically detect and cache expected shape from first load
        if not hasattr(self, '_feat_shape'):
            self._feat_shape = obj_feat.shape
            print(f'  Feature shape detected: {self._feat_shape}')

        # Shape validation with padding
        obj_shape = tuple(self.__C.FEAT_SIZE['OBJ_FEAT_SIZE']) if 'OBJ_FEAT_SIZE' in self.__C.FEAT_SIZE else self._feat_shape
        if obj_feat.shape != obj_shape:
            result = np.zeros(obj_shape, dtype=np.float32)
            r = min(obj_feat.shape[0], obj_shape[0])
            c = min(obj_feat.shape[1], obj_shape[1])
            result[:r, :c] = obj_feat[:r, :c]
            return result

        return obj_feat


    # ------------------------------------------------
    # ---------------- Utils -------------------------
    # ------------------------------------------------
    def proc_ques(self, ques, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words[:max_token]):
            ques_ix[ix] = self.token2ix.get(word, self.token2ix['UNK'])

        return ques_ix