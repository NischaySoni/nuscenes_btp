# ------------------------------------------------------------------
# Modified NuScenes-QA Dataset Loader
# Supports: BEV (80×69), YOLO (80×13), and Fusion (both)
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
            # Single-feature mode (existing behavior)
            feat_dir = __C.FEATS_PATH[__C.VISUAL_FEATURE][split]
            self.stk2featpath = {
                os.path.basename(p).split('.')[0]: p
                for p in glob.glob(feat_dir + '/*.npy')
            }

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

        # --------------------------
        # ---- Class Weights ----
        # --------------------------
        train_qa = qa_dict['train']['questions']
        ans_counts = np.zeros(self.ans_size)
        for item in train_qa:
            ans_str = str(item['answer'])
            if ans_str in self.ans2ix:
                ans_counts[self.ans2ix[ans_str]] += 1
                
        ans_counts = np.maximum(ans_counts, 1) # avoid div by zero
        total_samples = np.sum(ans_counts)
        # Use inverse frequency, clamped to prevent exploding gradients for extremely rare classes
        weights = total_samples / (self.ans_size * ans_counts)
        weights = np.clip(weights, 0.2, 5.0) 
        self.class_weights = torch.from_numpy(weights).float()

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
            bev_feat = self._load_feat_safe(scene_token, 'bev', (80, 69))
            yolo_feat = self._load_feat_safe(scene_token, 'yolo', (80, 13))

            return (
                torch.from_numpy(bev_feat),
                torch.from_numpy(yolo_feat),    # repurpose bbox_feat slot
                torch.from_numpy(ques_ix),
                torch.from_numpy(ans),
                torch.tensor(qtype_ix, dtype=torch.long),  # question type
            )
        else:
            obj_feat = self.load_obj_feats(scene_token)
            bbox_feat = np.zeros((80, 4), dtype=np.float32)

            return (
                torch.from_numpy(obj_feat),
                torch.from_numpy(bbox_feat),
                torch.from_numpy(ques_ix),
                torch.from_numpy(ans),
                torch.tensor(qtype_ix, dtype=torch.long),  # question type
            )


    def _load_feat_safe(self, scene_token, feat_type, expected_shape):
        """Load a feature file, zero-padding if missing."""
        if feat_type == 'bev':
            path_map = self.stk2bevpath
        else:
            path_map = self.stk2yolopath

        if scene_token in path_map:
            feat = np.load(path_map[scene_token], mmap_mode='r').astype(np.float32)
            feat = (feat - feat.mean()) / (feat.std() + 1e-6)

            # Clamp to prevent extreme outliers
            feat = np.clip(feat, -5.0, 5.0)

            # Shape validation
            if feat.shape != expected_shape:
                # Try to handle shape mismatches gracefully
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
        feat_path = self.stk2featpath[scene_token]

        obj_feat = np.load(feat_path, mmap_mode='r').astype(np.float32)
        obj_feat = (obj_feat - obj_feat.mean()) / (obj_feat.std() + 1e-6)

        # Dynamically detect and cache expected shape from first load
        if not hasattr(self, '_feat_shape'):
            self._feat_shape = obj_feat.shape
            print(f'  Feature shape detected: {self._feat_shape}')

        # Validate: must be (NUM_OBJECTS, feat_dim)
        if obj_feat.shape[0] != self._feat_shape[0] or obj_feat.shape[1] != self._feat_shape[1]:
            raise ValueError(f"Feature shape mismatch: expected {self._feat_shape}, got {obj_feat.shape}")

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
