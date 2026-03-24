# ------------------------------------------------------------------
# Modified NuScenes-QA Dataset Loader
# Image + Radar BEV Features (80 × 69)
# ------------------------------------------------------------------

import os
import json
import re
import glob
import numpy as np
import torch
import torch.utils.data as Data
import en_core_web_lg


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
        ques_ix, ans, scene_token = self.load_ques_ans(idx)
        obj_feat = self.load_obj_feats(scene_token)

        # dummy bbox (not used)
        bbox_feat = np.zeros((80, 4), dtype=np.float32)

        return (
            torch.from_numpy(obj_feat),
            torch.from_numpy(bbox_feat),
            torch.from_numpy(ques_ix),
            torch.from_numpy(ans)
        )


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

        return ques_ix, ans, scene_token


    # ------------------------------------------------
    # ------------- Load BEV Features ----------------
    # ------------------------------------------------
    def load_obj_feats(self, scene_token):
        feat_path = self.stk2featpath[scene_token]

        obj_feat = np.load(feat_path, mmap_mode='r').astype(np.float32)
        obj_feat = (obj_feat - obj_feat.mean()) / (obj_feat.std() + 1e-6)


        # ensure fixed shape
        if obj_feat.shape != (80, 69):
            raise ValueError(f"Invalid feature shape {obj_feat.shape}")

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
