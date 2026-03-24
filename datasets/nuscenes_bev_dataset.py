import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class NuScenesBEVQADataset(Dataset):
    def __init__(self, qa_json, feat_dir, mean_path, std_path):
        with open(qa_json, "r") as f:
            self.data = json.load(f)

        self.feat_dir = feat_dir
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        sample_token = item["sample_token"]
        question = item["question"]
        answer = item["answer"]

        # ---------- LOAD BEV FEATURES ----------
        feat_path = os.path.join(self.feat_dir, f"{sample_token}.npy")
        bev_feat = np.load(feat_path)          # (80, 69)
        bev_feat = (bev_feat - self.mean) / self.std

        bev_feat = torch.tensor(bev_feat, dtype=torch.float32)

        return {
            "visual_feats": bev_feat,   # (80, 69)
            "question": question,
            "answer": answer
        }
