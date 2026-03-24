import os
import numpy as np
from tqdm import tqdm

FEAT_DIR = "bev_features"
OUT_DIR = "stats"
os.makedirs(OUT_DIR, exist_ok=True)

files = os.listdir(FEAT_DIR)[:2000]  # sample subset

X = []
for f in tqdm(files):
    X.append(np.load(os.path.join(FEAT_DIR, f)))

X = np.concatenate(X, axis=0)  # (N*80, 69)

mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-6

np.save(os.path.join(OUT_DIR, "bev_mean.npy"), mean)
np.save(os.path.join(OUT_DIR, "bev_std.npy"), std)

print("✅ Saved BEV mean/std:", mean.shape)
