import yaml
import numpy as np
from src.configs.base_cfgs import Cfgs
from src.datasets.nuscenes_qa import NuScenesQA

with open('configs/mcan_trimodal_v12_map.yaml') as f:
    cfg = yaml.safe_load(f)

__C = Cfgs()
for k, v in cfg.items():
    setattr(__C, k, v)
__C.proc()

print(f"BBOX_FEAT_SIZE: {__C.FEAT_SIZE['BBOX_FEAT_SIZE']}")

# Let's see what happens in dataset
ds = NuScenesQA(__C, None)
# get the first sample
for i in range(1):
    res = ds[i]
    bev, rxf, ques, ans, qtype = res
    print("BEV shape:", bev.shape)
    print("RXF shape:", rxf.shape)

