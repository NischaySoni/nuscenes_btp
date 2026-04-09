import torch
import numpy as np
import sys
import yaml
from torch.utils.data import DataLoader

sys.path.append('.')
from src.models.model_loader import CfgLoader
from src.datasets.nuscenes_qa import NuScenes_QA
from src.models.mcan.net import Net

def main():
    print("Initializing Configurations...")
    __C = CfgLoader('mcan').load()
    __C.MODEL = 'mcan_radarxformer'
    
    with open('./configs/mcan_radarxformer.yaml', 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yaml_dict.items():
        setattr(__C, k, v)
    
    __C.VISUAL_FEATURE = 'radarxf'
    __C.RUN_MODE = 'val'

    print("Initializing Network...")
    net = Net(__C, None, 100, 100) # dummy token_size and ans_size
    net.eval()
    net.to('cpu')

    print("\n--- Running Diagnostic Test (Dummy Data) ---")
    
    # Dummy tensors
    B = 2
    N = 100
    feat_dim = 48
    
    obj_feat = torch.randn(B, N, feat_dim)
    # Mask out some objects (padding)
    obj_feat[0, 50:] = 0
    obj_feat[1, 75:] = 0
    
    bbox_feat = torch.randn(B, N, 4)
    ques_ix = torch.randint(1, 100, (B, 14))
    
    print("\n[Input Features]")
    print(f"obj_feat shape: {obj_feat.shape}")
    
    print("\n[Testing Adapter Forward Pass]")
    with torch.no_grad():
        from src.models.mcan.net import make_mask
        mask = make_mask(obj_feat)
        print(f"make_mask generated shape: {mask.shape}")
        
        try:
            # Manually test the component
            projected, proj_mask = net.radarxf_adapter(obj_feat)
            print(f"RadarXFormerAdapter output shape: {projected.shape}")
        except Exception as e:
            print(f"Adapter Error: {e}")
            
        try:
            # Full test pass
            res = net(obj_feat, bbox_feat, ques_ix)
            print(f"\n[Testing Full Network Pass]")
            print(f"Full output logits shape: {res.shape}")
            print("PASSED! Network runs cleanly.")
        except Exception as e:
            print(f"Network Error: {e}")

if __name__ == "__main__":
    main()
