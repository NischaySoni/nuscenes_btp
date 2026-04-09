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

    print("Loading Validation Dataset...")
    dataset = NuScenes_QA(__C)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print("Initializing Network...")
    net = Net(__C, dataset.pretrained_emb, dataset.token_size, dataset.ans_size)
    net.eval()
    net.to('cpu')

    print("\n--- Running Diagnostic Test ---")
    
    # Grab one batch
    for batch in dataloader:
        obj_feat, bbox_feat, ques_ix, ans, qtype_ix = batch
        
        print("\n[Input Features]")
        print(f"obj_feat shape: {obj_feat.shape}")
        
        # Check sparsity (how many valid objects)
        struct_dim = 16
        valid_masks = (torch.sum(torch.abs(obj_feat[:, :, :struct_dim]), dim=-1) > 0)
        
        for b in range(obj_feat.shape[0]):
            n_valid = valid_masks[b].sum().item()
            print(f"  Batch {b}: {n_valid} valid objects detected out of {obj_feat.shape[1]}")
            
            if n_valid > 0:
                # Get stats for valid objects only
                valid_struct = obj_feat[b, valid_masks[b]][:, :struct_dim]
                valid_clip = obj_feat[b, valid_masks[b]][:, struct_dim:]
                print(f"    - Struct Mean: {valid_struct.mean().item():.3f}, Std: {valid_struct.std().item():.3f}")
                if valid_clip.numel() > 0:
                    print(f"    - CLIP Mean: {valid_clip.mean().item():.3f}, Std: {valid_clip.std().item():.3f}")
            
            # Check if padding is perfectly zero
            n_pad = (~valid_masks[b]).sum().item()
            if n_pad > 0:
                pad_features = obj_feat[b, ~valid_masks[b]]
                pad_sum = pad_features.abs().sum().item()
                print(f"    - Padding Area Sum (must be 0.0): {pad_sum}")
        
        print("\n[Testing Adapter Forward Pass]")
        # Test just the Adapter logic
        with torch.no_grad():
            from src.models.mcan.net import make_mask
            mask = make_mask(obj_feat)
            print(f"make_mask generated shape: {mask.shape}")
            print(f"mask zero elements out of 100: {(mask[0, 0, 0, :] == True).sum().item()}")
            
            try:
                # Manually test the component
                projected, _ = net.radarxf_adapter(obj_feat)
                print(f"RadarXFormerAdapter output shape: {projected.shape}")
                print(f"Adapter Output Valid Mean: {projected[0, valid_masks[0]].mean().item():.3f}")
                if (~valid_masks[0]).any():
                    print(f"Adapter Output Pad Mean: {projected[0, ~valid_masks[0]].mean().item():.3f} (Will be ignored by attention mask)")
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
        break

if __name__ == "__main__":
    main()
