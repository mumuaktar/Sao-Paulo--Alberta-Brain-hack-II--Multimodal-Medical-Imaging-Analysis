#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 13:14:05 2026

@author: mumuaktar
"""
import numpy as np
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import os
import torch
from transforms_function import *

class SwinUNETR_image_text_fusion(nn.Module):
    def __init__(self,
                 # img_size=(),
                 in_channels=4,
                 feature_size=48,
                 seg_out_channels=3,
                 use_checkpoint=True,
                 text_embed_dim=768):
        super().__init__()

        self.backbone = SwinUNETR(
            # img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint)

        # project BioBERT 768 → Swin feature dim (48)
        self.text_proj = nn.Linear(text_embed_dim, feature_size)

        # fuse image+text channels: (2C → C)
        self.fusion = nn.Conv3d(feature_size * 2, feature_size, kernel_size=1)

        # output head
        self.seg_head = nn.Conv3d(feature_size, seg_out_channels, kernel_size=1)


    def forward(self, x, text_feature):

        # --- Image Backbone ---
        img_features = self.backbone(x)  # [B, C, D, H, W]
        # print(f"[Backbone] img_features: {img_features.shape}")

        # --- Text Processing ---
        text_feature = text_feature.mean(dim=1)
        # print('check shape:',text_feature.shape)
        text_feat = self.text_proj(text_feature)          # [B, 768] → [B, C]
        # print(f"[TextProj] text_feat after Linear: {text_feat.shape}")

        text_feat = text_feat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print(f"[TextProj] text_feat after unsqueeze: {text_feat.shape}")

        text_feat = text_feat.expand_as(img_features)     # [B, C, D, H, W]
        # print(f"[TextProj] text_feat after expand_as: {text_feat.shape}")

        fused = torch.cat([img_features, text_feat], dim=1)  # [B, 2C, D, H, W]
        # print(f"[Fusion] concatenated: {fused.shape}")

        fused = self.fusion(fused)                        # [B, C, D, H, W]
        # print(f"[Fusion] after Conv3d fusion: {fused.shape}")

        # --- Head ---
        seg_output = self.seg_head(fused)
        # print(f"[SegHead] seg_output: {seg_output.shape}")


        return seg_output

def convert_to_single_channel(multi_channel_np: np.ndarray) -> np.ndarray:
    """
    Convert BraTS-style one-hot (3, H, W, D) prediction or GT to single-channel label map:
        0: Background
        1: Tumor Core (TC) [label 1 in GT]
        2: Edema [label 2 in GT]
        3: Enhancing Tumor (ET) [label 3 in GT]

    Assumes:
        Channel 0: TC = 1 + 3
        Channel 1: WT = 1 + 2 + 3
        Channel 2: ET = 3
    """
    assert multi_channel_np.shape[0] == 3, "Expected 3 channels (TC, WT, ET)"
    
    tc = multi_channel_np[0]
    et = multi_channel_np[2]

    output = np.zeros_like(tc, dtype=np.uint8)

    # Priority-based assignment
    output[tc == 1] = 1  # TC gets label 1 (includes necrosis and ET)
    output[(multi_channel_np[1] == 1) & (tc == 0) & (et == 0)] = 2  # Edema only gets label 2
    output[et == 1] = 3  # Enhancing Tumor gets label 3 (overwrites TC if needed)

    return output


def test(test_loader, model, input_dir, results_dir):
    import os
    import nibabel as nib
    import numpy as np
    import torch
    from functools import partial
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.utils.enums import MetricReduction

    device = next(model.parameters()).device
    os.makedirs(results_dir, exist_ok=True)

    model.eval()

    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True
    )

    hd95_metric = HausdorffDistanceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        percentile=95.0
    )

    # predictor with text (keep your logic)
    predictor_with_text = lambda x: model(x, text)

    model_inferer_with_text = partial(
        sliding_window_inference,
        roi_size=[96,96,96],
        sw_batch_size=1,
        predictor=predictor_with_text,
        overlap=0.5,
        mode="gaussian"
    )
    # --- RESET metrics BEFORE evaluation ---
    dice_metric.reset()
    hd95_metric.reset()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            subject_id = batch["subject_id"][0]
            img = batch["img"].to(device)
            gt=batch["seg"].to(device)
            text = batch["text_feature"].to(device)
            print('check shape first',img.shape,text.shape,gt.shape)

            # --- Sliding window inference ---
            logits = model_inferer_with_text(img)
            # print('check:',logits)

            pred_prob = torch.sigmoid(logits)
            pred = (pred_prob > 0.5).int()
            # print('check:',pred,logits)
            # --- Accumulate metrics ---
            dice_metric(y_pred=pred, y=gt)
            hd95_metric(y_pred=pred, y=gt)

            ###############saving segmentation#############3
            
           
            pred_np = pred[0].cpu().numpy().astype(np.uint8)
            affine_modality = "0001"
            img_path = os.path.join(input_dir,"imagesTr", f"{subject_id}_{affine_modality}.nii")
            affine = nib.load(img_path).affine
            single_channel_pred = convert_to_single_channel(pred_np)
            save_path = os.path.join(results_dir, f"{subject_id}.nii")
            nib.save(nib.Nifti1Image(single_channel_pred, affine), save_path)
    
            print('saved prediction')

    # --- Aggregate ONCE ---
    print('final:',dice_metric,hd95_metric)
    dice, not_nans = dice_metric.aggregate()
    hd95 = hd95_metric.aggregate()


    print("Before NaN/Inf fix:", hd95)

    # --- Option C: NaN / Inf-safe HD95 (ET-safe) ---
    hd95 = torch.nan_to_num(hd95, nan=0.0, posinf=0.0, neginf=0.0)

    print("After NaN/Inf fix:", hd95)

    dice = dice.cpu().numpy()
    hd95 = hd95.cpu().numpy()

    print(
        f"Dataset Avg Dice -> "
        f"TC={dice[0]:.4f}, WT={dice[1]:.4f}, ET={dice[2]:.4f}"
    )
    print(
        f"Dataset Avg HD95 -> "
        f"TC={hd95[0]:.2f}, WT={hd95[1]:.2f}, ET={hd95[2]:.2f}"
    )





#################test###################################

import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser(description="Swin UNETR for Automated Brain Tumor Segmentation")

    parser.add_argument("--data_dir", default="/home/ai2lab/workshop_February/dataset", type=str, help="Dataset directory")
    parser.add_argument("--output_dir",default="/home/ai2lab/workshop_February/dataset/output",type=str, help="output directory")
  #  parser.add_argument("--save_checkpoint", action="store_true", help="Save checkpoint during training")
   # parser.add_argument("--max_epochs", default=200, type=int, help="Max number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")

    # Detect if running in Jupyter
    if "ipykernel" in sys.modules:
        # Jupyter: ignore sys.argv to prevent conflicts
        return parser.parse_args(args=[])
    else:
        # Standard script: parse normally
        return parser.parse_args()

# Get arguments
args = get_args()

# Example usage
print(args)

directory_name=args.data_dir
output_dir=args.output_dir
from data_loader import *
test_loader = load_data(
    base_path=directory_name,
    split="test",
    transforms=test_transforms,
    batch_size=1,
    shuffle=False,
)



def main():
    

 
    model = SwinUNETR_image_text_fusion(
           # img_size=(256, 256, 160),
           in_channels=4,
           seg_out_channels=3,  
           feature_size=48
       )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    checkpoint_path = os.path.join(directory_name, "best_model_test.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print("Checkpoint loaded.")
    
    # val_loader = load_data_validation(directory_name)
    # print('reached to val_loader')
        # Run inference and save predictions
    with torch.no_grad():
        test(test_loader, model,directory_name, output_dir)  




if __name__ == "__main__":
    main()
