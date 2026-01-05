"""
Created on Mon Jan  5 12:57:05 2026

@author: mumuaktar
"""

import os
import torch
from monai.networks.nets import SwinUNETR
import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser(description="Swin UNETR for Automated Brain Tumor Segmentation")

    parser.add_argument("--data_dir", default="/home/ai2lab/workshop_February/dataset", type=str, help="Dataset directory")
    parser.add_argument("--output_dir",default="/home/ai2lab/workshop_February/dataset/output",type=str, help="output directory")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save checkpoint during training")
    parser.add_argument("--max_epochs", default=200, type=int, help="Max number of training epochs")
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




# In[9]:


 # === Step 1: Load data ===
from transforms_function import *
from data_loader import *

directory_name=args.data_dir
output_dir=args.output_dir
train_loader = load_data(
    base_path=directory_name,
    split="train",
    transforms=train_transforms,
    batch_size=2,
    shuffle=True,
)

val_loader = load_data(
    base_path=directory_name,
    split="val",
    transforms=val_transforms,
    batch_size=1,
    shuffle=False,
)



for batch in train_loader:
        print(batch["text_feature"].shape, batch["img"].shape, batch["seg"].shape)
        break


# In[14]:



# In[8]:


import torch
from monai.networks.nets import SwinUNETR  # or your custom SwinUNet
from monai.optimizers import generate_param_groups
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from train_function import *
from data_loader import *

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


# In[11]:


def main():
    # === Step 2: Set up device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Step 3: Initialize model ===

    model = SwinUNETR_image_text_fusion(
        # img_size=(),
       in_channels=4,
       seg_out_channels=3,      # tumor classes
       feature_size=48
   )

    # print(model) 

    # === Step 4: parameter setup ===
    start_epoch=1
    max_epochs=200
    from torch.optim.lr_scheduler import CosineAnnealingLR
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)


    # === Step 5: Call train() ===

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=max_epochs,
        directory_name=directory_name,
        start_epoch=start_epoch,

    )



if __name__ == "__main__":
    main()





