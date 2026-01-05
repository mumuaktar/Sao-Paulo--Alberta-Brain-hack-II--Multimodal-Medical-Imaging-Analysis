#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 12:57:05 2026

@author: mumuaktar
"""

import torch
import numpy as np
from monai.transforms import MapTransform
from monai.transforms import Compose
from monai.transforms import (
     ToTensord,
    LoadImaged,
RandSpatialCropd,
NormalizeIntensityd
)

class ConvertToMultiChannelBasedOnCustomBratsClassesd(MapTransform):
    """
    Converts label values to multi-channel format for BraTS-like task.
    Your dataset label IDs:
    - 1: necrosis/NCR
    - 2: edema
    - 3: enhancing tumor (ET)

    Channels:
    - Channel 0: Tumor Core (TC) = 1 + 3
    - Channel 1: Whole Tumor (WT) = 1 + 2 + 3
    - Channel 2: Enhancing Tumor (ET) = 3
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]  # (C, H, W, D) or (H, W, D)

            if isinstance(seg, torch.Tensor):
                seg = seg.numpy()




            # make sure we're working with 3D (no extra channel dim)
            if seg.ndim == 4 and seg.shape[0] == 1:
                seg = np.squeeze(seg, axis=0)

            seg = np.where(seg == 4, 3, seg)
            tc = np.logical_or(seg == 1, seg == 3)   # Tumor Core
            wt = np.logical_or(tc, seg == 2)         # Whole Tumor
            et = seg == 3                             # Enhancing Tumor

            multi_channel = np.stack([tc, wt, et], axis=0).astype(np.float32)  # (3, H, W, D)
            d[key] = multi_channel
        return d

def print_shape(d):
    for k, v in d.items():
        print(f"{k}: {v.shape}")
    return d


class LoadNumpyd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys)
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key '{key}' not found in data and allow_missing_keys=False")

            arr = np.load(d[key])  # (1, 128, 768)
            arr = np.squeeze(arr, axis=0)  # (128, 768)
            arr = arr.astype(np.float32)

            d[key] = arr
        return d



train_transforms = Compose([
    LoadImaged(keys=["img",  "seg"], allow_missing_keys=True, ensure_channel_first=True),
    LoadNumpyd(keys=["text_feature"],allow_missing_keys=True),
    ConvertToMultiChannelBasedOnCustomBratsClassesd(keys="seg", allow_missing_keys=True),
    RandSpatialCropd(keys=["img", "seg"], roi_size=(96,96,96), allow_missing_keys=True),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    ToTensord(keys=["img", "seg","text_feature"], dtype=torch.float32, allow_missing_keys=True),    
])    

val_transforms = Compose([
    LoadImaged(keys=["img",  "seg"], ensure_channel_first=True,allow_missing_keys=True),
    LoadNumpyd(keys=["text_feature"],allow_missing_keys=True),
    ConvertToMultiChannelBasedOnCustomBratsClassesd(keys="seg",allow_missing_keys=True),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    ToTensord(keys=["img", "seg","text_feature"],dtype=torch.float32, allow_missing_keys=True),
])
test_transforms = Compose([
    LoadImaged(keys=["img",  "seg"], ensure_channel_first=True,allow_missing_keys=True),
    LoadNumpyd(keys=["text_feature"],allow_missing_keys=True),
    ConvertToMultiChannelBasedOnCustomBratsClassesd(keys="seg",allow_missing_keys=True),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    ToTensord(keys=["img", "seg","text_feature"],dtype=torch.float32, allow_missing_keys=True),
])
