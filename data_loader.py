#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 13:19:13 2026

@author: mumuaktar
"""

import os
import pandas as pd

def build_data_list(df, base_path, label_path, text_feature_root):
    def get_modality_paths(subject_id):
        img_root = os.path.join(base_path, "imagesTr")
        return [
            os.path.join(img_root, f"{subject_id}_0000.nii"),
            os.path.join(img_root, f"{subject_id}_0001.nii"),
            os.path.join(img_root, f"{subject_id}_0002.nii"),
            os.path.join(img_root, f"{subject_id}_0003.nii"),
        ]

    data_list = []
    for sid in df["SubjectID"]:
        item = {
            "img": get_modality_paths(sid),
            "seg": os.path.join(label_path, f"{sid}.nii"),
            "subject_id": sid,
            "text_feature": os.path.join(
                text_feature_root, sid, f"{sid}_flair_text.npy"
            ),
        }
        data_list.append(item)

    return data_list
from monai.data import Dataset
from torch.utils.data import DataLoader

def load_data(
    base_path,
    split,
    transforms,
    batch_size=1,
    shuffle=False,
    num_workers=1,
):
    """
    General data loader for train / val / test

    split: 'train', 'val', or 'test'
    """

    assert split in ["train", "val", "test"], "Invalid split name"

    label_path = os.path.join(base_path, "labelsTr")
    text_feature_root = os.path.join(base_path, "text_data/TextBraTSData")

    csv_map = {
        "train": "imagesTr/train_set.csv",
        "val": "imagesTr/validation_set.csv",
        "test": "imagesTr/test_set.csv",
    }

    csv_path = os.path.join(base_path, csv_map[split])
    print(f"Loading {split} data from:", csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)

    data_list = build_data_list(
        df,
        base_path=base_path,
        label_path=label_path,
        text_feature_root=text_feature_root,
    )

    dataset = Dataset(data=data_list, transform=transforms)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader


if __name__ == "__main__":
    """
    Testing the data loader
    """
    raise NotImplementedError("Not implemented")




