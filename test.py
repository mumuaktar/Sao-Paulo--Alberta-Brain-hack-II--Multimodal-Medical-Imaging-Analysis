#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 13:14:05 2026

@author: mumuaktar, dscarmo
"""
# Standard library imports
import os
from functools import partial

# Third-party library imports
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction

# Local module imports
from baseline_fusion_model import SwinUNETR_image_text_fusion
from data_loader import load_data
from load_config import get_config_args
from train_function import convert_to_single_channel
from transforms_function import test_transforms


def test(test_loader, model, input_dir: str, results_dir: str):
    """
    Test the model on the test dataset and compute metrics.
    
    Args:
        test_loader: DataLoader for test dataset
        model: Trained model to evaluate
        input_dir: Directory containing input images
        results_dir: Directory to save prediction results
    """
    device = next(model.parameters()).device
    os.makedirs(results_dir, exist_ok=True)

    model.eval()

    # Initialize metrics
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

    # Reset metrics before evaluation
    dice_metric.reset()
    hd95_metric.reset()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            subject_id = batch["subject_id"][0]
            img = batch["img"].to(device)
            gt = batch["seg"].to(device)
            text = batch["text_feature"].to(device)

            # Define a predictor lambda that includes text_feature
            predictor_with_text = lambda x: model(x, text)

            # Create a sliding_window_inference instance with this predictor
            model_inferer_with_text = partial(
                sliding_window_inference,
                roi_size=[96, 96, 96],
                sw_batch_size=1,
                predictor=predictor_with_text,
                overlap=0.5,
                mode="gaussian"
            )

            # Run sliding window inference
            logits = model_inferer_with_text(img)

            # Convert logits to probabilities and then to discrete predictions
            pred_prob = torch.sigmoid(logits)
            pred = (pred_prob > 0.5).int()

            # Accumulate metrics
            dice_metric(y_pred=pred, y=gt)
            hd95_metric(y_pred=pred, y=gt)

            # Save predictions
            pred_np = pred[0].cpu().numpy().astype(np.uint8)
            affine_modality = "0001"
            img_path = os.path.join(input_dir, "imagesTr", f"{subject_id}_{affine_modality}.nii")
            affine = nib.load(img_path).affine
            single_channel_pred = convert_to_single_channel(pred_np)
            save_path = os.path.join(results_dir, f"{subject_id}.nii")
            nib.save(nib.Nifti1Image(single_channel_pred, affine), save_path)

            print(f"Saved prediction for {subject_id}")

    # Aggregate metrics
    dice, not_nans = dice_metric.aggregate()
    hd95 = hd95_metric.aggregate()

    # NaN / Inf-safe HD95 (ET-safe)
    hd95 = torch.nan_to_num(hd95, nan=0.0, posinf=0.0, neginf=0.0)

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

def main():
    """
    Main entry point for testing.
    
    Loads the model, checkpoint, and runs inference on the test dataset.
    """
    # Get configuration from command line arguments and config file
    config = get_config_args(
        description="Swin UNETR for Automated Brain Tumor Segmentation",
        example_usage="Example: python test.py configs/train_config.yaml",
        default_config="configs/train_config.yaml"
    )

    # Initialize data loader
    test_loader = load_data(
        base_path=config['data_dir'],
        split="test",
        transforms=test_transforms,
        batch_size=config['test_batch_size'],
        shuffle=False,
    )

    # Initialize model
    model = SwinUNETR_image_text_fusion(
        in_channels=4,
        seg_out_channels=3,
        feature_size=48
    )

    # Move model to device and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create output directory based on config name
    output_dir = os.path.join(config['output_base_dir'], config['config_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint from the config-based output directory
    checkpoint_path = os.path.join(output_dir, "best_model_test.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print("Checkpoint loaded.")

    # Run inference and save predictions
    with torch.no_grad():
        test(test_loader, model, config['data_dir'], output_dir)


if __name__ == "__main__":
    main()
