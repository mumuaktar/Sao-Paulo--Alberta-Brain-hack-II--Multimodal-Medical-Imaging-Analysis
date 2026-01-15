#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 13:09:19 2026

@author: mumuaktar, dscarmo
"""
# Standard library imports
import os
from functools import partial

# Third-party library imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

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


def train(train_loader, val_loader, model, optimizer, scheduler, config: dict):
    """
    Train the model, monolithic function dealing with model training, evaluation and saving.
    Args:
        train_loader: DataLoader, training data loader
        val_loader: DataLoader, validation data loader
        model: nn.Module, model to train
        optimizer: torch.optim.Optimizer, optimizer
        scheduler: torch.optim.lr_scheduler, learning rate scheduler
        start_epoch: int, start epoch
        config: dict, configuration dictionary
    """
    # Prepare model and output directory
    start_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: nn.Module = model.to(device)            # Move model to cuda:0
    model.train()
    
    # Create output directory based on config name
    output_dir = os.path.join(config['output_base_dir'], config['config_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    results_dir = os.path.join(output_dir, "results_test")
    os.makedirs(results_dir, exist_ok=True)

    # Initialize loss functions, metrics, and auxiliary objects for output processing
    criterion = DiceLoss(to_onehot_y=False, sigmoid=True)  # internal sigmoid!
    criterion_ce = nn.BCEWithLogitsLoss()
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    # Initialize checkpoint paths and directories
    checkpoint_path = os.path.join(output_dir, "best_model_test.pth")
    last_model_path = os.path.join(output_dir, "last_model_test.pth")
    training_results_dir = os.path.join(output_dir, "training_results_test")
    os.makedirs(training_results_dir, exist_ok=True)

    # Initialize best dice score and resume training if last model exists
    best_val_loss = float("inf")
    best_dice_score = -1.0
    
    # Initialize lists to track losses and Dice scores for plotting
    train_losses = []
    val_losses = []
    mean_dice_scores = []
    tc_dice_scores = []
    wt_dice_scores = []
    et_dice_scores = []
    epochs_list = []
    
    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=device)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        best_dice_score = checkpoint.get('best_dice_score', -1)
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Last model loaded. Resuming training from epoch: {start_epoch}")
        print(f"Resuming with best Dice score: {best_dice_score:.4f}")
        
        # Load previous history if available
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        mean_dice_scores = checkpoint.get('mean_dice_scores', [])
        tc_dice_scores = checkpoint.get('tc_dice_scores', [])
        wt_dice_scores = checkpoint.get('wt_dice_scores', [])
        et_dice_scores = checkpoint.get('et_dice_scores', [])
        epochs_list = checkpoint.get('epochs_list', list(range(1, start_epoch)))

    # Each epoch performs one training loop and one validation loop
    for epoch in range(start_epoch, config['max_epochs'] + 1):
        print(f"\n🔁 Epoch {epoch}")
        model.train()
        train_loss = 0.0

        # Training loop
        for batch in train_loader:
            img = batch["img"].to(device)
            seg = batch.get("seg").to(device)
            text = batch.get("text_feature").to(device)
            B, C, H, W, D = img.shape

            optimizer.zero_grad()

            # Get output activations (logits)
            pred_seg = model(img,text)

            # Compute loss, BCE + Dice
            # BCEWithLogitsLoss uses the raw logits
            # DiceLoss uses an internal sigmoid activation
            loss_seg = criterion(pred_seg, seg) + criterion_ce(pred_seg, seg)

            # Accumulate loss
            train_loss += loss_seg.item()

            # Backpropagate loss
            loss_seg.backward()
            optimizer.step()

        # Average loss over all batches
        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        # ----------------------
        # Validation
        # ----------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            dice_metric.reset()
            for batch_idx, batch in enumerate(val_loader):
                img = batch["img"].to(device)
                seg = batch.get("seg").to(device)
                text = batch.get("text_feature").to(device)

                # Define a predictor lambda that includes text_feature
                predictor_with_text = lambda x: model(x, text)

                # Create a sliding_window_inference instance with this predictor
                model_inferer_with_text = partial(
                    sliding_window_inference,
                    roi_size = [96, 96, 96],
                    sw_batch_size = 1,
                    predictor = predictor_with_text,
                    overlap = 0.5,
                )

                # Run inference
                pred_seg = model_inferer_with_text(img)

                # Convert logits to first probabilities and then to discrete predictions
                pred = post_pred(post_sigmoid(pred_seg))

                # Compute validation loss
                loss_seg = criterion(pred_seg, seg) + criterion_ce(pred_seg, seg)
                val_loss += loss_seg.item()

                # Compute Dice score
                dice_metric(y_pred=pred, y=seg)

                # Save predictions and ground truths for each subject
                for i in range(img.shape[0]):
                    subject_id = batch["subject_id"][i]

                    # Get the image path for the original image to build output paths
                    img_paths = [os.path.join(config['data_dir'], "imagesTr", f"{subject_id}_0000.nii")]
                    img_path = img_paths[0]

                    # Build save filenames
                    save_filename = f"{subject_id}"
                    save_img_path = os.path.join(results_dir, f"{save_filename}_gt.nii")
                    save_pred_path = os.path.join(results_dir, f"{save_filename}_pred.nii")
                    
                    # Outputs need to be moved to the CPU and converted to numpy arrays before saving as NIfTI files
                    # Note uint8 type, this saves disk space and is faster to save and load
                    pred_np = pred[i].detach().cpu().numpy().astype(np.uint8)
                    seg_np = seg[i].detach().cpu().numpy().astype(np.uint8)            
                    single_channel_pred = convert_to_single_channel(pred_np)
                    single_channel_gt = convert_to_single_channel(seg_np)
                    
                    # Save this single-channel prediction and ground truth as NIfTI
                    # Loading the original image's affine is important for saving the predictions as NIfTI files with the correct orientation
                    affine = nib.load(img_path).affine
                    nib.save(nib.Nifti1Image(single_channel_pred, affine), save_pred_path)
                    nib.save(nib.Nifti1Image(single_channel_gt, affine), save_img_path)

        # Average validation loss over all batches
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Aggregate Dice scores over all validation samples
        per_class_dice, _ = dice_metric.aggregate()
        mean_dice = per_class_dice.mean().item()
        print(f"Dice Scores — TC: {per_class_dice[0].item():.4f}, "
              f"WT: {per_class_dice[1].item():.4f}, ET: {per_class_dice[2].item():.4f}")
        print(f"Mean Dice: {mean_dice:.4f}")
        
        # Track losses and Dice scores for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mean_dice_scores.append(mean_dice)
        tc_dice_scores.append(per_class_dice[0].item())
        wt_dice_scores.append(per_class_dice[1].item())
        et_dice_scores.append(per_class_dice[2].item())
        epochs_list.append(epoch)
        
        # Plot and save loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs_list, val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save loss plot
        loss_plot_path = os.path.join(output_dir, "loss_curves.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot and save Dice score curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, mean_dice_scores, 'g-', label='Mean Dice', linewidth=2)
        plt.plot(epochs_list, tc_dice_scores, 'b-', label='TC Dice', linewidth=2)
        plt.plot(epochs_list, wt_dice_scores, 'r-', label='WT Dice', linewidth=2)
        plt.plot(epochs_list, et_dice_scores, 'm-', label='ET Dice', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Dice Score', fontsize=12)
        plt.title('Validation Dice Scores', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()
        
        # Save Dice plot
        dice_plot_path = os.path.join(output_dir, "dice_curves.png")
        plt.savefig(dice_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Is this the best Dice score so far?
        if mean_dice > best_dice_score:
            best_dice_score = mean_dice

            # Save best model based on Dice score
            torch.save({
                'epoch': epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice_score": best_dice_score
            }, checkpoint_path)

            print(f"Best model saved based on Dice score: {best_dice_score:.4f}")
        
        # Also update what was the last model
        torch.save({
            'epoch': epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice_score": best_dice_score,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "mean_dice_scores": mean_dice_scores,
            "tc_dice_scores": tc_dice_scores,
            "wt_dice_scores": wt_dice_scores,
            "et_dice_scores": et_dice_scores,
            "epochs_list": epochs_list
        }, last_model_path)

        # Step the learning rate scheduler
        scheduler.step()

    # This is the closing of all loops in the training function. Training is complete!
    print(f"Training complete. Best Dice score: {best_dice_score:.4f}")
