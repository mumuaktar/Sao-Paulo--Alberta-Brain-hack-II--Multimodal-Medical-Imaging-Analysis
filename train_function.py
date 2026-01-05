#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 13:09:19 2026

@author: mumuaktar
"""




import numpy as np
import nibabel as nib
import torch
import os
from monai.data import decollate_batch
import nibabel as nib
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from functools import partial
from monai.inferers import sliding_window_inference
from functools import partial
from monai.losses import DiceLoss
import torch.nn.functional as F
from monai.losses import FocalLoss
import torch.nn as nn
import torch.nn.functional as F

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


def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)            # Move model to cuda:0
    model.train()
    results_dir=os.path.join(directory_name,"results_test")
    os.makedirs(results_dir,exist_ok=True)

    criterion = DiceLoss(to_onehot_y=False, sigmoid=True)
    criterion_ce = nn.BCEWithLogitsLoss()

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    checkpoint_path = os.path.join(directory_name, "best_model_test.pth")
    last_model_path = os.path.join(directory_name, "last_model_test.pth")
    training_results_dir = os.path.join(directory_name, "training_results_test")
    os.makedirs(training_results_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_dice_score=-1.0
    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=device)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        best_dice_score=checkpoint.get('best_dice_score',-1)
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Last model loaded. Resuming training from epoch: {start_epoch}")
        print(f"Resuming with best Dice score: {best_dice_score:.4f}")



    for epoch in range(start_epoch, max_epochs + 1):
        print(f"\n🔁 Epoch {epoch}")
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            img = batch["img"].to(device)
            seg = batch.get("seg").to(device)
            text=batch.get("text_feature").to(device)
            B, C, H, W, D = img.shape

            optimizer.zero_grad()
            pred_seg = model(img,text)

            loss_seg = criterion(pred_seg, seg) + criterion_ce(pred_seg, seg)

            train_loss += loss_seg.item()
            loss_seg.backward()
            optimizer.step()


        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        # ----------------------
        # Validation
        # ----------------------

        model.eval()
        val_loss = 0.0
        dice_scores = []
        import numpy as np

        affine = np.eye(4)

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
                        roi_size=[96,96,96],
                        sw_batch_size=1,
                        predictor=predictor_with_text,
                        overlap=0.5,
                    )

                    # Run inference
                    pred_seg = model_inferer_with_text(img)
                    # val_output_convert = [post_pred(post_sigmoid(p)) for p in pred_seg]
                    # pred_seg = [p for p in zip(val_output_convert)]
                    # true_seg = [s for s in zip(seg)]
                    pred = post_pred(post_sigmoid(pred_seg))



                    dice_metric(y_pred=pred, y=seg)
                    for i in range(img.shape[0]):
                        subject_id = batch["subject_id"][i]


                        img_paths = [os.path.join(directory_name, "imagesTr", f"{subject_id}_0000.nii")]

                        img_path = img_paths[0]
                        save_filename = f"{subject_id}"

                        save_img_path = os.path.join(results_dir, f"{save_filename}_gt.nii")
                        save_pred_path = os.path.join(results_dir, f"{save_filename}_pred.nii")

                        affine = nib.load(img_path).affine
                        pred_np = pred[i].detach().cpu().numpy().astype(np.uint8)
                        seg_np = seg[i].detach().cpu().numpy().astype(np.uint8)            
                        single_channel_pred = convert_to_single_channel(pred_np)
                        single_channel_gt = convert_to_single_channel(seg_np)
                                   # Save this single-channel prediction as NIfTI
                        nib.save(nib.Nifti1Image(single_channel_pred, affine), save_pred_path)
                        nib.save(nib.Nifti1Image(single_channel_gt, affine), save_img_path)


        per_class_dice, _ = dice_metric.aggregate()
        mean_dice = per_class_dice.mean().item()
        print(f"Dice Scores — TC: {per_class_dice[0].item():.4f}, "
                  f"WT: {per_class_dice[1].item():.4f}, ET: {per_class_dice[2].item():.4f}")
        print(f"Mean Dice: {mean_dice:.4f}")

        if mean_dice > best_dice_score:
                best_dice_score = mean_dice
                torch.save({
                    'epoch': epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    # "val_loss": val_loss,
                    "best_dice_score": best_dice_score
                }, checkpoint_path)
                print("Best model saved based on Dice score.")




        torch.save({
            'epoch': epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            # "val_loss": val_loss,
            # "best_val_loss": best_val_loss,
            "best_dice_score":best_dice_score
        }, last_model_path)

        # Step the scheduler
        scheduler.step()
    print("Training complete.")

