import os
import torch
import argparse
from glob import glob
from pathlib import Path
from load_config import get_config_args


if __name__ == "__main__":
    config = get_config_args()

    print(config)

    # Expected output folder
    output_folder = Path(config["output_base_dir"]) / Path(config["config_name"])

    # Last model details, contains all the metrics for the full run
    last_model_path = os.path.join(output_folder, "last_model_test.pth")
    last_model_dict = torch.load(last_model_path)
    epoch = last_model_dict['epoch']
    print(f"\n{last_model_path}\nepoch {epoch}:\nbest_dice_score {last_model_dict['best_dice_score']}\nmean_dice_score {last_model_dict['mean_dice_scores'][epoch-1]}\ntc_dice_score {last_model_dict['tc_dice_scores'][epoch-1]}\nwt_dice_score {last_model_dict['wt_dice_scores'][epoch-1]}\net_dice_score {last_model_dict['et_dice_scores'][epoch-1]}")

    # Best model details
    best_model_path = os.path.join(output_folder, "best_model_test.pth")
    best_model_dict = torch.load(best_model_path)
    epoch = best_model_dict['epoch']
    print(f"\n{best_model_path}\nepoch {epoch}:\nbest_dice_score {best_model_dict['best_dice_score']}")
