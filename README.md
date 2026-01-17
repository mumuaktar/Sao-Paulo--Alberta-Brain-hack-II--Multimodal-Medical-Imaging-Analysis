***Overview of the Challenge Task***
The brain hack challenge focuses on developing a segmentation algorithm delineating the tumor sub-regions: tumor core (TC), enhancing tumor (ET) and the whole tumor (WT). 
***Baseline Model Description***

This repository provides a baseline multimodal brain tumor segmentation model that integrates MRI imaging data with associated textual information from medical reports using the TextBraTS dataset.
The image branch is built upon a transformer-based segmentation backbone (Swin UNETR), which extracts rich volumetric feature representations from multimodal MRI inputs. 
In parallel, textual features are extracted using a pretrained BioBERT encoder and projected into the same latent feature space as the image features through a linear projection layer.

To enable multimodal fusion, the projected text embeddings are spatially broadcast and concatenated with the image feature maps along the channel dimension. 
A lightweight 1×1×1 convolution is then applied to fuse the combined features, producing a unified representation that conditions image-based segmentation on global textual context. 
The fused features are finally passed through a segmentation head to predict tumor sub-regions, including whole tumor (WT), tumor core (TC), and enhancing tumor (ET).

This simple late-fusion strategy serves as a strong and interpretable baseline for text-guided brain tumor segmentation on the BraTS dataset, providing a foundation for more advanced multimodal interaction mechanisms.


***Dataset Description***

This project uses the TextBraTS dataset. Imaging data (4 MRI modalities per case) is downloaded from Kaggle, and corresponding text data is obtained from the official TextBraTS page (https://github.com/Jupitern52/TextBraTS). 
The dataset was saved on the ARC cluster and prepared locally for this project; while the folder structure may differ from the original, the same JSON files from TextBraTS are used for train, validation, and test splits. Images and labels are stored in imagesTr and labelsTr, with the CSV files provided inside imagesTr for easy reference.


***Requirements***

We recommend `uv` for dependency management, but you can also use `pip` with `requirements.txt`.

**Option 1: Using `uv` (recommended)**

1. Install `uv` (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

This will create a local virtual environment and install all required packages as specified in `pyproject.toml`.

**Option 2: Using `conda` with `requirements.txt`**

1. Create a conda environment with Python 3.12:
   ```bash
   conda create -n brainhack python=3.12
   conda activate brainhack
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


***Getting Started***

To run the codes, depending on how you created your environment:

**Option 1: Using `uv` (recommended)**

```bash
uv run python baseline_fusion_model.py configs/debug_config.yaml
uv run python test.py configs/debug_config.yaml
```

**Option 2: Using `conda`**

```bash
conda activate brainhack
python baseline_fusion_model.py configs/debug_config.yaml
python test.py configs/debug_config.yaml
```

Make sure to adjust the config YAML files in the configs folder to point to your source directory and your desired output folder. 
