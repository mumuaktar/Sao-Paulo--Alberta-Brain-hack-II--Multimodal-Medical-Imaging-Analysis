This repository provides a baseline multimodal brain tumor segmentation model that integrates MRI imaging data with associated textual information from medical reports using the TextBraTS dataset.
The image branch is built upon a transformer-based segmentation backbone (Swin UNETR), which extracts rich volumetric feature representations from multimodal MRI inputs. 
In parallel, textual features are extracted using a pretrained BioBERT encoder and projected into the same latent feature space as the image features through a linear projection layer.

To enable multimodal fusion, the projected text embeddings are spatially broadcast and concatenated with the image feature maps along the channel dimension. 
A lightweight 1×1×1 convolution is then applied to fuse the combined features, producing a unified representation that conditions image-based segmentation on global textual context. 
The fused features are finally passed through a segmentation head to predict tumor sub-regions, including whole tumor (WT), tumor core (TC), and enhancing tumor (ET).

This simple late-fusion strategy serves as a strong and interpretable baseline for text-guided brain tumor segmentation on the BraTS dataset, providing a foundation for more advanced multimodal interaction mechanisms.
