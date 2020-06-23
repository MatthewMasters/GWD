# Global Wheat Detection
Deep learning system for wheat head detection. Below you can find a complete outline of how to reproduce my solution for the Global Wheat Detection challenge hosted by Kaggle. If you run into any trouble with running the code or have any questions, feel free to email me at matthewrmasters@gmail.com.



### Archive Contents

- **data/train_corr** - dataset used in training (restored full images and corrected labels)
- **data/models** - model weights with best LB submission
- **pipeline/preparation** - code used in the preparation of the data
- **pipeline/training** - code to reproduce the models from scratch
- **pipeline/inference** - code to run prediction on test set (including test-time augmentation, pseudolabeling, and ensembling)
- **pipeline/wheatdet** - code required by multiple components of the pipeline such as network architectures and dataloaders



### Hardware

- 512 GB SSD
- 16 GB RAM

- Intel Core i5-3570K

- 1x NVIDIA GeForce RTX 2070 GPU



### Software

- Ubuntu 18.04.1 LTS

- Python 3.6.7

- CUDA 10.2

- NVIDIA Driver 430.14




## Instructions

### Data Preparation

Data was downloaded from the Kaggle competition website. Several steps were taken to prepare and clean the dataset before training. First, images were (mostly) reconstructed back into the original full size images. This was done in order to improve sampling during training through random cropping. This step was performed by running the `preparation/stitch_dataset.py` script and then manually removing the incorrect images. An initial deep learning model was trained using this updated dataset. The predictions were plotted against the ground truth and errors were highlighted. A manual review of every image was conducted and many corrections were made by hand using an open-source image annotation tool called LabelImg. The tool can be downloaded via [this github repository](https://github.com/tzutalin/labelImg). Many labels were removed, modified, and added to the collection of images to more accurately reflect the ground truth.



### Model Training

Text



### Model Prediction

Text



### Things to Note

Text



## License

In accordance with the Global Wheat Detection challenge rules, my solution is open sourced under the [MIT license](LICENSE).



## Citation

Citation to biorxiv