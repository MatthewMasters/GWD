# Global Wheat Detection
Authored by Matthew Masters

Deep learning pipeline for wheat head detection from outdoor imagery. This code was developed for the [Global Wheat Head Detection competition](https://www.kaggle.com/c/global-wheat-detection) hosted by Kaggle.

Features:
 - Two deep learning detectors: FasterRCNN and EfficientDet.
 - Custom augmentations built into albumentations
 - Ensembling via weighted box fusion
 - Test time augmentation
 - Pseudolabeling techniques
