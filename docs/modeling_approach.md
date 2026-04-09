# Modeling Approach

## Dataset

This project uses the **State Farm Distracted Driver Detection** dataset from Kaggle. The dataset contains around 22,000 labeled driver images and is organized into 10 distracted driving behavior categories.

The class labels are:

- `c0`: normal driving
- `c1`: texting, right hand
- `c2`: texting, left hand
- `c3`: talking on the phone, right hand
- `c4`: talking on the phone, left hand
- `c5`: operating the radio
- `c6`: drinking
- `c7`: reaching behind
- `c8`: hair and makeup
- `c9`: talking to passenger

The dataset provides a suitable benchmark for distracted driving classification because it contains real in-car driver images with a relatively balanced class distribution.

## Data Preparation

The modeling pipeline uses image transforms appropriate for transfer learning with VGG16.

Typical preprocessing steps include:

- resizing images
- cropping to match the expected input size
- normalization using ImageNet statistics
- loading images through a folder-based dataset structure

The training data is loaded using `ImageFolder`, while a custom dataset class can be used for unlabeled test images.

## Model Selection

The core classifier is a **pretrained VGG16** model.

VGG16 was selected because:

- it is a widely used and well-understood CNN architecture
- prior distracted driving work has shown strong performance with VGG16
- its convolutional structure works well with Grad-CAM for interpretability

The model is adapted by replacing the final classification layer so that it outputs 10 classes instead of ImageNet classes.

## Training Setup

The training setup follows a standard fine-tuning workflow:

- pretrained VGG16 backbone
- final linear layer replaced for 10-class output
- loss function: cross-entropy loss
- optimizer: SGD
- learning rate: 0.001
- momentum: 0.9
- batch size: 32
- training epochs: 5

This configuration is designed to balance simplicity, reproducibility, and interpretability.

## Evaluation

The model is evaluated on a validation split using:

- overall accuracy
- per-class precision / recall / F1
- confusion matrix

This evaluation is important because distracted driving classes are visually similar in some cases. For example, normal driving and talking to a passenger may be confused more often than visually distinct actions. The confusion matrix helps reveal where those class-level weaknesses appear.

