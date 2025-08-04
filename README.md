# Glacier-Mapping Using Deep Learning

This project focuses on **semantic segmentation of glaciers in the Indian Himalayas** using deep learning models. It is part of an internship at the Indian Institute of Technology Roorkee.

## Project Objective

To **accurately map glacier regions** using multi-band satellite imagery and reduce manual labor in monitoring glacier retreat. The aim is to enable **long-term glacial change analysis** using automated tools.

## Repository Structure

- `models/`: Contains implementations of **ResUNet** (CNN) and **Segformer B4** (Transformer).
- `losses/`: Combinations of loss functions like Dice, IoU, Focal, BCE, and Tversky.
- `train.py`, `evaluate.py`: Model training and evaluation scripts.
- `dataloader.py`: Custom data loader for handling raster and mask inputs.
- `patches.py`: Generates 128×128 pixel patches from large raster files.
- `main.py`: End-to-end pipeline runner.
- `postprocessing_visualization/`: Visual overlays of predictions vs. ground truth.
- `weights/`: Pretrained model weights for different loss combinations.
- `requirements.txt`: Python dependencies.
- `README.md`: You're here :)

## Dataset

Provided by **Sarvesh Verma**, the dataset includes 18-band satellite raster `.tif` files and glacier masks from 5 regions:
- Himachal
- Himachal–Ladakh
- Kashmir
- Sikkim
- Uttarakhand

Bands include **NIR, Slope, Thermal, SWIR, Elevation**, etc.

## Models Used

- **ResUNet**: A hybrid of ResNet and UNet with residual connections to avoid vanishing gradients.
- **Segformer B4**: A state-of-the-art Transformer-based model with MiT (Mix Vision Transformer) encoder and lightweight MLP decoder.

## Results

- **ResUNet (Focal + Dice + Tversky Loss)**: Achieved **Mean IoU ≈ 82%**
- **Segformer B4 (Dice + BCE Loss)**: Achieved **Mean IoU ≈ 79%**

## Visual Results

Yellow = Ground Truth | Red = Prediction

**ResUNet (Dice + BCE Loss) – Uttarakhand Region**

<p float="left">
  <img src="https://github.com/MukeshK17/Glacier-Mapping/blob/main/assets/Results%20ResUNet/BCE%20%2B%20DICE%20loss/Uttrakhand.png" width="45%">
  <img src="https://github.com/MukeshK17/Glacier-Mapping/blob/main/assets/Results%20ResUNet/BCE%20%2B%20DICE%20loss/Uttrakhand_postprocessed.png" width="45%">
</p>

### Overall Comparison

![Results](https://github.com/MukeshK17/Glacier-Mapping/blob/main/assets/Results.jpg)

## Acknowledgements

Grateful to:
- **Prof. Sparsh Mittal** (IIT Roorkee)
- **Prof. Saurabh Vijay** (IIT Roorkee)
- **Sarvesh Verma** (PhD Scholar, IIT Roorkee)
- **Sarthak Patil** (Student, IIT BHU)

For guidance, mentorship, and data support during the internship.

