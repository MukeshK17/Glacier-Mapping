# Glacier Mapping Using Deep Learning

This project focuses on **semantic segmentation of glaciers in the Indian Himalayas** using deep learning models. It was developed as part of an internship at the Indian Institute of Technology Roorkee.

---

## Project Objective

To **automatically map glacier regions** from multi-band satellite imagery and reduce manual effort in monitoring glacier retreat. The goal is to support **long-term glacial change analysis** using scalable ML pipelines.

---

## Repository Structure

src/glacier_mapping/
│
├── data/ # Dataloader and dataset logic
├── models/ # ResUNet and SegFormer B4
├── losses/ # Custom loss functions
├── training/ # train.py and evaluate.py
├── utils/ # Band analysis & visualization tools
│
outputs/ # predictions, weights, visualizations
notebooks/ # experiments
scripts/ # entry scripts

---

## Dataset

Provided by **Sarvesh Verma**, the dataset contains 18-band satellite `.tif` rasters and glacier masks from:

- Himachal  
- Himachal–Ladakh  
- Kashmir  
- Sikkim  
- Uttarakhand  

Bands include:
**NIR, SWIR, Thermal Infrared, Elevation (DEM), Slope, Velocity, Backscatter**, etc.

---

## Models Used

### ResUNet
- UNet-based architecture with residual connections  
- Helps improve gradient flow and stability  

### SegFormer B4
- Transformer-based segmentation model  
- MiT encoder + lightweight decoder  

---

## Results

| Model | Loss | Mean IoU |
|------|------|---------|
| ResUNet | Focal + Dice + Tversky | ~82% |
| SegFormer B4 | Dice + BCE | ~79% |

---

## Visual Results

**Yellow = Ground Truth | Red = Prediction**

### ResUNet (Dice + BCE Loss) – Uttarakhand


<p float="left">
  <img src="https://github.com/MukeshK17/Glacier-Mapping/blob/main/assets/Results%20ResUNet/BCE%20%2B%20DICE%20loss/Uttrakhand.png" width="45%">
  <img src="https://github.com/MukeshK17/Glacier-Mapping/blob/main/assets/Results%20ResUNet/BCE%20%2B%20DICE%20loss/Uttrakhand_postprocessed.png" width="45%">
</p>

### Overall Comparison

![Results](https://github.com/MukeshK17/Glacier-Mapping/blob/main/assets/Results.jpg)

---

## Current Work

- Improving performance using **ensemble learning**
- Combining:
  - ResUNet  
  - SegFormer B4  
- Using a **Tiny UNet meta-model**

---

## Acknowledgements

- **Prof. Sparsh Mittal** (IIT Roorkee)  
- **Prof. Saurabh Vijay** (IIT Roorkee)  
- **Sarvesh Verma** (PhD Scholar, IIT Roorkee)  
- **Sarthak Patil** (IIT BHU)  

For guidance, mentorship, and dataset support.