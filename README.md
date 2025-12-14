# Smartphone Human Activity Recognition (HAR)

This repository contains all materials for a course project on human activity recognition using smartphone motion sensors. The goal is to classify short windows of sensor data into four activities:

- Walking
- Running
- Climbing stairs
- Sitting

We compare two models:

- A **Random Forest** trained on hand‑crafted statistical features
- A **1D Convolutional Neural Network (1D‑CNN)** trained on raw sensor windows

---

## Repository layout

The repo is organized into three main folders:

- `raw_data/`  
  Original recordings from the smartphones. Each file contains time‑stamped motion sensor readings (accelerometer, AccelerometerUncalibrated, Annotation, gravity, gyroscope, GyroscopeUncalibrated, Metadata, orientation, TotalAcceleration) collected at 100 Hz from multiple devices and participants.
  Please note that the Annotation file is empty as it wasn't configured during recording data.

- `preprocessed_data/`  
  Cleaned and curated data used for modeling. This includes:
  - Synchronized and resampled sensor streams
  - 2.56‑second windows (256 samples) with labels for walking, running, stairs, and sitting
  - Metadata indicating participant, device, operating system, and activity

- `code/`  
  All notebooks and scripts for:
  - Loading and preprocessing the raw recordings
  - Creating the windowed dataset and metadata
  - Training and evaluating the Random Forest and 1D‑CNN models
  - Generating confusion matrices, learning curves, and summary tables

---
## Interactive notebook

All steps of the pipeline and the main model outputs can also be viewed in an interactive Colab notebook:

[Colab notebook – Smartphone HAR project](https://colab.research.google.com/drive/1YTbFhyy_04UZm4T7vbTwPBsKj3gnADoM?usp=sharing)

The notebook walks through:
- Loading the preprocessed data
- Training the Random Forest and 1D‑CNN
- Reporting accuracy and macro F1 on the person‑wise, cross‑device test set
- Visualizing confusion matrices and learning curves

---
## What the project demonstrates

- How to go from **raw multi‑sensor smartphone data** to a labeled windowed dataset.
- How a **feature‑based Random Forest** performs compared to a **1D‑CNN on raw time series**.
- How using a **person‑wise, cross‑device split** (train on some users/devices, test on a new user/device) gives a realistic measure of generalization.

This repository is meant as a complete, end‑to‑end example for an applied machine learning project on smartphone HAR.
