# ADS-B Anomaly Detection with Explainable AI

This repository contains data preprocessing, a trained model, and explanation tools used in our Bachelor's thesis project on detecting anomalies in Automatic Dependent Surveillance–Broadcast (ADS-B) flight data using a Recurrent Autoencoder and XAI (SHAP and LIME).

## Overview

This project proposes a method for identifying injected or manipulated ADS-B sequences using a LSTM-based Autoencoder. The model is trained to reconstruct normal flight behavior, and sequences with high reconstruction error are flagged as anomalous. The pipeline combines time series modeling with interpretable machine learning to support trust and transparency.

## Model Architecture

- **Model Type**: Recurrent Autoencoder (LSTM-based)
- **Input Format**: Sliding windows of multivariate ADS-B sequences (8 timesteps)
- **Loss Function**: L1 reconstruction loss
- **Classifier**: Threshold-based anomaly detector based on total reconstruction error

## Explainability Tools

To understand and interpret the model’s decisions, the project includes two complementary explainability methods:

### SHapley Additive exPlanations (SHAP)

- Uses `shap.KernelExplainer` to estimate feature contributions to anomaly classification.
- Explains model output based on reconstruction error–driven anomaly scores.
- Supports batch explanations on large datasets (e.g. 500+ samples) using efficient tensor processing.
- Especially suited to highlight the most influential timesteps or features in anomalous flight patterns.

### Local Interpretable Model-agnostic Explanations (LIME)

- Used to generate local, interpretable surrogate models (linear) around individual samples.
- Helps explain why a specific sequence was flagged as anomalous by perturbing the input and observing change in the anomaly score.
- Provides instance-specific insight, useful for case studies and visual inspection of particular flight paths.

These tools are integrated into the post-analysis pipeline and are essential for supporting model transparency, debugging, and hypothesis generation around ADS-B attacks.

## How to Run

_To be completed._

## Results Summary

_To be completed._

## Dataset Description

The dataset is constructed from a combination of real ADS-B flight sequences and synthetically generated attack data. It includes both normal and manipulated trajectories to evaluate detection capability.

Each sample is a time window of 8 consecutive timesteps, with the following features:

- `long`, `lat`, `true_track`
- `baro_altitude`, `velocity`, `vertical_rate`
- `geo_altitude`, `squawk`

The data is normalized prior to training and labeled into one of the following classes:
- `0` = Normal
- `1` = Path Modification
- `2` = Ghost Aircraft
- `3` = Velocity Drift

