# ADS-B Anomaly Detection with Explainable AI

This repository contains, data preprocessing, trained model, and explanation tools used in our Bachelor's thesis project on detecting anomalies in ADS-B (Automatic Dependent Surveillance–Broadcast) flight data using a Recurrent Autoencoder and XAI (SHAP and LIME).

## Overview

This project proposes a method for identifying injected or manipulated ADS-B sequences using a LSTM Autoencoder. The model is trained to reconstruct normal flight sequences and flag sequences with high reconstruction error as anomalies. 

## Model Architecture

- Model Type: Recurrent Autoencoder (LSTM-based)
- Input Format: Sliding windows of multivariate ADS-B sequences
- Loss Function: L1 reconstruction loss
- Classifier: Threshold-based anomaly classifier using reconstruction error

## Explainability Tools

- SHAP: Visualizes feature importance using kernel SHAP approximation. Supports batch processing for large datasets.
- LIME: Provides instance-specific explanations for anomalous samples.

## How to Run

## Results Summary

## Dataset Description

The dataset is constructed from:
- Real ADS-B sequences
- Simulated attacks

Each sample is a sequence of 8 timesteps with the following features:
- `long`, `lat`, `true_track`,`baro_altitude`, `velocity`, `vertical_rate`, `geo_altitude`, `squawk`
