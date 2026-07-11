# Early Detection of Halo CMEs using Aditya-L1

A reproducible forecasting pipeline for detecting halo coronal mass ejections (CMEs) using Aditya-L1 mission data.

## Project Overview

This repository contains the data processing and machine learning workflow developed to support early detection of solar halo CMEs. It includes:

- feature extraction and data preprocessing for Aditya-L1 and SWIS observations
- random forest model training and threshold tuning
- evaluation results and prediction analysis

## Repository Structure

- `Project-Code/01_swis_cme_preprocessing.ipynb` — preprocessing notebook for SWIS and CME label preparation
- `Project-Code/ML_Model/rf_model.py` — model training and inference code for random forest
- `Project-Code/ML_Model/rf_tune.py` — model tuning and threshold optimization
- `Project-Code/ML_Model/RESULTS/` — recorded experiment outputs and evaluation summaries
- `Documents/` — supporting documentation, reports, and presentations

## Highlights

- Machine learning pipeline designed for space weather forecasting
- Clear separation of preprocessing, training, and evaluation
- Focus on reproducibility and modular model development

## Notes

Large binary artifacts such as trained model weights and processed datasets are excluded from version control via `.gitignore`.

## How to use

1. Review the preprocessing notebook to understand the input data and feature pipeline.
2. Run `rf_model.py` to train or evaluate the model.
3. Adjust hyperparameters in `rf_tune.py` to improve performance.

## GitHub Push

To publish this repository, connect it to your GitHub remote and push the `main` branch:

```bash
git remote add origin https://github.com/IsheethaSukumar/Early-detection-of-Halo-CME-using-Aditya-L1.git
git branch -M main
git push -u origin main
```
