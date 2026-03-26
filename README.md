# Automatic Detection of Parkinson's Disease from vGRF Signals

**Course:** Project Time Series — MSc Artificial Intelligence, FAU Erlangen-Nürnberg  
**Supervisor:** Dr.-Ing. Tomás Arias-Vergara  

---

## Overview

This project implements a fully automated pipeline for Parkinson's disease (PD) 
detection using vertical ground reaction force (vGRF) signals from the 
[PhysioNet Gait in Parkinson's Disease database](https://physionet.org/content/gaitpdb/1.0.0/).

The pipeline covers data preprocessing, TSFresh feature extraction, PCA 
visualisation, and nested cross-validated classification using classical 
machine learning models.

---

## Dataset

- 166 subjects: 93 PD, 73 Healthy Controls (HC)
- 16 vGRF channels (8 sensors per foot) at 100 Hz
- Usual-walk trials only; dual-task trials excluded

---

## Pipeline

1. **Preprocessing** — 8-sensor summation per foot, 6 Hz zero-phase 
   Butterworth filter, body-weight normalisation
2. **Feature Extraction** — 300 ms sliding window (150 ms step), 
   TSFresh `ComprehensiveFCParameters`, aggregated per subject using 
   mean, std, skewness, kurtosis
3. **Feature Configurations** — Left foot, Right foot, Combined (L + R)
4. **Classification** — Random Forest, SVM-Linear, SVM-RBF under nested 
   stratified 5-fold cross-validation with inner 3-fold GridSearchCV
5. **Evaluation** — Accuracy, Sensitivity, Specificity, AUC

---

## Results

| Features | Model    | Acc.         | Sens.        | Spec.        | AUC          |
|----------|----------|--------------|--------------|--------------|--------------|
| Left     | RF       | 0.64 ± 0.07  | 0.75 ± 0.11  | 0.49 ± 0.11  | 0.68 ± 0.05  |
| Left     | SVM-Lin  | 0.64 ± 0.08  | 0.83 ± 0.07  | 0.40 ± 0.09  | 0.62 ± 0.11  |
| Left     | SVM-RBF  | 0.67 ± 0.07  | 0.89 ± 0.11  | 0.45 ± 0.24  | 0.65 ± 0.05  |
| Right    | RF       | 0.69 ± 0.04  | 0.83 ± 0.07  | 0.52 ± 0.13  | 0.76 ± 0.08  |
| Right    | SVM-Lin  | 0.67 ± 0.05  | 0.78 ± 0.10  | 0.54 ± 0.11  | 0.75 ± 0.07  |
| Right    | SVM-RBF  | 0.72 ± 0.05  | 0.76 ± 0.05  | 0.65 ± 0.13  | 0.78 ± 0.05  |
| Combined | RF       | 0.73 ± 0.03  | 0.82 ± 0.08  | 0.62 ± 0.06  | 0.85 ± 0.07  |
| Combined | SVM-Lin  | 0.75 ± 0.04  | 0.75 ± 0.07  | 0.75 ± 0.11  | 0.82 ± 0.06  |
| **Combined** | **SVM-RBF** | **0.78 ± 0.05** | **0.78 ± 0.09** | **0.72 ± 0.11** | **0.85 ± 0.04** |

Best result: Combined + SVM-RBF, AUC **0.85 ± 0.04**

---

## Repository Structure
```
├── data/                        # see PhysioNet
    ├── raw/
    ├── processed/
├── reports/
│   ├── figs/                    # All generated figures
    ├── final_report/
    ├── ppt/
│   ├── baseline_result_summary.csv
│   └── baseline_folds.csv
├── src/
    ├── TSFresh_PCA.py                 # Feature extraction and PCA visualisation
    ├── baseline_models.py             # Classification pipeline
    ├── data_analysis_ppt_plots.ipynb  # Signal plots and data analysis
    ├── diagrams.py                    # Plots for report
└── README.md
```

---

## Requirements

Dependencies are managed via conda. Python 3.10 is required.

Key packages: `tsfresh`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `pyarrow`

---

## Installation
```bash
# Clone the repository
git clone https://github.com/saranyab21/Project-Time-Series-WS-25-26.git
cd Project-Time-Series-WS-25-26

# Create and activate the conda environment
conda env create -f environment.yml
conda activate gait_mamba
```

## Usage
```bash
# Step 1: Extract features and generate PCA plots
python src/TSFresh_PCA.py

# Step 2: Run baseline classifiers
python src/baseline_models.py
```

---

## Key Findings

- Bilateral fusion (Combined) consistently outperforms single-foot configurations
- Right foot features are stronger than left, consistent with 
  gait asymmetry commonly reported in PD.
- SVM-Linear underperforms across all configurations, confirming a 
  nonlinear decision boundary in the TSFresh feature space
- Nested CV was essential given the small sample size (n=166) to avoid 
  optimistically biased estimates

---

## References

- Goldberger et al., PhysioNet, 2000
- Christ et al., TSFresh, 2018
- Trabassi et al., Sensors, 2022
- Tsakanikas et al., Sensors, 2023
- Navita et al., Scientific Reports, 2025

## Author

**Saranya Bhattacharjee (aj81owid)**
