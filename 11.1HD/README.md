
# Reproduction and Methodological Extension of a Stacking-Based Ensemble for Heart Disease Prediction

This repository contains the implementation for my **SIT307 Machine Learning** research assessment at Deakin University.

The project reproduces the stacking-based heart-disease prediction approach proposed by Bhagat, Sharma and Agarwal, and then extends the evaluation with a stricter duplicate-aware experimental protocol.

## Project Overview

The work is divided into two main stages.

### Part 1 - Reproduction

Part 1 reproduces the published modelling approach using six base classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Gaussian Naive Bayes
- K-Nearest Neighbours

These models are combined using a **five-fold stacking ensemble**. Out-of-fold predictions from the six base classifiers are used as the input features for the stacking meta-classifier.

The reproduced stacking model achieved:

- **Accuracy:** 98.54%
- **F1-score:** 0.9852
- **Sensitivity:** 0.9709
- **MCC:** 0.9712
- **ROC-AUC:** 1.0000

The published stacking accuracy was **98.53%**, so the reproduced headline result differs by only **0.01 percentage points**.

## Part 2 - Methodological Extension

A subsequent audit of the supplied dataset found substantial repetition:

- Original observations: **1,025**
- Additional repeated observations: **723**
- Distinct predictor profiles: **302**
- Repeated proportion: **70.54%**
- Conflicting targets among identical predictor profiles: **0**

Because the selected paper does not report a duplicate-handling procedure, Part 2 evaluates the same modelling problem under a stricter profile-based protocol.

The refined evaluation uses:

- One observation per distinct predictor profile
- Stratified 80:20 train/test split
- **241 training profiles**
- **61 held-out test profiles**
- Zero exact predictor-profile overlap between train and test
- Fold-contained preprocessing
- Training-only model selection
- Five-fold GridSearchCV for base-model optimisation
- Explicit meta-classifier comparison
- Re-optimisation of the final stacking meta-classifier
- Bootstrap uncertainty analysis

## Meta-Classifier Selection

Three candidate meta-classifier families were compared using training-only cross-validation:

| Meta-classifier | Mean CV AUC |
|---|---:|
| Logistic Regression | **0.9126** |
| MLP | 0.8833 |
| Decision Tree | 0.8695 |

Logistic Regression was therefore selected as the stacking meta-classifier family.

## Hyperparameter Optimisation

All six base classifiers were tuned independently using **five-fold stratified GridSearchCV**, with ROC-AUC used as the primary refit metric.

The strongest tuned individual model by held-out accuracy was Random Forest:

- **Accuracy:** 0.8197
- **F1-score:** 0.8406
- **MCC:** 0.6371
- **AUC:** 0.8755

## Final Optimised Stacking Ensemble

After tuning the six base models, the Logistic Regression meta-classifier was re-optimised because the probability features produced by the tuned base learners had changed.

The final stacking ensemble achieved:

| Metric | Result |
|---|---:|
| Accuracy | **0.8033** |
| F1-score | **0.8421** |
| Recall / Sensitivity | **0.9697** |
| Precision | 0.7442 |
| Specificity | 0.6071 |
| MCC | **0.6303** |
| ROC-AUC | **0.8896** |

The final confusion matrix contained:

- True negatives: **17**
- False positives: **11**
- False negatives: **1**
- True positives: **32**

The model therefore detected **32 of 33 positive profiles**, while accepting a lower specificity.

## Key Finding

The main finding is not that optimisation restores the original near-perfect result.

Instead, the project shows that the published headline performance can be reproduced closely under the reproduction-oriented protocol, but that the near-perfect behaviour does **not persist under the refined unique-profile evaluation protocol**.

The methodological contribution is therefore the introduction of a more controlled and transparent evaluation pipeline based on:

- duplicate-aware data auditing,
- zero exact train/test profile overlap,
- leakage-controlled preprocessing,
- explicit meta-model selection,
- training-only hyperparameter optimisation,
- full stacking re-optimisation, and
- uncertainty-aware reporting.

The Part 2 results should not be interpreted as proving that duplicate removal alone caused the performance reduction, because the refined protocol also changes the partitioning strategy and introduces stratification.

## Running the Notebook

### 1. Clone the repository

```bash
git clone https://github.com/Architbh007/SIT307.git
cd SIT307/11.1HD
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install the required Python packages

The notebook uses the standard scientific Python and machine-learning stack. Install the required packages with:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

### 4. Start Jupyter

```bash
jupyter notebook
```

Open:

```text
main.ipynb
```

Then run the notebook from top to bottom.

## Reproducibility Notes

The notebook uses fixed random states where appropriate so that the experimental protocol can be reproduced consistently.

Part 1 follows the published workflow as closely as possible, while explicitly documenting assumptions where the paper does not provide a complete executable specification.

Part 2 keeps the final held-out test set isolated from model selection. Preprocessing, cross-validation, meta-classifier selection and hyperparameter optimisation are performed using the training subset only.

The bootstrap confidence intervals are calculated from fixed held-out predictions and therefore represent uncertainty conditional on the fitted model and observed test sample. They do not capture the additional variation that would arise from retraining the full pipeline on different train/test partitions.

## Selected Research Paper

M. Bhagat, A. Sharma and P. Agarwal,  
**“An efficient stacking-based ensemble technique for early heart attack prediction,”**  
*Multimedia Tools and Applications*, vol. 84, pp. 36351–36375, 2025.

DOI: `10.1007/s11042-024-19293-7`

## Video Demonstration

A concise video walkthrough of the reproduction, methodological extension and key results is available here:

https://youtu.be/EoPNhlELgj4

## Notebook

The main implementation notebook is available here:

https://github.com/Architbh007/SIT307/blob/main/11.1HD/main.ipynb

## Academic Context

This repository was created for the **SIT307 — Machine Learning** unit at Deakin University.

The work is a research reproduction and methodological evaluation exercise. It should not be interpreted as a clinically validated diagnostic system.
