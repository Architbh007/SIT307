
# SIT307 8.1D ~ House Price Prediction (Random Forest Tuning)

This repository contains the work for SIT307 Task 8.1D, covering model training, hyperparameter tuning, evaluation, and a Streamlit app for interactive predictions.

## Repository Structure

- **`data_collected/`** - Data used for model prediction and experimentation. This is *not* used to run the application; it was used solely to tune the model and test different hyperparameter configurations.
- **`notebook/`** - Contains the full model development code, including detailed evaluation, visualizations, and step-by-step explanations of the approach.
- **`streamlit_codebase/`** - Contains the tuned model integrated into a Streamlit application for interactive predictions.

## Model Results

**Best Random Forest Parameters:**

| Parameter | Value |
|---|---|
| `max_depth` | None |
| `max_features` | 0.5 |
| `min_samples_leaf` | 1 |
| `min_samples_split` | 2 |
| `n_estimators` | 800 |

**Performance:**

| Metric | Value |
|---|---|
| Baseline MAE | $173,540 |
| Tuned MAE | $165,260 |
| Improvement | 4.77% |

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/Architbh007/SIT307.git
```

Navigate to the 8.1D directory:

```bash
cd SIT307/8.1D
```

## Application Build and Run Instructions

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```