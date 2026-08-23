# SIT307 8.1D ~ House Price Prediction (Random Forest Tuning)

> **Note:** The submission portal did not accept the `.zip` file for this task. As a result, all code (data collection, notebook, and Streamlit app) is included directly in this README for submission purposes.

This repository contains the work for SIT307 Task 8.1D, covering model training, hyperparameter tuning, and a Streamlit app for interactive predictions.

## Repository Structure

- **`data_collected/`** - Data used for model prediction and experimentation. Not used to run the application; used to tune the model and test different hyperparameter configurations.
- **`notebook/`** - Model development code, including evaluation and visualizations.
- **`streamlit_codebase/`** - Tuned model integrated into a Streamlit application for interactive predictions.

## Tuning Summary

Random Forest tuning improved MAE from $173,540 to $165,260 (4.77% improvement) using `n_estimators=800, max_features=0.5, min_samples_leaf=1, min_samples_split=2, max_depth=None`.

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/Architbh007/SIT307.git
```

Navigate to the 8.1D directory:

```bash
cd SIT307/8.1D/streamlit_codebase
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

## Code

### data_collected

```python
https://github.com/Architbh007/SIT307/tree/main/8.1D/data_collected
```

### notebook

```python
https://github.com/Architbh007/SIT307/blob/main/8.1D/notebook/main.ipynb
```

### streamlit_codebase (app.py)

```python
https://github.com/Architbh007/SIT307/tree/main/8.1D/streamlit_codebase
```
