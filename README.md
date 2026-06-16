# Predictive Analytics on Tabular Data

This project applies supervised machine learning methods to a real-world tabular dataset to solve both a classification and a regression task. The workflow covers data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualisation, using Python and scikit-learn.  

## Project structure

- `data/` – Information about the dataset and how to obtain it.
- `notebooks/` – Jupyter notebooks for EDA, preprocessing, modelling, and visualisation.
- `src/` – Reusable Python modules for data preparation, model training, and plotting.
- `reports/figures/` – Exported figures such as confusion matrix, feature importance, and regression plots.
- `requirements.txt` – Python dependencies.

## Methods

- **Classification:** Random Forest classifier with accuracy, precision, recall, F1-score, and confusion matrix.
- **Regression:** K-Nearest Neighbors (KNN) regressor with MSE, RMSE, MAE, and R² metrics.
- **Visualisations:** Confusion matrix heatmap, feature importance bar plot, predicted vs actual scatter plot, residuals histogram.

## How to run

```bash
pip install -r requirements.txt

# Option 1: work in notebooks
jupyter lab  # or jupyter notebook

# Option 2: run scripts (example)
python src/train_classification.py
python src/train_regression.py
```

## Results (brief)

- Random Forest classifier reaches ~96% test accuracy with balanced precision and recall across classes.
- KNN regressor achieves R² ≈ 0.90 with low RMSE, indicating good fit to the continuous target.
