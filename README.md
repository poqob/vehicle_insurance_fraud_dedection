# Vehicle Insurance Fraud Detection

This project aims to detect fraudulent claims in vehicle insurance datasets using a variety of machine learning and deep learning techniques. The workflow includes data exploration, preprocessing, handling imbalanced data, and building several classification models to identify fraud effectively.

## Features
- **Data Exploration & Visualization:** Initial analysis and visualization of the dataset to understand distributions and correlations.
- **Preprocessing:** Handling missing values, encoding categorical variables (one-hot and label encoding), and feature selection.
- **Imbalanced Data Handling:** Techniques such as undersampling, oversampling (RandomOverSampler, SMOTE, SMOTENC) to address class imbalance.
- **Modeling:**
  - XGBoost Classifier
  - Logistic Regression
  - Multi-Layer Perceptron (MLPClassifier)
  - PyTorch-based neural network (see `torch-example/`)
- **Evaluation:** Model performance is assessed using confusion matrices, classification reports, and cross-validation.

## Dataset
- The main dataset is located at `data/carclaims.csv`.

## Notebooks & Scripts
- Main workflow and experiments are in `main.ipynb`.
- Additional PyTorch models and analysis are in the `torch-example/` directory.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the notebook `main.ipynb` to follow the full workflow.
2. Explore the `torch-example/` directory for deep learning models implemented with PyTorch.

## Project Structure
```
vehicle_insurance_fraud_dedection/
├── main.ipynb
├── requirements.txt
├── data/
│   └── carclaims.csv
├── torch-example/
│   ├── model.py
│   ├── predict.py
│   └── ...
└── ...
```

## Topics
vehicle-insurance, fraud-detection, machine-learning, deep-learning, imbalanced-data, xgboost, logistic-regression, mlp, pytorch, smote, data-analysis, classification, insurance-fraud, python, scikit-learn

## License
This project is for educational and research purposes.
