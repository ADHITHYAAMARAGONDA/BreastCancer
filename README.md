Title: Breast Cancer Prediction using KNN, Logistic Regression, and Decision Tree

Description:
This project performs predictive analysis on the Breast Cancer dataset using K-Nearest Neighbors KNN, Logistic Regression, and Decision Tree models. The objective is to classify tumors as Benign or Malignant based on relevant features.

Key Features:

Data Preprocessing:

Handled missing values, performed feature scaling, and encoded categorical variables to improve model accuracy.
Model Training and Evaluation:

Implemented KNN, Logistic Regression, and Decision Tree for classification.
Evaluated models using confusion matrices, accuracy, precision, recall, F1-score, and error rate.
Performance Visualization:

Compared model metrics using bar plots Accuracy, Precision, Recall, F1-score, and Error Rate for better interpretation.
Technologies Used:

R, Tidymodels, Caret, ggplot2, dplyr, tidyr


# Breast Cancer Prediction Using Machine Learning (R)

## Overview
This project focuses on building machine learning models to predict the presence of breast cancer based on medical features. It uses supervised learning algorithms and performance evaluation metrics to classify tumors as malignant or benign.

## Features
- Data preprocessing: handling missing values, feature scaling, encoding
- Trained and evaluated multiple models:
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
- Performance metrics: accuracy, precision, recall, F1-score, ROC curves
- Data visualization for insight and model analysis

## Project Structure
```
BreastCancer/
├── breast_cancer_analysis.R       # Main R script
├── dataset.csv                    # Breast cancer dataset (if included)
├── results/                       # Confusion matrix, plots, ROC curves
└── README.md                      # Project documentation
```

## Technologies Used
- R
- Tidymodels
- Caret
- ggplot2
- dplyr
- tidyr

## Setup and Installation
1. Ensure R and RStudio are installed.
2. Install the required packages:
   ```R
   install.packages(c("tidymodels", "caret", "ggplot2", "dplyr", "tidyr"))
   ```
3. Open the `breast_cancer_analysis.R` script in RStudio.
4. Run the script to execute the full analysis.

## Output / Results
- Model with highest accuracy: **Logistic Regression (90%)**
- Evaluation metrics:
  - Precision: 88%
  - Recall: 85%
  - F1-Score: 86%
- Visualization includes:
  - Confusion matrix
  - ROC curves
  - Feature importance plots

## Learnings
- Hands-on experience with multiple ML algorithms in R
- Applied key data science steps: cleaning, modeling, evaluation
- Learned performance tuning and comparison of classification models
- Gained insights into healthcare-related data

## Future Improvements
- Add cross-validation and hyperparameter tuning
- Integrate Shiny dashboard for model interaction
- Expand dataset and test deep learning models (optional)

## Acknowledgements
- UCI Machine Learning Repository (dataset source)
- Tidyverse and Caret documentation

## License
This project is open-source and available under the MIT License.
