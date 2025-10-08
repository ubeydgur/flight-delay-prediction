# Flight Delay Prediction

A machine learning project that predicts the likelihood of flight delays based on flight, weather, and scheduling data using various classification algorithms.  
This project aims to classify whether a flight will be delayed or on-time using structured aviation data.  
Several models such as **Logistic Regression** and **Decision Tree Classifier** are applied and evaluated.  
The goal is to find the most accurate and reliable model through evaluation metrics and cross-validation.

## Features

- **Data Analysis & Visualization**  
  Explore and visualize flight, time, and delay-related features using pandas, NumPy, matplotlib, and seaborn.

- **Data Preprocessing & Handling Imbalance**  
  Perform missing value imputation with **SimpleImputer** and handle class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).

- **Model Training & Evaluation**  
  Train models such as **Logistic Regression** and **Decision Tree** using scikit-learn and evaluate them using metrics like precision, recall, ROC-AUC, and cross-validation scores.

- **Feature Encoding**  
  Apply **OneHotEncoder** to transform categorical features into numerical format for model compatibility.

- **Model Optimization**  
  Use **StratifiedKFold** and **cross-validation** techniques to ensure balanced and robust evaluation.

- **Visualization of Model Performance**  
  Visualize ROC curves and performance metrics using **Yellowbrick’s ROCAUC** visualizer.

## Dependencies
- os  
- numpy  
- pandas  
- seaborn  
- matplotlib  
- scikit-learn  

## The script will

Load and preprocess the flight delay dataset  

Handle missing values and perform feature encoding  

Train and evaluate different models  

Visualize ROC-AUC scores and classification metrics for performance comparison  
