# BCDML (Breast Cancer Diagnosis with Machine Learning)

### Overview

This project aims to build a machine-learning model that accurately classifies breast tumors as benign or malignant based on cell nucleus features. The dataset used for this analysis is the Breast Cancer Wisconsin (Diagnostic) dataset1. We will explore the data, apply various classification techniques, and recommend the best model.

### Dataset Description

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Features:** 30 numerical features describing cell nucleus properties

**Target Variable:** Binary classification (Benign or Malignant)


### Introduction
In this report, I investigate the efficacy of various machine learning models for classifying breast cancer tumors as benign or malignant. The analysis utilizes a dataset obtained from the [UCI Machine Learning Repository
](http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). My goal was to identify the most suitable model for accurate tumor classification while also exploring potential hidden information within the data.

### Methodology
I employed a range of classification algorithms, including:

Logistic Regression
Support Vector Machine (SVM)
Decision Tree
Naive Bayes
K-Nearest Neighbors (KNN)
Random Forest
Gradient Boosting
AdaBoost
SGD Classifier
For each model, I implemented a grid search to optimize hyperparameters and subsequently evaluated their performance on a test set. The evaluation metrics included accuracy, precision, recall, F1-score, mean squared error (MSE), and R^2 score.

### Results
The analysis revealed Logistic Regression as the leading performer, achieving exceptional results:

Accuracy: 98.25%
Precision: 1.00 (perfect precision for identifying malignant tumors)
Recall: 0.9535 (correctly identifies a high proportion of malignant tumors)
F1-Score: 0.9762 (well-balanced between precision and recall)
Other models, such as AdaBoost and SGD Classifier, also achieved high accuracy (around 97%).

### Hidden Information Exploration
While the provided code snippet lacked explicit techniques for uncovering hidden information, I utilized Principal Component Analysis (PCA) to identify underlying patterns in the data. This dimensionality reduction technique revealed two principal components (PCs) explaining a significant portion of the data's variance. Further analysis of these PCs with respect to the original features could potentially reveal:

Correlations: Examining how features load onto the PCs can identify groups of features that tend to vary together, suggesting potential redundancies or underlying biological processes at play.
Cluster analysis: Projecting the data onto the PCs and performing clustering techniques might reveal distinct subgroups within the benign and malignant categories, suggesting hidden tumor subtypes.
### Discussion
Logistic Regression's remarkable performance, coupled with its interpretability, makes it a highly compelling choice for this specific task. Here's a breakdown of some key considerations:

Logistic Regression: Offers ease of interpretation, exceptional performance on this dataset, and a strong balance between accuracy and interpretability. This allows for better understanding of the factors influencing tumor classification.
AdaBoost & SGD Classifier: While achieving good accuracy, they are less interpretable compared to Logistic Regression.
### Recommendations
Given Logistic Regression's superior performance, it is the recommended model for this task. However, depending on the specific needs of the project, further analysis might be warranted:

Feature importance analysis: Utilize Logistic Regression to understand which tumor characteristics are most significant for classification.
PCA insights: Explore potential feature redundancies and hidden tumor subtypes using the insights from PCA. This can guide further biological investigation and potentially lead to the discovery of new diagnostic markers.
Dataset limitations & external validation: Investigate the limitations of the dataset and potential biases. External validation on a separate dataset with known clinical outcomes is highly recommended to assess the generalizability of the model's performance in a real-world setting.
### Conclusion
This analysis highlights Logistic Regression as the most effective model for classifying breast cancer tumors in this dataset. Its interpretability alongside exceptional performance makes it a strong candidate for clinical applications. Exploring hidden information within the data using PCA provides valuable insights for further investigation. Addressing dataset limitations and performing external validation are crucial steps before deploying the model in a clinical setting.

