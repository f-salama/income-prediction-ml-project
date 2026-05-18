# Adult Income Prediction using Machine Learning

This project applies the full machine learning lifecycle to the UCI Adult Census Income dataset. The goal is to predict whether an individual earns more than $50K per year based on demographic, educational, occupational, and financial features.

## Project Overview

The project answers the question:

**Does this person earn more than $50K per year?**

The dataset contains information from the 1994 U.S. Census Bureau database, including features such as age, education level, occupation, marital status, hours worked per week, capital gain, and capital loss.

This project includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Train/test splitting
- Model training using three different model categories
- Model evaluation and comparison
- Model optimisation using hyperparameter tuning and feature selection

## Dataset

**Dataset:** Census Income / Adult Dataset  
**Source:** UCI Machine Learning Repository  
**Task Type:** Supervised binary classification  
**Target Variable:** Income  
**Classes:**
- `<=50K`
- `>50K`

Dataset link: https://archive.ics.uci.edu/dataset/20/census+income

## Models Used

Three machine learning models were trained and compared:

### 1. Decision Tree Classifier

Used as the traditional machine learning model. It is interpretable and works well with both numerical and encoded categorical features.

### 2. Random Forest Classifier

Used as the ensemble model. It improves over a single decision tree by combining multiple trees, reducing overfitting, and improving generalization.

### 3. Multilayer Perceptron (MLP)

Used as the neural network model. It was trained to learn non-linear relationships between features and the income class.

## Preprocessing Steps

The preprocessing stage included:

- Handling missing values represented as `?`
- Removing duplicate rows
- Dropping redundant or less useful features such as `fnlwgt`
- Applying log transformation to skewed financial features
- Encoding categorical variables using one-hot encoding
- Splitting the dataset into training and testing sets
- Scaling features for the MLP model

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Since the dataset is imbalanced, special attention was given to recall and F1-score for the `>50K` class.

## Model Optimisation

Optimisation techniques included:

- GridSearchCV for hyperparameter tuning
- Stratified K-Fold cross-validation
- Feature importance analysis
- Feature selection
- Neural network hyperparameter tuning
- Regularization for the MLP model

## Key Findings

The dataset is imbalanced, with most individuals earning `<=50K`. Because of this, accuracy alone was not enough to evaluate model performance.

The Decision Tree was simple and interpretable but struggled with the minority class. The Random Forest provided the best overall balance between accuracy, recall, and F1-score. The MLP performed reasonably well, but it did not outperform the tree-based models, which is expected for this type of tabular dataset.

## Files

```text
project.ipynb        # Main Jupyter Notebook containing the full project
CMPS460-Project.pdf  # Project description and grading rubric
README.md            # Project documentation
adult.data           # dataset
