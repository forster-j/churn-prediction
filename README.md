# Churn prevention using ML

## Overview
This project aims to provide a comprehensive framework for practicing churn prediction, a crucial task for businesses aiming to retain customers. Churn prediction involves identifying customers who are likely to stop using a service or product, enabling proactive retention strategies.

We chose this to deepen our knowledge about data science and obtain some new skills, too.
First we did some data cleaning and EDA, then trained three different ML models and compared them. Find our best model in 07_conclusion.ipynb.

#

*This project serves as a portfolio project. Team members are: [Johannes Forster](https://github.com/forster-j), [Dr. John Alexander Preuss](https://github.com/PAJJAP) and [Tobias Pötzl](https://github.com/TopAudioData). Additional input by [Jagoda Mika](https://github.com/JagodaMika92).*

## About the Dataset
We used the following dataset from kaggle: [Customer Churn](https://www.kaggle.com/datasets/undersc0re/predict-the-churn-risk-rate/data). Since no additional information was give as to the nature of the dataset, it was sometimes difficult to make assumptions because of missing domain knowledge.

Before cleaning and feature engineering, it contained 
* 36992 observations
* 24 features, including 'churn_risk_score' as y
* numerical, categorical and time series data

The final dataset contained
* 34716 observations
* 20 features, including 'churn_risk_score' as y
* numerical, categorical and time series data

## Repository Structure

| File | Content |
|---- | ----------|
|/data| folder to store dataframes (empty by default)|
|/model| folder to store models (empty by default)|
|01_import_dataset.ipynb| Import the dataset|
|02_metrics.ipynb| Choose evaluation metrics for the models|
|03_EDA.ipynb| Exploratory Data Analysis|
|04_data_cleaning_feature.ipynb | Clean data and feature engineering|
|05_01_logreg.ipynb| Train model Logistic Regression and plot metrics|
|05_02_decision_tree.ipynb| Train model Decision Tree and plot metrics|
|05_03_catboost.ipynb| Train model CatBoost and plot metrics|
|06_01_evaluation_logreg.ipynb| Evaluate the prediction outcomes of Logistic Regression model|
|06_02_evaluation_decision_tree.ipynb| Evaluate the prediction outcomes of Decision Tree model|
|06_03_evaluation_catboost.ipynb| Evaluate the prediction outcomes of CatBoost model|
|07_conclusion.ipynb| Comparisons between the three models, their feature importance and model recommendation|
|myfunctions.py| Contains custom functions for prediction metrices, plot categorical features, and save metrices|

## Getting Started

Clone this repository.
Set up a Python environment and install the necessary dependencies listed in requirements.txt.
Utilize the provided scripts in the src directory for model training, data preprocessing, etc.

## Set up your Environment

### **`macOS`** type the following commands : 


- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```