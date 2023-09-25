# Project: Predict Customer Churn

A program requirement for Machine Learning DevOps Engineer Nanodegree @ Udacity School of Artificial Intelligence

## Project Description
This project involves creating and implementing modular functions and classes to train a machine learning classification model to predict credit card customers that are most likely to churn. 

The completed project includes a Python package that follows coding (pylint, PEP8) and engineering best practices for implementing software (modularity, documentation, testing and logging).

## Files and data description
Overview of the files and data present in the root directory: 

1)data/bank_data.csv: data containing customer information.

2)churn_notebook.ipynb: interactive jupyter notebook for machine learning concept development (i.e. data overview, analysis and ML model training and testing).

3)churn_library.py: contains modular functions to implements machine learning concept developed in churn_notebook.ipynb.

4)models/lrc_model.pkl: logistic regression classification model saved/dumped (in pkl format) when churn_library.py is executed.

5)models/rfc_model.pkl: random forest classification model saved/dumped (in pkl format) when churn_library.py is executed.

6)images/eda: contains images generated from exploratory data analysis (univariate/multivariate analysis) when churn_library.py is executed.

7)images/results: contains model performance reports (classification report, ROC curve, feature importances) generated when churn_library.py is executed.

8)churn_script_logging_and_tests.py: contains unit tests to test modular functions in churn_library.py

9)logs/churn_library.log: contains logs for unit tests in churn_script_logging_and_tests.py

10)requirements.txt: list of needed dependencies and libraries to successfully run package.


## Running Files

1)To install dependencies, execute in terminal: ipython -m pip install -r requirements.txt

2)Peruse churn_notebook.ipynb to understand machine learning concept involved.

3)In terminal, execute: ipython churn_script_logging_and_tests.py

Expect logs/churn_library.log to be populated.
    
4)In terminal, execute: ipython churn_library.py

Expect the following folders to be populated: images, models


## Dependencies

joblib==1.1.0

matplotlib==3.5.2

numpy==1.21.5

pandas==1.4.4

scikit_learn==1.0.2

seaborn==0.11.2
