'''
This library contains modular functions to build and
evaluate credit card customer churn prediction model

Author: Femi Bolarinwa
August, 2023
'''

# import libraries
import warnings
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# suppressing unnecessary warnings while executing some code
warnings.filterwarnings("ignore")
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df.drop(
        columns=[
            'Unnamed: 0',
            'CLIENTNUM',
            'Attrition_Flag'],
        inplace=True)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.xlabel('Churn (0/1)')
    plt.ylabel('Count')
    plt.title('Churn Histogram')
    plt.savefig('./images/eda/churn_histogram.png')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age Distribution')
    plt.savefig('./images/eda/customer_age_histogram.png')

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion')
    plt.title('Marital Status')
    plt.savefig('./images/eda/marital_status_barplot.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Transaction count distribution')
    plt.savefig('./images/eda/total_trans_ct.png')

    plt.figure(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Correlation Heatmap')
    plt.savefig('./images/eda/correlation_map.png')


def encoder_helper(df):
    '''
    helper function to hot-encode categorical variable

    input:
            df: pandas dataframe
    output:
            X: predictor variables
            y: target variable (churn)
    '''
    y = df['Churn']
    X = df.drop(["Churn"], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def perform_feature_engineering(X, y):
    '''
    function to split data into train-test.

    input:
              X: predictor variables
              y: target variable (churn)
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models.
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # grid search for best rf parameters
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # making predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save the model as a pickle in a file
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/lrc_model.pkl')

    plt.figure(figsize=(20, 10))
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig('./images/results/lr_roc_curve.png')

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/lr_rf_roc_curve.png')
    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    plt.figure(figsize=(20, 10))
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.1, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.5, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.1, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/classification_report_rf.png')

    plt.figure(figsize=(20, 10))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.1, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.5, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.1, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/classification_report_lr.png')


def feature_importance_plot(model, X_train):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_train: pandas dataframe of X values

    output:
             None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_train.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 20))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_train.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_train.shape[1]), names, rotation=90)
    plt.savefig('./images/results/rf_feature_importance.png')


if __name__ == "__main__":
    '''
    executing modular functions in the right order to build and evaluate prediction model
    '''
    my_df = import_data("./data/bank_data.csv")
    perform_eda(my_df)
    my_X, my_y = encoder_helper(my_df)
    my_X_train, my_X_test, my_y_train, my_y_test = perform_feature_engineering(
        my_X, my_y)
    my_y_train_preds_lr, my_y_train_preds_rf, my_y_test_preds_lr, my_y_test_preds_rf = train_models(
        my_X_train, my_X_test, my_y_train, my_y_test)
    classification_report_image(
        my_y_train,
        my_y_test,
        my_y_train_preds_lr,
        my_y_train_preds_rf,
        my_y_test_preds_lr,
        my_y_test_preds_rf)
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/lrc_model.pkl')
    feature_importance_plot(rfc_model, my_X_train)
