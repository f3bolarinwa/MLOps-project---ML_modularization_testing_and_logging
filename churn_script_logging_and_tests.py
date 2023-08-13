'''
This library contains unit tests to test modular functions in churn_library.py

Author: Femi Bolarinwa
August, 2023
'''

# importing libraries
import os
import logging
from churn_library import (import_data, perform_eda, encoder_helper,
                           perform_feature_engineering, train_models)

# log set up
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():  # (import_data):
    '''
    test data import
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():  # (perform_eda):
    '''
    test perform eda function
    '''

    try:
        df = import_data("./data/bank_data.csv")
        logging.info(
            "Testing perform_eda: SUCCESS - was able to load dataframe")
    except (FileNotFoundError, KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing perform_eda: FAIL - was not able to load dataframe")
        raise err

    try:
        perform_eda(df)
        logging.info(
            "Testing perform_eda: SUCCESS - was able to execute perform_eda")
    except (KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing perform_eda: FAIL - was not able to execute perform_eda")
        raise err

    try:
        assert os.path.isfile("./images/eda/churn_histogram.png")
        assert os.path.isfile("./images/eda/customer_age_histogram.png")
        assert os.path.isfile("./images/eda/marital_status_barplot.png")
        assert os.path.isfile("./images/eda/total_trans_ct.png")
        assert os.path.isfile("./images/eda/correlation_map.png")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Could not generate one or more EDA plots")
        raise err


def test_encoder_helper():  # (encoder_helper):
    '''
    test encoder helper
    '''

    try:
        df = import_data("./data/bank_data.csv")
        logging.info(
            "Testing encoder_helper: SUCCESS - was able to load dataframe")
    except (FileNotFoundError, KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing encoder_helper: FAIL - was not able to load dataframe")
        raise err

    try:
        X, y = encoder_helper(df)
        logging.info(
            "Testing encoder_helper: SUCCESS - was able to execute encoder_helper.")
    except (KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing encoder_helper: FAIL - was not able to execute encoder_helper.")
        raise err

    try:
        assert X.shape[0] > 0
        assert X.shape[1] > 0
        assert y.shape[0] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: FAIL -  One of the generated dataframes doesn't appear to have rows and/or columns")
        raise err


def test_perform_feature_engineering():  # (perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    try:
        df = import_data("./data/bank_data.csv")
        logging.info(
            "Testing perform_feature_engineering: SUCCESS - was able to load dataframe")
    except (FileNotFoundError, KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing perform_feature_engineering: FAIL - was not able to load dataframe")
        raise err

    try:
        X, y = encoder_helper(df)
        logging.info(
            "Testing perform_feature_engineering: SUCCESS - was able to execute encoder_helper.")
    except (KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing perform_feature_engineering: FAIL - was not able to execute encoder_helper.")
        raise err

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(X, y)
        logging.info(
            "Testing perform_feature_engineering: SUCCESS - was able to execute perform_feature_engineering")
    except (KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing perform_feature_engineering: FAIL - was not able to exxecute perform_feature_engineering")
        raise err

    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: One of the generated predictor vector 'X' or target vector 'y' doesn't appear to have rows and/or columns")
        raise err


def test_train_models():  # (train_models):
    '''
    test train_models
    '''

    try:
        df = import_data("./data/bank_data.csv")
        logging.info(
            "Testing train_models: SUCCESS - was able to load dataframe")
    except (FileNotFoundError, KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing train_models: FAIL - was not able to load dataframe")
        raise err

    try:
        X, y = encoder_helper(df)
        logging.info(
            "Testing train_models: SUCCESS - was able to execute encoder_helper.")
    except (KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing train_models: FAIL - was not able to execute encoder_helper.")
        raise err

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(X, y)
        logging.info(
            "Testing train_models: SUCCESS - was able to execute perform_feature_engineering")
    except (KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing train_models: FAIL - was not able to execute perform_feature_engineering")
        raise err

    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except AssertionError as err:
        logging.error("Testing train_models: One of the generated predictor vector 'X' or target vector 'y' doesn't appear to have rows and/or columns. Check perform_feature_engineering")
        raise err

    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info(
            "Testing train_models: SUCCESS - was able to train models")
    except (KeyError, ValueError, NameError, TypeError) as err:
        logging.error(
            "Testing train_models: FAIL - was not able to train models")
        raise err

    try:
        assert os.path.isfile("./images/results/lr_rf_roc_curve.png")
        assert os.path.isfile("./images/results/lr_roc_curve.png")
        assert os.path.isfile("./models/lrc_model.pkl")
        assert os.path.isfile("./models/rfc_model.pkl")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Could not save model(s) or generate model plots")
        raise err


if __name__ == "__main__":
    '''
    executing test functions to perform unit test on modular functions in churn_library.py
    '''
    test_import()  # (import_data)
    test_eda()  # (perform_eda)
    test_encoder_helper()  # (encoder_helper)
    test_perform_feature_engineering()  # (perform_feature_engineering)
    test_train_models()  # (train_models)
