'''
test library module that includes testing most functions in churn_library module

Author: Pouya Sharifi
Date: 1/7/2023

'''


import os
import logging
from pathlib import Path
import pytest
from sklearn.model_selection import train_test_split
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper
from churn_library import train_models, train_test_split, perform_feature_engineering
# from churn_library import *
from constants import TEST_SIZE, QUANT_COLS, PATH_PREFIX
from constants import RESPONSE, CAT_COLS


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture(scope="module")
def path():
    '''
    just the path to the data to be used across the whole script.
    '''
    return "./data/bank_data.csv"


def create_label(df):
    '''
    this function takes the dataframe as input and outputs the dataframe with encoded label churn.
      
    '''
    df[RESPONSE] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def test_import(path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(path)
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


def test_eda(path):
    '''
    test perform eda function
    '''
    df = pd.read_csv(path)
    df = create_label(df)
    try:
        perform_eda(df)
        logging.info("SUCCESS: Testing EDA function on the dataframe")
    except FileNotFoundError as err:
        logging.error(
            "ERROR: Testing EDA function: the folder or directory not found")
        logging.info(
            "DEBUG: Testing EDA function: You need to include the whole prefix path '/workspace/home/' in your output path")
        raise err

    try:
        assert len(os.listdir(PATH_PREFIX + '/images/results/')) > 0
        logging.info(
            "SUCCESS: The Directory is not empty and the EDA results are saved in results folder")

    except AssertionError as err:
        logging.error(
            "ERROR: The Directory is empty and the results are NOT saved here")
        raise err


def test_encoder_helper(path):
    '''
    test encoder helper
    '''
    df = pd.read_csv(path)
    df = create_label(df)
    length = df.shape[1]
    print(length)
    print(CAT_COLS)
    try:
        print(df[RESPONSE])
        df_encoded = encoder_helper(df, CAT_COLS, RESPONSE)
        logging.info("SUCCESS: Testing encoder function on dataframe")
    except NameError as err:
        logging.error("ERROR: the response variable is not an string")
        raise err

    try:
        assert df_encoded.shape[1] == length + len(CAT_COLS)
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: the output did not create enough encoded columns")
        raise err


def test_perform_feature_engineering(path):
    '''
    test perform_feature_engineering
    '''
    df = pd.read_csv(path)
    df = create_label(df)
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, cols= QUANT_COLS)
        logging.info("SUCCESS: Testing perform_feature_engineering function")

    except NameError as err:
        logging.info(
            "ERROR: Train test split is not defined, import the library")
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
            "ERROR: Testing perform_feature_engineering: The outputs don't appear to have rows and columns")
        raise err


# def test_train_models(path):
    '''
    test train_models
    '''
    df = pd.read_csv(path)
    X = df[QUANT_COLS]
    df = create_label(df)
    y = df[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42)

    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("SUCCESS: Testing train_models function")
    except NameError as err:
        logging.error("ERROR: Testing train_models: a module is not found")
        raise err

    try:
        path = Path(PATH_PREFIX + '/models/rfc_model.pkl')
        assert path.is_file()
        path = Path(PATH_PREFIX + '/models/logistic_model.pkl')
        assert path.is_file()
    except AssertionError as err:
        logging.error(
            "Testing train_models: the models are not saved in to models folder")

    try:
        path = Path(PATH_PREFIX + '/images/results/logistic_model_auc_roc.png')
        assert path.is_file()

        path = Path(PATH_PREFIX + '/images/results/rfc_model_auc_roc.png')
        assert path.is_file()

    except AssertionError as err:
        logging.error("Testing train_models: the model auc plots are not saved in to images/results folder")


# if __name__ == "__main__":
#     pytest ./churn_script_logging_and_tests.py
    