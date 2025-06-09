# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project is designed for customer churn prediction using Python. We want to identify (and later intervene with) customers who are likely to churn. 

## Files and data description
The churn data set is stored in /data folder. There are three main .py files as described below. Upon running the jupyter notebook "churn_notebook.ipynb", the results of eda, model training, and feature importances are saved inside images folder.

The data set includes information about:
* customer sociodemographic factors such as, gender, age, marital status, income, education, etc.
* customer credit profile information, such as open to buy, month on book, utilization, revolving balance, credit limit
* customer transaction history


Here is the file structure for this project:

.
├── Guide.ipynb          # Getting started and troubleshooting tips
├── churn_notebook.ipynb # Contains the code to predict customer churn
├── churn_library.py     # Contains the functions used in churn_notebook ipynb file.
├── churn_script_logging_and_tests.py # tests and logs
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Data folder
│   └── bank_data.csv
├── images               # Store EDA results as well as final model results
│   ├── eda
│   └── results
├── logs				 # Store logs
└── models               # Store models (logsitic regression and Random Forest)


### 1. churn_library.py
The churn_library.py is a library of functions to find customers who are likely to churn. The main functionality is encapsulated in the churn_library.py file, which includes functions for importing data, performing feature engineering, training models, and more. This module containts all necessary functions, such as import_data, perform_eda, encoder_helper, train_models, train_test_split, perform_feature_engineering, classification_report_image, etc.

### 2. churn_script_logging_and_tests.py
This file contains unit tests for the churn_library.py functions. Each unit tests includes basic assert statements that test functions work properly. The goal of test functions is to checking the returned items aren't empty or folders where results should land have results after the function has been run.

Upon running this test scripts, it stores the logs of any errors and INFO messages into the log/Churn_library.log file.

### 3. constants.py
This file includes all the constant variables used across all files.

## Running Files
All the dependencies are in requirements text files. You can install all dependencies as:

python -m pip install -r requirements_py3.6.txt

All the code for customer churn is inside the churn_notebook.ipynb file. This notebooks is using the functions defined in module churn_library.py.

To test the functions of churn_library module, you can run the churn_script_logging_and_test.py using pytest library. Upon running this library, all the test results will be logged inside the logs/churn_library.log file.


## Contributing
If you want to contribute to this project, feel free to fork the repository and submit a pull request.

