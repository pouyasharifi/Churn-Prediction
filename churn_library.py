'''
Churn Library module that includes all necessary function to run churn prediction notebook.

Author: Pouya Sharifi
Date: 1/7/2023

'''

# import libraries
import os
import joblib
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from constants import *

# sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    print('reading the csv file from {} path', pth)
    dataframe = pd.read_csv(pth)   # read the csv file as a Pandas DataFrame
    return dataframe


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    df[RESPONSE].hist()  # plot histogram of response variable
    # save figure in the images folder
    plt.savefig(PATH_PREFIX + '/images/eda/hist_{}.png'.format(RESPONSE))

    # for all quantitative columns, plot and save the histograms
    for col in QUANT_COLS:
        plt.figure(figsize=(20, 10))
        sns.histplot(df[col], stat='density', kde=True)
        plt.savefig(PATH_PREFIX + '/images/eda/hist_{}.png'.format(col))

    # Bivariate - Correlation plot
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(PATH_PREFIX + '/images/eda/correlation_plot.png')

    # for all categorical columns, plot the value counts
    for col in CAT_COLS:
        plt.figure(figsize=(20, 10))
        df[col].value_counts('normalize').plot(kind='bar')
        plt.savefig(
            PATH_PREFIX +
            '/images/eda/value_count_plot_{}.png'.format(col))


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        col_lst = []
        category_groups = df.groupby(col)[response].mean()

        for val in df[col]:
            col_lst.append(category_groups.loc[val])

        df[col + '_Churn'] = col_lst

    return df


def perform_feature_engineering(df, cols = FINAL_COLUMNS):
    '''
    input:
              df: pandas dataframe
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()

    X[cols] = df[cols]
    y = df[RESPONSE]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42)

    return X_train, X_test, y_train, y_test


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

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(PATH_PREFIX + '/images/results/classification_report_rf.png')
    plt.clf()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(PATH_PREFIX + '/images/results/classification_report_lr.png')
    plt.clf()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)
    print("Saving the feature importance plot in {} path".format(output_pth))


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)

    # Create a logistic regression model, if fails use other solvers
    lrc = LogisticRegression(solver='lbfgs', max_iter=MAX_ITER)

    # perform hyperparam tuning using grid search using 5-fold cross validation
    # and fit the model
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAM_GRID, cv=5)
    cv_rfc.fit(X_train, y_train)
    rfc_model = cv_rfc.best_estimator_

    # fit logistic regression model
    lrc.fit(X_train, y_train)

    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_rf = rfc_model.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    # save and dump both models in models folder
    joblib.dump(rfc_model, PATH_PREFIX + '/models/rfc_model.pkl')
    joblib.dump(lrc, PATH_PREFIX + '/models/logistic_model.pkl')

    # AUC ROC plot for logistic regression model and save it in
    # /images/results/ folder
    plt.figure(figsize=(15, 8))
    plot_roc_curve(lrc, X_test, y_test)
    plt.savefig(PATH_PREFIX + '/images/results/logistic_model_auc_roc.png')

    # AUC ROC plot for random forest model and save it in /images/results/
    # folder
    plt.figure(figsize=(15, 8))
    plot_roc_curve(rfc_model, X_test, y_test)
    plt.savefig(PATH_PREFIX + '/images/results/rfc_model_auc_roc.png')


# if __name__ == "__main__":
#     df = import_data(DATA_PATH)
#     df[RESPONSE] = df['Attrition_Flag'].apply(
#         lambda val: 0 if val == "Existing Customer" else 1)
#     df = encoder_helper(df, CAT_COLS, RESPONSE)
#     X_train, X_test, y_train, y_test = perform_feature_engineering(df)
#     train_models(X_train, X_test, y_train, y_test)
