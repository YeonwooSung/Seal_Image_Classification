from __future__ import division
import argparse
import warnings
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsOneClassifier
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from dataLoading import load_data
from features import mapYValues_binary, mapYValues_multiclass, cleanData
from learning import binaryClassification, multiclassClassification


def arg_parse():
    """
    Parse the command line arguments by using the argparse parser.

    :return parser: The argparse parser
    """
    parser = argparse.ArgumentParser(description='Argument parser for the Seal Image Classification')
    parser.add_argument('--mode', dest='mode', type=str, help='binary for binary classification, and multi for multi-class classification',
                        choices=['binary', 'multi'], default='binary')

    # logistic = logistic regression
    # sgd = SGDClassifier
    # xgb = XGBClassifier
    # rf  = RandomForestClassifier
    # vc  = VotingClassifier
    parser.add_argument('--estimator', dest='estimator', type=str, choices=['logistic', 'sgd', 'xgb', 'rf', 'vc'], default='logistic')
    parser.add_argument('--data_path', dest='data_path', type=str, default='../../data')

    return parser.parse_args()


def generate_feature_subset_PCA(X, n=20):
    """
    Generate subset of features by using PCA.

    :param X: The dataset
    :param n: The number of components for the PCA.
    :return newX: generated feature subset
    """
    pca = PCA(n_components=n)
    newX = pca.fit_transform(X)
    return newX


def need_ovo(model_name):
    """
    Check if the given model is a linear model.

    :return Bool: If the given model is a linear model, then returns True. Otherwise, returns False.
    """
    return (model_name == 'logistic') or (model_name == 'sgd')


if __name__ == '__main__':
    # ignore warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

    # generate argument parser to parse command line arguments
    args = arg_parse()
    # get arguments by using the argparse
    mode, estimator_name, data_path = args.mode, args.estimator, args.data_path

    # load data frames
    x_train_df, y_train_df, x_test_df = load_data(data_path, mode)

    # clean the dataframes by replacing NaN values with median values
    x_train_df = cleanData(x_train_df)
    x_test_df = cleanData(x_test_df)

    # list for the number of components of the PCA
    pca_list = [10, 15, 20, 25, 30]
 
    # generate dictionary that maps the estimator name to the corresponding ML model
    estimators = {
        'logistic': [LogisticRegression(), 'LogisticRegression'],
        'sgd': [SGDClassifier(max_iter=10, random_state=42), 'Stochastic Gradient Descent'],
        'xgb': [XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3), 'XGBoost'],
        'vc': [VotingClassifier(estimators=[('LR', LogisticRegression()), ('SGD', SGDClassifier(max_iter=10, random_state=42))], voting='soft'), 'Voting --> LogisticRegression & StochasticGradientDescent'],
        'rf': [RandomForestClassifier(random_state=0), 'RandomForest']
    }

    # check if the program executed for the binary classification
    if mode == 'binary':
        # get the model
        estimator = estimators[estimator_name]
        model = estimator[0]
        print('Start binary classification')
        print('Algorithm = ' + estimator[1])

        # replace strings to numbers (1, 2)
        y_train_df = mapYValues_binary(y_train_df)

        for pca_val in pca_list:
            print('\nPCA :: # of components = {}'.format(pca_val))
            pca_x_train = generate_feature_subset_PCA(x_train_df, n=pca_val)
            pca_x_test = generate_feature_subset_PCA(x_test_df, n=pca_val)

            # perform the binary classification
            binaryClassification(model, pca_x_train, y_train_df, pca_x_test)

    else:
        # get the model
        estimator = estimators[estimator_name]
        model = estimator[0]

        print('Start multi-class classification')
        print('Algorithm = ' + estimator[1])

        # replace strings to numbers
        # {'background': 1, 'dead pup': 2, 'juvenile': 3, 'moulted pup': 4, 'whitecoat': 5}
        y_train_df = mapYValues_multiclass(y_train_df)

        if need_ovo(estimator_name):
            # use OneVsOneClassifier so that the program could perform the multi-class classification with the linear models
            model = OneVsOneClassifier(model)

        for pca_val in pca_list:
            print('\nPCA :: # of components = {}'.format(pca_val))
            pca_x_train = generate_feature_subset_PCA(x_train_df, n=pca_val)
            pca_x_test = generate_feature_subset_PCA(x_test_df, n=pca_val)

            # execute the multi-class classification
            multiclassClassification(model, pca_x_train, y_train_df, pca_x_test)
