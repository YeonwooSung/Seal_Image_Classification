from __future__ import division
import argparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
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
    parser.add_argument('--estimator', dest='estimator', type=str, choices=['logistic', 'sgd', 'ovo', 'polinomial'], default='logistic')
    parser.add_argument('--data_path', dest='data_path', type=str, default='../../data')

    return parser.parse_args()


def need_ovo(model_name):
    return (model_name == 'logistic') or (model_name == 'sgd')


if __name__ == '__main__':
    args = arg_parse()
    # get arguments by using the argparse
    mode, estimator_name, data_path = args.mode, args.estimator, args.data_path

    # load data frames
    x_train_df, y_train_df, x_test_df = load_data(data_path, mode)

    # clean the dataframes by replacing NaN values with median values
    x_train_df = cleanData(x_train_df)
    x_test_df = cleanData(x_test_df)

    estimators = {
        'logistic': [LogisticRegression(C=1e5), 'LogisticRegression'],
        'sgd': [SGDClassifier(max_iter=10, random_state=42), 'Stochastic Gradient Descent']
    }

    if mode == 'binary':
        print('Start binary classification')
        print('Algorithm = ' + estimator_name)

        # replace strings to numbers (1, 2)
        y_train_df = mapYValues_binary(y_train_df)

        #TODO
        model = estimators[estimator_name]
        binaryClassification(model, x_train_df, y_train_df, x_test_df)

    else:
        print('Start multi-class classification')
        print('Algorithm = ' + estimator_name)

        # replace strings to numbers
        # {'background': 1, 'dead pup': 2, 'juvenile': 3, 'moulted pup': 4, 'whitecoat': 5}
        y_train_df = mapYValues_multiclass(y_train_df)

        #TODO
        model = estimators[estimator_name]

        if need_ovo(estimator_name):
            model = OneVsOneClassifier(model)
        multiclassClassification(model, x_train_df, y_train_df, x_test_df)
