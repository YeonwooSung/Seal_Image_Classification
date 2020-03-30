from __future__ import division
import argparse
from dataLoading import load_data


def arg_parse():
    """
    Parse the command line arguments by using the argparse parser.

    :return parser: The argparse parser
    """
    parser = argparse.ArgumentParser(description='Argument parser for the Seal Image Classification')
    parser.add_argument('--mode', dest='mode', type=str, help='binary for binary classification, and multi for multi-class classification',
                        choices=['binary', 'multi'], default='binary')
    parser.add_argument('--estimator', dest='estimator', type=str, choices=['logistic', 'polinomial'], default='logistic')
    parser.add_argument('--data_path', dest='data_path', type=str, default='../data')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    # get arguments by using the argparse
    mode, estimator_name, data_path = args.mode, args.estimator, args.data_path

    # load data frames
    x_train_df, y_train_df, x_test_df = load_data(data_path, mode)


    if mode == 'binary':
        print('Start binary classification')

        #TODO
    else:
        print('Start multi-class classification')

        #TODO
