import argparse
import pandas as pd


def arg_parse():
    """
    Parse the command line arguments by using the argparse parser.

    :return parser: The argparse parser
    """
    parser = argparse.ArgumentParser(description='Argument parser for the Seal Image Classification')
    parser.add_argument('--mode', dest='mode', type=str, help='binary for binary classification, and multi for multi-class classification',
                        choices=['binary', 'multi'], default='binary')
    parser.add_argument('--data_path', dest='data_path', type=str, default='/data/CS5014-P2/')

    return parser.parse_args()


def load_data(path, mode):
    """
    Load data from the given path.

    :param path: The file path of the data directory.
    :param mode: Either 'binary' or 'multi'
    :return: Dataframes of the loaded data.
    """
    # build the file path string of the target directory
    data_path = path + '/' + mode

    path1, path2, path3 = data_path + '/X_test.csv', data_path + '/X_train.csv', data_path + '/Y_train.csv'

    x_test_df = pd.read_csv(path1)
    x_train_df = pd.read_csv(path2)
    y_train_df = pd.read_csv(path3)

    return x_train_df, y_train_df, x_test_df


def analyseDataFrame(df):
    """
    Print out information that describes the given dataframe.

    :param df: The target dataframe.
    """
    print(df.head())
    print('Shape of the dataframe: {}'.format(df.shape))
    df.info()
    print('\ndataframe.describe() : ', df.describe())
    print()


if __name__ == '__main__':
    args = arg_parse()
    # get arguments by using the argparse
    mode, data_path = args.mode, args.data_path

    # load data frames
    x_train_df, y_train_df, x_test_df = load_data(data_path, mode)

    df_list = [(x_train_df, 'X_train.csv'), (y_train_df, 'Y_train.csv'), (x_test_df, 'X_test.csv')]

    # iterate the df_list
    for df, name in df_list:
        print('\nDataframe of {}'.format(name))
        analyseDataFrame(df)
