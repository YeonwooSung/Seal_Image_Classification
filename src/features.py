import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler



def cleanData(df):
    """
    Clean data by replace NaN values with median.
    """
    # use SimpleImputer to clean training data with median values
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df)

    # transform the features
    x = imputer.transform(df)
    # convert the numpy array x to dataframe
    df = pd.DataFrame(x, columns=df.columns, index=list(df.index.values))

    # standardize columns whose max value is greater than threshold value
    # use the MinMaxScaler to standardize the target columns
    df = standardize_big_values_by_min_max(df)

    return df


def standardize_big_values_by_min_max(df, thresh=100):
    # find columns whose max value is greater than threshold value
    overThresh = df.max() > thresh
    targetCols = overThresh[overThresh].index.values

    scaler = MinMaxScaler()
    scaler.fit_transform(df[targetCols])

    return df


def mapYValues_binary(df):
    mapping = {'background': 1, 'seal': 2}
    df.replace({'background': mapping}, inplace=True)
    return df


def mapYValues_multiclass(df):
    mapping = {'background': 1, 'dead pup': 2, 'juvenile': 3, 'moulted pup': 4, 'whitecoat': 5}
    df.replace({'background': mapping}, inplace=True)
    return df
