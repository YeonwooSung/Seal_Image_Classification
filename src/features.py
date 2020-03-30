import pandas as pd
from sklearn.impute import SimpleImputer



def cleanData(df):
    # use SimpleImputer to clean training data with median values
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df)

    # transform the features
    x = imputer.transform(df)
    # convert the numpy array x to dataframe
    df = pd.DataFrame(x, columns=df.columns, index=list(df.index.values))

    #TODO find_columns_by_max_val

    return df


def find_columns_by_max_val(df, thresh=100):
    # find columns whose max value is greater than threshold value
    overThresh = df.max() > thresh
    targetCols = overThresh[overThresh].index.values

    targetCols_df = df[targetCols]

    #TODO standardize

def mapYValues_binary(df):
    mapping = {'background': 1, 'seal': 2}
    df.replace({'background': mapping}, inplace=True)
    return df


def mapYValues_multiclass(df):
    mapping = {'background': 1, 'dead pup': 2, 'juvenile': 3, 'moulted pup': 4, 'whitecoat': 5}
    df.replace({'background': mapping}, inplace=True)
    return df
