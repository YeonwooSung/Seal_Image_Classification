import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from dataLoading import load_data
from features import mapYValues_binary, mapYValues_multiclass, cleanData


BIN_CLASSES = ['background', 'seal']
MULTI_CLASSES = ['background', 'dead pup', 'juvenile', 'moulted pup', 'whitecoat']


def plotData(X, y, X_class, y_class, path='../image/plotData.png'):
    """
    Scatter plot of feature data reduced to 3 components
    :param X: binary feature data
    :param y: binary target data
    :param X_class: multiclass feature data
    :param y_class: mutliclass target data
    """
    bin_classes = len(np.unique(y))
    multi_classes = len(np.unique(y_class))

    assert bin_classes is len(BIN_CLASSES)
    assert multi_classes is len(MULTI_CLASSES)

    data = PCA(n_components=3).fit_transform(X)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121, projection='3d')

    for i in range(bin_classes):
        temp = data[y.iloc[:,0] == i]
        ax.scatter(temp[:,0], temp[:,1], temp[:,2], label=BIN_CLASSES[i])

    ax.set_title("Binary Dataset")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()

    data = PCA(n_components=3).fit_transform(X_class)
    ax = fig.add_subplot(122, projection='3d')

    for i in range(multi_classes):
        temp = data[y_class.iloc[:,0] == i]
        ax.scatter(temp[:,0], temp[:,1], temp[:,2], label=MULTI_CLASSES[i])

    ax.set_title("Mutli-class Dataset")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()

    fig.savefig(path)


def frange(x, y, jump):
    # function to produce floating number ranges
    while x < y:
        yield round(x,2)
        x += jump


def plotVariance(X_bin, X_class, path='../image/plotVariance.png'):
    """
    Plot data variance as a function of number of components.

    :param X_bin: binary feature data
    :param X_class: multiclass feature data
    :param path: The file path of the generated image
    """
    plt.subplots(figsize=(18,5))
    Xs = [X_bin, X_class]

    for i in range(len(Xs)):
        ns = []
        variances = []
        for n in frange(0.9, 1, 0.01):
            pca = PCA(n_components=n)
            X2D = pca.fit_transform(Xs[i])
            variances.append(X2D.shape[1])
            ns.append(n)

        if i == 0:
            title = "Binary Data: "
        else:
            title = "Mutli-class Data: "
        plt.subplot(1, 2, i + 1).set_title(title + "Number of required components with rising variance")
        plt.xlabel("Data variance")
        plt.ylabel("Number of components")
        plt.plot(ns, variances)

    plt.savefig(path)


def getPearsonPlot(X, y):
    """
    Bar graph of pearson-s correlation coefficient
    :param X: feature data
    :param y: target data
    """
    num_of_features = X.shape[1]
    num_of_classes = len(np.unique(y))
    if num_of_classes == 2:
        title = "binary"
    else:
        title = "mutli-class"
    cols = []
    vals = []
    for col in range(0, num_of_features):
        cols.append("X" + str(col + 1))
        vals.append(np.abs(pearsonr(X[:,col], y.iloc[:,0])[0]))
        

    plt.title("Pearson's correlation coefficient for " + title + " data")
    plt.bar(cols, vals)


def generatePearsonPlot(bin_x_train_df, bin_y_train_df, multi_x_train_df, multi_y_train_df, path='../image/generatePearsonPlot.png'):
    plt.subplots(figsize=(18, 4))
    plt.subplot(1, 2, 1)
    getPearsonPlot(PCA(n_components=10).fit_transform(bin_x_train_df), bin_y_train_df)
    plt.subplot(1, 2, 2)
    getPearsonPlot(PCA(n_components=10).fit_transform(multi_x_train_df), multi_y_train_df)
    plt.savefig(path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 visualise.py [1, 2, 3]')
        exit(1)

    mode = 1
    try:
        mode = int(sys.argv[1])
    except:
        print('The first argument should be one of 1, 2, 3')
        exit(1)

    path = '../../data'

    # load data
    bin_x_train_df, bin_y_train_df, bin_x_test_df = load_data(path, 'binary')
    multi_x_train_df, multi_y_train_df, multi_x_test_df = load_data(path, 'multi')

    # clean the dataframes by replacing NaN values with median values
    bin_x_train_df = cleanData(bin_x_train_df)
    bin_x_test_df = cleanData(bin_x_test_df)
    multi_x_train_df = cleanData(multi_x_train_df)
    multi_x_test_df = cleanData(multi_x_test_df)

    # standardize y values
    bin_y_train_df = mapYValues_binary(bin_y_train_df)
    multi_y_train_df = mapYValues_multiclass(multi_y_train_df)

    # check the commandline argument, and generate corresponding plot
    if mode == 1:
        plotVariance(bin_x_train_df, multi_x_train_df)
    elif mode == 2:
        plotData(bin_x_train_df, bin_y_train_df, multi_x_train_df, multi_y_train_df)
    elif mode == 3:
        generatePearsonPlot(bin_x_train_df, bin_y_train_df, multi_x_train_df, multi_y_train_df)
