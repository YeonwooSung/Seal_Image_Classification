from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold #TODO ShuffleSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataLoading import load_data
from features import mapYValues_binary, mapYValues_multiclass, cleanData



def train_and_validate_model(model, x_train_df, y_train_df, x_test_df, mode, n):
    # clone the model to constructs a new estimator with the same parameters.
    # this cloned model will be used for the validation.
    cloned_model = clone(model)

    classification_mode = 'multi-class' if mode == 'multi' else mode
    print('Start validation of {} classification with PCA(n={})'.format(classification_mode, n))
    # do the validation with the given training set
    accuracy_score, accuracy_kfold = validation(cloned_model, x_train_df, y_train_df, draw_plot=False)
    print(' - Accuracy score of validation ......... {}'.format(accuracy_score))
    print(' - Accracy score of K-Fold validation ... {}'.format(accuracy_kfold))

    print('Start training model with PCA(n={})'.format(n))

    # train model with the training set
    model.fit(x_train_df, y_train_df)
    # make the trained model to predict answers for the testing set
    preds = model.predict(x_test_df)

    # generate suitable file path string
    fileName = 'prediction_bin' if mode == 'binary' else 'prediction_multi'
    name = '../result/{}_{}.csv'.format(fileName, n)

    # save the predictions as a csv file with a suitable name
    pd.DataFrame(preds, columns=['predictions']).to_csv(name)
    print('Generated the csv file for ' + mode + ' classification with PCA(n={}) successfully'.format(n))


def validation(model, x, y, draw_plot=True, k_val=5):
    # clone the model to constructs a new estimator with the same parameters.
    # this cloned model will be used for the K-Fold validation.
    model_for_kfold = clone(model)

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    # calculate the accuracy
    accuracy = accuracy_score(y_test, pred)


    # validate the model with the K-Fold cross validation

    accuracy_kfold = 0
    kf = KFold(n_splits=k_val)

    # loop for the K-Fold
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_for_kfold.fit(x_train, y_train)
        pred = model.predict(x_test)
        accuracy_kfold += accuracy_score(y_test, pred)

    accuracy_kfold = (accuracy_kfold / k_val)

    # check if the function should terminate at this point.
    if not draw_plot:
        return accuracy, accuracy_kfold


    #TODO cross validation (GridSearchCV, etc)
    #TODO confusion matrix, precision, recall, f1


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, path='../image/plot_confusion_matrix.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), 
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title=title, ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    # calculate the threshold value
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Color the cell with white if the cm value is greater than threshold. Otherwise, color it with black.
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    # set up the layout to make matplotlib automatically adjusts subplot parameters to give specified padding.
    fig.tight_layout()

    # save figure as an image file.
    plt.savefig(path)


def plotScoreFromN(X_test, y_test, model, path=''):
    """
    Plot testing accuracy against a range of numbers of components.

    :param X_test: feature data
    :param y_test: target data
    :param model: model to evaluate
    :param path: file path of the generated image
    :return: number of components for the best model
    """
    ns = []
    scores = []
    tr_scores = []
    best_score = 0
    best_score_n = 0
    modelNew = clone(model)

    for n in range(1, 100):
        model = make_pipeline(PCA(n_components=n), clone(modelNew))
        score = cross_val_score(model, X_test, y_test, cv=3).mean()

        if score > best_score:
            best_score = score
            best_score_n = n
        ns.append(n)
        scores.append(score)

        transf = PCA(n_components=n).fit_transform(X_test)
        modelNew.fit(transf, y_test)
        y_pred = modelNew.predict(transf)
        tr_scores.append(accuracy_score(y_test, y_pred))

    plt.plot(ns, scores, label="Testing score")
    plt.plot(ns, tr_scores, label="Training score")
    plt.title("Number of required components as a function of model score")
    plt.xscale
    plt.xticks(range(0,len(ns), 5))
    plt.xlabel("Number of components")
    plt.axvline(x= best_score_n, c='black', label="Best testing score (score=" + str(round(best_score, 2)) + ", x=" + str(best_score_n) + ")")
    plt.ylabel("Cross-validation score")
    plt.legend()

    # save figure as an image file.
    plt.savefig(path)

    return best_score_n



if __name__ == '__main__':
    #TODO command line argument

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

    #TODO check command line argument, and execute suitable method
