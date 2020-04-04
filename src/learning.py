from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold #TODO ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support #TODO
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from dataLoading import load_data
from features import mapYValues_binary, mapYValues_multiclass, cleanData




def generate_estimators_dict():
    # generate dictionary that maps the estimator name to the corresponding ML model
    estimators = {
        'logistic': [LogisticRegression(), 'LogisticRegression'],
        'sgd': [SGDClassifier(max_iter=10, random_state=42), 'Stochastic Gradient Descent'],
        'xgb': [XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3), 'XGBoost'],
        #'vc': [VotingClassifier(estimators=[('LR', LogisticRegression()), ('SGD', SGDClassifier(max_iter=10, random_state=42))], voting='soft'), 'Voting --> LogisticRegression & SVM'],
        'rf': [RandomForestClassifier(random_state=0), 'RandomForest']
    }

    return estimators


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
    model_for_validation = clone(model)

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    # calculate the accuracy
    accuracy = accuracy_score(y_test, pred)


    # validate the model with the K-Fold cross validation

    accuracy_kfold = 0
    kf = KFold(n_splits=k_val)

    num_of_iteration = 0

    # loop for the K-Fold
    for train_index, test_index in kf.split(x):
        x_train1, x_test1 = x[train_index], x[test_index]
        y_train1, y_test1 = y.loc[train_index], y.loc[test_index]

        if len(np.unique(y_train1)) == 1:
            continue

        model_for_kfold.fit(x_train1, y_train1)
        pred1 = model.predict(x_test1)
        accuracy_kfold += accuracy_score(y_test1, pred1)

    if num_of_iteration == 0:
        accuracy_kfold = accuracy
    else:
        accuracy_kfold = (accuracy_kfold / num_of_iteration)

    # check if the function should terminate at this point.
    if not draw_plot:
        return accuracy, accuracy_kfold

    # initialise the model to continue the validation
    model = model_for_validation

    BIN_CLASSES = ['background', 'seal']
    MULTI_CLASSES = ['background', 'dead pup', 'juvenile', 'moulted pup', 'whitecoat']
    num_of_classes = len(np.unique(y))
    classes = BIN_CLASSES if num_of_classes <= 2 else MULTI_CLASSES

    # get the name of the model
    model_name = type(model).__name__
    suffix = '_binary' if num_of_classes <= 2 else '_multi'
    # generate file path string for the confusion matrix
    file_path = '../image/cm_' + model_name + suffix

    # plot the confusion matrix
    plot_confusion_matrix(y_test, pred, classes, normalize=True, path=file_path)


    model.fit(x, y)
    pred = model.predict(x)

    # get precision, recall, and f1 score for the trained model
    precision_val, recall_val, f_score, support = precision_recall_fscore_support(y, pred, average='macro') #TODO macro, micro, weighted, None
    print(' - Precision = {}'.format(precision_val))
    print(' - Recall    = {}'.format(recall_val))
    print(' - F1 score  = {}'.format(f_score))

    return accuracy, accuracy_kfold



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

    :return: Returns 1) the cross validation score 2) the number of components for the best model
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

    return best_score, best_score_n



if __name__ == '__main__':
    # ignore warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

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

    # get the dictionary of estimators
    estimators = generate_estimators_dict()

    # use for loop to iterate through the dictionary of estimators
    for estimator_name in estimators:
        curr = estimators[estimator_name]
        model, description = curr[0], curr[1]

        # clone model to use for multi-class classification
        model_multi = clone(model)

        print('\nAlgorithm : {}'.format(description))

        # get the name of the model
        model_name = type(model).__name__

        # generate file path strings with the model name
        path_bin = '../image/{}_binary.png'.format(model_name)
        path_multi = '../image/{}_multi.png'.format(model_name)

        best_score, best_score_n = plotScoreFromN(bin_x_train_df, bin_y_train_df, model, path=path_bin)
        print('The best score of binary classification : {} :: PCA(n={})'.format(best_score, best_score_n))

        # clean the pyplot figure to draw next figures
        plt.clf()

        if need_ovo(estimator_name):
            # use OneVsOneClassifier so that the program could perform the multi-class classification with the linear models
            model_multi = OneVsOneClassifier(model_multi)

        best_score, best_score_n = plotScoreFromN(multi_x_train_df, multi_y_train_df, model_multi, path=path_multi)
        print('The best score of multi-class classification : {} :: PCA(n={})'.format(best_score, best_score_n))

        # clean the pyplot figure to draw next figures
        plt.clf()


        # validation

        print('Validation start - ' + model_name)

        pca_val = 30

        bin_pca_x_train = generate_feature_subset_PCA(bin_x_train_df, n=pca_val)
        bin_pca_x_test = generate_feature_subset_PCA(bin_x_test_df, n=pca_val)
        multi_pca_x_train = generate_feature_subset_PCA(multi_x_train_df, n=pca_val)
        multi_pca_x_test = generate_feature_subset_PCA(multi_x_test_df, n=pca_val)

        print('1) Binary classification :')
        # do the validation for the given model with the training dataset
        accuracy_score, accuracy_kfold = validation(model, bin_pca_x_train, bin_y_train_df)
        # clean the pyplot figure to draw next figures
        plt.clf()

        print(' - Accuracy score of validation ......... {}'.format(accuracy_score))
        print(' - Accracy score of K-Fold validation ... {}'.format(accuracy_kfold))

        print('2) Multi-class classification :')
        # do the validation for the given model with the training dataset
        accuracy_score, accuracy_kfold = validation(model_multi, multi_pca_x_train, multi_y_train_df)
        # clean the pyplot figure to draw next figures
        plt.clf()

        print(' - Accuracy score of validation ......... {}'.format(accuracy_score))
        print(' - Accracy score of K-Fold validation ... {}'.format(accuracy_kfold))
