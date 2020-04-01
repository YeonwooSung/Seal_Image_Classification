from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone


def binaryClassification(model, x_train_df, y_train_df, x_test_df):
    # clone the model to constructs a new estimator with the same parameters
    # this cloned model will be used for the validation
    cloned_model = clone(model)

    # do the validation to test the model
    validation(cloned_model, x_train_df, y_train_df)

    model.fit(x_train_df, y_train_df)


def multiclassClassification(model, x_train_df, y_train_df, x_test_df):
    print('')


def validation(model, x, y):
    print('Start validation for the given model')
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    # calculate the accuracy
    accuracy = accuracy_score(y_test, pred)
    print(accuracy)

    #TODO cross validation (K-Fold, GridSearchCV, etc)
    #TODO confusion matrix, precision, recall, f1
