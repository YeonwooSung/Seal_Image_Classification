from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def binaryClassification(model, x_train_df, y_train_df, x_test_df):
    columns = x_train_df.columns.values
    x_train_df = x_train_df[columns[0:900]]

    x_train, x_test, y_train, y_test = train_test_split(x_train_df, y_train_df)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print(accuracy)


def multiclassClassification(model, x_train_df, y_train_df, x_test_df):
    print('')
