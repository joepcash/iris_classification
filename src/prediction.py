from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def predict(x_train, x_validation, y_train, y_validation):
    model = SVC(gamma='auto')
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)

    print(accuracy_score(y_validation, predictions))
    print(confusion_matrix(y_validation, predictions))
    print(classification_report(y_validation, predictions))
