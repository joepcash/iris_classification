from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def train(x_train, x_validation, y_train, y_validation):
    models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('NB', GaussianNB()),
              ('SVM', SVC(gamma='auto'))]

    results = []
    names = []
    for name, model in models:
        k_fold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, x_train, y_train, cv=k_fold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print(f"{name}: {round(cv_results.mean(), 3)} ± {round(cv_results.std(), 3)}")

    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    return x_train, y_train, x_validation, y_validation
