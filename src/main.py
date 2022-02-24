import pandas as pd
from sklearn.model_selection import train_test_split

from exploration import explore
from training import train
from prediction import predict

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

explore(dataset)
train(x_train, x_validation, y_train, y_validation)
predict(x_train, x_validation, y_train, y_validation)
