from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt


def explore(dataset):
    print(dataset.shape)
    print(dataset.head(20))
    print(dataset.describe())
    print(dataset.groupby('class').size())

    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()
    dataset.hist()
    plt.show()
    scatter_matrix(dataset)
    plt.show()
