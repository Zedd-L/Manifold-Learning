from sklearn import datasets
from sklearn.neighbors import KDTree

def gen_data(n_points):
    # X, color = datasets.make_s_curve(n_points, random_state=0)
    X, color = datasets.make_swiss_roll(n_points, random_state=0)
    return X, color

def Knn(X, n_neighbors):
    '''
    :param X: dataset shape as (n_samples, n_features)
    :param n_neighbors: number of neighbors
    :return: index of n nearest neighbors
    '''
    tree = KDTree(X)
    dst, ind = tree.query(X, k = n_neighbors + 1)
    return ind[:, 1:]