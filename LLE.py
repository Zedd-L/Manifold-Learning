import numpy as np
import matplotlib.pyplot as plt
from helper import gen_data, Knn

#------------------LLE------------------#
# 1. find knn x_1, x_2, ..., x_k        #
# 2. calculate w1, w2, ..., wk          #
#    w1*x_1 + w2*x_2 + ... + wk*x_k ~= x#
# 3. M = (I-W)'(I-W)                    #
#    find bottom d+1 eigenvectors of M  #
#---------------------------------------#

def LinearRefactor(X, n_neighbors, gamma = None):
    '''
    : param X: dataset shape as (n_samples, n_features)
    : param n_neighbors: number of neighbors
    : param gamma: ?
    : return: weights of n neighbors (n_samples, n_neighbors)
    '''
    N = Knn(X, n_neighbors=n_neighbors)
    n_samples, n_features = X.shape
    w = np.zeros((n_samples, n_neighbors))
    I = np.ones((n_neighbors, 1))
    # wtf?
    tol = 0 # tol越小，重构越接近
    if gamma:
        if n_neighbors > n_features:
            tol = gamma
    for i in range(n_samples):
        Xi = np.tile(X[i, :], (n_neighbors, 1))
        Ni = X[N[i]]
        Zi = (Xi - Ni) @ (Xi - Ni).T
        Zi = Zi + np.eye(n_neighbors) * np.trace(Zi) * tol
        Zi_inv = np.linalg.pinv(Zi)
        w_i = (Zi_inv @ I) / (I.T @ Zi_inv @ I)
        w[i] = w_i[:, 0]
    return N, w

def DimReduce(w, N, n_component):
    '''
    : param w: weights of n neighbors (n_samples, n_neighbors)
    : param N: n neighbors (n_samples, n_neighbors)
    : n_component: target dims
    : return: low-dimensional(n_component) reconstruction of X (n_samples, n_component)
    '''
    n_samples, n_neighbors = w.shape
    W = np.zeros((n_samples, n_samples))
    I = np.eye(n_samples)
    for i in range(n_samples):
        neighbor = N[i]
        for j in range(n_neighbors):
            W[i, neighbor[j]] = w[i, j]
    M = (I - W).T @ (I - W)
    L, V = np.linalg.eig(M)
    _ = np.argsort(L)[1 : n_component+1]
    Y = V[:, _]
    return Y

def LLE(X, n_neighbors, n_component, gamma):
    N, W_x = LinearRefactor(X, n_neighbors, gamma)
    Y = DimReduce(W_x, N, n_component)
    return Y

if __name__ == '__main__':
    n_ponits = 1000
    n_neighbors = 10
    X, color = gen_data(n_points=n_ponits)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(251, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    _gamma = [0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    for i, gamma in enumerate(_gamma):
        Y = LLE(X, 10, 2, gamma)
        bx = fig.add_subplot(2, 4, i + 2)
        bx.set_title('gamma = {}'.format(str(gamma)))
        bx.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.show()