import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from helper import gen_data

if __name__ == '__main__':
    X, color = gen_data(3000)   
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

    n_neighbors = 10 #用10个近邻表示
    n_components = 2 #降到2维

    LLE = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
    ISOmap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    # out = LLE.fit_transform(X)
    out = ISOmap.fit_transform(X)
    bx = fig.add_subplot(1, 2, 2)
    bx.scatter(out[:, 0], out[:, 1], c=color, cmap=plt.cm.Spectral)

    plt.show()