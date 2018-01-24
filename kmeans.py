import numpy as np
import matplotlib.pyplot as plt


def distance(X, M):
    # calculate the euclidean distance between numpy arrays X and M
    (m, n) = X.shape
    d = np.zeros(m)
    for i in range(m):
        for j in range(n):
            d[i] = d[i] + np.square(X[i, j] - M[i, j])
    d = list(d)
    return d


def findClosestCentres(X, M):
    # finds the centre in M closest to each point in X
    (k, n) = M.shape  # k is number of centres
    (m, n) = X.shape  # m is number of data points
    C = list()
    for j in range(k):
        C.append(list())
    for j in range(m):
        d = []
        for i in range(k):
            dis = np.linalg.norm(X[j] - M[i])
            d.append(dis)
        a = np.argmin(d)
        C[a].append(j)
    return C


def updateCentres(X, C):
    # updates the centres to be the average of the points closest to it.
    k = len(C)  # k is number of centres
    (m, n) = X.shape  # n is number of features
    M = np.zeros((k, n))
    for i in range(k):
        c = 0
        for j in C[i]:
            M[i] = np.add(M[i], X[j])
            c = c + 1
        M[i] = M[i] / c
    return M


def plotData(X, C, M):
    # plot the data, coloured according to which centre is closest. and also plot the centres themselves
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X[C[0], 0], X[C[0], 1], c='c', marker='o')
    ax.scatter(X[C[1], 0], X[C[1], 1], c='b', marker='o')
    ax.scatter(X[C[2], 0], X[C[2], 1], c='g', marker='o')
    # plot centres
    ax.scatter(M[:, 0], M[:, 1], c='r', marker='x', s=100, label='centres')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    fig.savefig('graph.png')


def main():
    print('testing the distance function ...')
    print(distance(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [1, 2]])))

    print('testing the findClosestCentres function ...')
    print(findClosestCentres(np.array([[1, 2], [3, 4], [0.9, 1.8]]), np.array([[1, 2], [2.5, 3.5]])))

    print('testing the updateCentres function ...')
    print(updateCentres(np.array([[1, 2], [3, 4], [0.9, 1.8]]), [[0, 2], [1]]))

    print('loading test data ...')
    X = np.loadtxt('data.txt')
    [m, n] = X.shape
    iters = 10
    k = 3
    print('initialising centres ...')
    init_points = np.random.choice(m, k, replace=False)
    M = X[init_points, :]  # initialise centres randomly
    print('running k-means algorithm ...')
    for i in range(iters):
        C = findClosestCentres(X, M)
        M = updateCentres(X, C)
    print('plotting output')

    plotData(X, C, M)


if __name__ == '__main__':
    main()