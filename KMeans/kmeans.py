import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k: int = 3, max_iter: int = 100) -> None:
        self.centroids = None
        self.k = k
        self.max_iter = max_iter

    def get_distance(self, point: np.ndarray, centroids: np.ndarray) -> int:
        return np.linalg.norm(point - centroids)

    def get_closest_centroid(self, point: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        closet_cluster = None
        min_distance = float('inf')
        for cluster, centroid in enumerate(centroids):
            distance = self.get_distance(point=point, centroids=centroid)
            if distance < min_distance:
                min_distance = distance
                closet_cluster = cluster
        return closet_cluster

    def recalculate_centroid(self, ) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray) -> np.ndarray:
        # initialize k centroids random data points
        # getting random k index from 0 - len(data)
        centroids_idx = np.random.randint(low=0, high=X.shape[0], size=self.k)

        # fetch k centroids from datapoints
        centroids = X[centroids_idx]

        # structure to store index of point  -> cluster c
        assigned = np.array([0] * X.shape[0], dtype=np.uint)

        while self.max_iter > 0:
            self.max_iter = self.max_iter - 1

            # iterate over each point to assign to closest cluster
            for point in range(X.shape[0]):
                # get closest cluster/centroid
                closest_cluster = self.get_closest_centroid(
                    X[point], centroids)

                # assign point to closest cluster
                assigned[point] = closest_cluster

            # recenter cluster centroids
            for cluster, _ in enumerate(centroids):
                # replacing the old centroids of cluster with mean of cluster data points
                centroids[cluster] = X[assigned == cluster].mean(axis=0)
        # coverged cluster centroids
        self.centroids = centroids
        return assigned


if __name__ == '__main__':
    data = pd.read_csv('Mall_Customers.csv')
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

    kmeans = KMeans(k=5, max_iter=100)
    assigned = kmeans.fit(X)
    clormap = np.array(['r', 'g', 'b', 'orange', 'yellow'])
    plt.scatter(X[:, 0], X[:, 1], c=clormap[assigned])
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=clormap[[
                0, 1, 2, 3, 4]], edgecolors='black', alpha=0.5, s=200)
    plt.show()
