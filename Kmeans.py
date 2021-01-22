import numpy as np
from numpy.linalg import norm

class Kmeans:
    '''Implementation of the algorith'''

    def __init__(self,n_cluster, max_iter=100, random_state=123, type = "Standart"):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = 0
        self.type = type

    def initializ_centroids_plusplus(self, X):
        #def plus_plus(ds, k, random_state=42):
        np.random.seed(self.random_state)
        centroids = [X[0]]

        for i in range(1, self.n_cluster):
            dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in X])
            probs = dist_sq/dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
            
            centroids.append(X[i])
        return np.array(centroids)
    
    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_cluster]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_cluster, X.shape[1]))
        for k in range(self.n_cluster):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_cluster))
        for k in range(self.n_cluster):
            row_norm = norm(X - centroids[k,:], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance
    
    def find_closet_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_cluster):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X):
        if self.type == "Standart": self.centroids = self.initializ_centroids(X)
        else: self.centroids = self.initializ_centroids_plusplus(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closet_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)

    def predict(self, X):
        old_centroids = self.centroids
        distance = self.compute_distance(X, old_centroids)
        return self.find_closet_cluster(distance)