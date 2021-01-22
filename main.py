from Kmeans import Kmeans
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt 
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
print(X.shape)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

#TO SELECT THE KMEANS++ initialization add field type = "++"
kmeans = Kmeans(n_cluster=4, max_iter=100, type="++")
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.centroids
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()