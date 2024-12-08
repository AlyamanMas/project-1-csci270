import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# parameters (can be used to customize the program):
cluster_samples = 500
np.random.seed(42)
# first two elements of the tuples are cluster center's x and y,
# second point is spread
cluster_xyspread_list = [
    (2, 2, 1.5),
    (10, 10, 2.5),
    (1, 12, 2)
]

# generate the cluster points
clusters_list = [
    np.random.normal([cluster_data[0], cluster_data[1]], cluster_data[2], size=(cluster_samples, 2))
    for cluster_data in cluster_xyspread_list
]
# combine the three clusters into one big array
X = np.vstack([cluster for cluster in clusters_list])

# train kmeans instance and predict clusters with it
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
y_pred = kmeans.predict(X)

# start plotting
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='Clusters', alpha=0.5)
cluster_centers = kmeans.cluster_centers_
# plot the centroids
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers',
            linewidths=3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(
    'Plotting of the artificial data with their clusters and\n cluster centers as determined by our kmeans algorithm')
plt.legend()
plt.show()

# make into pd dataframe to nice print
cluster_centers_df = pd.DataFrame(cluster_centers, columns=['x', 'y'])
print(cluster_centers_df)