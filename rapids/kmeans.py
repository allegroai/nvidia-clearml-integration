from clearml import Task

task = Task.init(project_name="Rapids", task_name="kmeans")


n_samples = 10**6
n_features = 2
n_clusters = 5
random_state = 23

"""## Generate Data"""

import cudf
from cuml.datasets import make_blobs

device_data, device_labels = make_blobs(n_samples=n_samples,
                                        n_features=n_features,
                                        centers=n_clusters,
                                        random_state=random_state,
                                        cluster_std=0.1)

# Create cuDF DataFrame and Series from CuPy ndarray.
device_data = cudf.DataFrame(device_data)
device_labels = cudf.Series(device_labels)

# Copy dataset from GPU memory to host memory.
# This is done to later compare CPU and GPU results.
host_data = device_data.to_pandas()
host_labels = device_labels.to_pandas()

"""## Scikit-learn model

### Fit
"""

from sklearn.cluster import KMeans

kmeans_skl = KMeans(init="k-means++",
                    n_clusters=n_clusters,
                    random_state=random_state)
kmeans_skl.fit(host_data)

"""## cuML Model

### Fit
"""

from cuml.cluster import KMeans

kmeans_cuml = KMeans(init="k-means||",
                     n_clusters=n_clusters,
                     random_state=random_state)
kmeans_cuml.fit(device_data)

"""## Visualize Centroids

Scikit-learn's k-means implementation uses the `k-means++` initialization strategy while cuML's k-means uses `k-means||`. As a result, the exact centroids found may not be exact as the std deviation of the points around the centroids in `make_blobs` is increased.

*Note*: Visualizing the centroids will only work when `n_features = 2` 
"""

import cupy
import matplotlib.pyplot as plt

#fig = plt.figure(figsize=(16, 10))
#plt.scatter(host_data.iloc[:, 0], host_data.iloc[:, 1], c=host_labels, s=50, cmap='viridis')
#plt.show()

# plot the sklearn kmeans centers with blue filled circles
centers_skl = kmeans_skl.cluster_centers_
plt.scatter(centers_skl[:,0], centers_skl[:,1], c='blue', s=100, alpha=.5)
plt.show()


# plot the cuml kmeans centers with red circle outlines
centers_cuml = kmeans_cuml.cluster_centers_
plt.scatter(cupy.asnumpy(centers_cuml[0].values),
            cupy.asnumpy(centers_cuml[1].values),
            facecolors = 'none', edgecolors='red', s=100)

plt.title('cuml and sklearn kmeans clustering')
plt.show()
"""## Compare Results"""

from cuml.metrics import adjusted_rand_score

score_cuml = adjusted_rand_score(host_labels, kmeans_cuml.labels_)

from sklearn.metrics import adjusted_rand_score

score_skl = adjusted_rand_score(host_labels, kmeans_skl.labels_)

threshold = 1e-4

passed = (score_cuml - score_skl) < threshold
print('compare kmeans: cuml vs sklearn labels_ are ' + ('equal' if passed else 'NOT equal'))

fig = plt.figure(figsize=(16, 10))
plt.scatter(host_data.iloc[:, 0], host_data.iloc[:, 1], c=host_labels, s=50, cmap='viridis')
plt.show()
