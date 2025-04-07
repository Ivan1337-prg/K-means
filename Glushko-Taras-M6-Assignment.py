import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=488)
raw_features = dataset.data.features
raw_targets = dataset.data.targets

numeric_data = raw_features.drop(columns=['status_type', 'status_published'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

num_clusters = 3
model = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
model.fit(scaled_data)

cluster_centers = model.cluster_centers_
print(cluster_centers)

data_2d = scaled_data[:, 1:3]
centers_2d = cluster_centers[:, 1:3]

grid_resolution = 0.1
x_range = np.arange(data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1, grid_resolution)
y_range = np.arange(data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1, grid_resolution)
xx, yy = np.meshgrid(x_range, y_range)
grid_points = np.c_[xx.ravel(), yy.ravel()]

model_2d = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
model_2d.fit(data_2d)
predicted_labels = model_2d.predict(grid_points)
predicted_labels = predicted_labels.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.clf()
plt.imshow(predicted_labels, interpolation='nearest',
           extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')

plt.scatter(data_2d[:, 0], data_2d[:, 1], c='black', edgecolor='k', s=20)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
            marker='x', s=200, linewidths=3, color='white')
plt.title("K-Means Clustering on Facebook Seller Posts (2D Projection)")
plt.xlabel("Feature 2 (scaled)")
plt.ylabel("Feature 3 (scaled)")
plt.show()
