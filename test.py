
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

A = np.array(
[[0, 1, 0, 0, 1],
[0, 0, 1, 1, 1],
[1, 1, 0, 1, 0]])

dist_out = 1-pairwise_distances(A, metric="cosine")
print(dist_out)