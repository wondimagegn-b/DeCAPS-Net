"""COCO-17 skeleton graph used for the ASDPose dataset.

Joint order (0-based, standard COCO keypoint layout):

    0  nose          1  left_eye     2  right_eye    3  left_ear
    4  right_ear     5  left_shoulder 6 right_shoulder
    7  left_elbow    8  right_elbow  9  left_wrist   10 right_wrist
    11 left_hip      12 right_hip    13 left_knee    14 right_knee
    15 left_ankle    16 right_ankle

The adjacency is a single column-normalized undirected partition with
self-loops, shape (1, 17, 17). The skeleton model internally replicates
this single partition into three identical partitions to match its
3-subset convolution kernels.
"""

import numpy as np

NUM_NODE = 17

# Undirected bone connections (0-based).
COCO17_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def _edge2mat(edges, num_node):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edges:
        A[j, i] = 1
        A[i, j] = 1  # undirected
    return A


def _normalize_digraph(A):
    """Column-wise normalization (each column sums to 1 where possible)."""
    col_sum = np.sum(A, axis=0)
    Dn = np.zeros_like(A)
    for i in range(A.shape[0]):
        if col_sum[i] > 0:
            Dn[i, i] = col_sum[i] ** (-1)
    return np.dot(A, Dn)


class Graph:
    """ASDPose skeleton graph. `A` is (1, 17, 17)."""

    def __init__(self, labeling_mode="spatial"):
        if labeling_mode != "spatial":
            raise ValueError(f"Unsupported labeling_mode: {labeling_mode}")
        self.num_node = NUM_NODE
        self.self_link = [(i, i) for i in range(NUM_NODE)]
        self.neighbor = COCO17_EDGES
        A = _edge2mat(self.self_link + self.neighbor, NUM_NODE)
        self.A = np.stack([_normalize_digraph(A)], axis=0).astype(np.float32)
