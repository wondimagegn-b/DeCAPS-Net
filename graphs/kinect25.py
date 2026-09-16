"""Kinect v2 25-joint skeleton graph used for the GFBMD dataset.

Joint order follows the standard NTU RGB+D convention (this is the order
produced by our preprocessing pipeline, preprocessing/gfbmd/02_xlsx_to_skeleton.py):

    0  SpineBase      1  SpineMid       2  Neck          3  Head
    4  ShoulderLeft   5  ElbowLeft     6  WristLeft     7  HandLeft
    8  ShoulderRight  9  ElbowRight    10 WristRight    11 HandRight
    12 HipLeft        13 KneeLeft      14 AnkleLeft     15 FootLeft
    16 HipRight       17 KneeRight     18 AnkleRight    19 FootRight
    20 SpineShoulder  21 HandTipLeft   22 ThumbLeft     23 HandTipRight
    24 ThumbRight

The adjacency uses the "spatial" labeling of Yan et al. (ST-GCN, AAAI 2018):
three partitions (identity / inward / outward), each column-normalized.
Graph.A has shape (3, 25, 25).
"""

import numpy as np

NUM_NODE = 25

SELF_LINK = [(i, i) for i in range(NUM_NODE)]

# Bone connections in 1-based indexing (parent -> child, toward the body center).
INWARD_1BASED = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),
]
INWARD = [(i - 1, j - 1) for (i, j) in INWARD_1BASED]
OUTWARD = [(j, i) for (i, j) in INWARD]


def _edge2mat(links, num_node):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in links:
        A[j, i] = 1
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
    """GFBMD skeleton graph. `A` is (3, 25, 25): identity, inward, outward."""

    def __init__(self, labeling_mode="spatial"):
        if labeling_mode != "spatial":
            raise ValueError(f"Unsupported labeling_mode: {labeling_mode}")
        self.num_node = NUM_NODE
        self.self_link = SELF_LINK
        self.inward = INWARD
        self.outward = OUTWARD
        self.neighbor = INWARD + OUTWARD
        self.A = np.stack(
            (
                _edge2mat(SELF_LINK, NUM_NODE),
                _normalize_digraph(_edge2mat(INWARD, NUM_NODE)),
                _normalize_digraph(_edge2mat(OUTWARD, NUM_NODE)),
            )
        ).astype(np.float32)
