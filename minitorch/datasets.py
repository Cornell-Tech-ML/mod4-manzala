import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate random points.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A list of N tuples where each tuple represents a 2D point.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """Graph data structure to hold points and labels."""

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a simple graph dataset where points are classified
    based on whether their x-coordinate is less than 0.5.

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        A graph dataset with points and binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a diagonal graph dataset where points are classified
    based on whether the sum of their coordinates is less than 0.5.

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        A graph dataset with points and binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a graph dataset where points are classified
    based on whether their x-coordinate is in the range (0.2, 0.8).

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        A graph dataset with points and binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate an XOR-like graph dataset where points are classified
    based on an XOR operation on their coordinates.

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        A graph dataset with points and binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a circular graph dataset where points are classified
    based on whether their distance from the center exceeds a threshold.

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        A graph dataset with points and binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a spiral graph dataset where points form two spirals
    and are classified based on which spiral they belong to.

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        A graph dataset with points and binary labels.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
