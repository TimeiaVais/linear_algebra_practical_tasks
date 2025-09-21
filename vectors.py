from typing import Sequence
import numpy as np
from scipy import sparse
from numpy.linalg import norm as np_norm, solve


def get_vector(dim: int) -> np.ndarray:
    return np.random.rand(dim, 1)


def get_sparse_vector(dim: int, density: float = 0.1) -> sparse.coo_matrix:
    dense_vec = sparse.random(dim, 1, density=density, format="coo", dtype=float)
    return dense_vec


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.add(x, y)


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    return a * x


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    if len(vectors) != len(coeffs):
        raise ValueError("Number of vectors and coefficients must match.")
    result = np.zeros_like(vectors[0])
    for v, c in zip(vectors, coeffs):
        result += c * v
    return result


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x.ravel(), y.ravel()))


def norm(x: np.ndarray, order: int | float = 2) -> float:
    return float(np_norm(x, ord=order))


def distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np_norm(x - y))


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    dot = dot_product(x, y)
    norms = norm(x, 2) * norm(y, 2)
    if norms == 0:
        raise ValueError("Zero vector provided.")
    cos_theta = np.clip(dot / norms, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return float(np.degrees(theta_rad))


def is_orthogonal(x: np.ndarray, y: np.ndarray, tol: float = 1e-10) -> bool:
    return abs(dot_product(x, y)) < tol


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return solve(a, b)


# Example usage
if __name__ == "__main__":
    v1 = get_vector(3)
    v2 = get_vector(3)

    print("Vector v1:\n", v1)
    print("Vector v2:\n", v2)
    print("Addition:\n", add(v1, v2))
    print("Scalar multiplication:\n", scalar_multiplication(v1, 2))
    print("Dot product:", dot_product(v1, v2))
    print("Norm v1:", norm(v1, 2))
    print("Distance:", distance(v1, v2))
    print("Cosine angle (deg):", cos_between_vectors(v1, v2))
    print("Orthogonal?", is_orthogonal(v1, v2))

    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([8, 13], dtype=float)
    print("Solution of Ax=b:", solves_linear_systems(A, b))
