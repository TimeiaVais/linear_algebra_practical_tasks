import numpy as np
from scipy.ndimage import affine_transform as scipy_affine_transform

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    return -x


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    if x.ndim == 1:
        return np.array(list(x)[::-1])
    reversed_mat = [row[::-1] for row in list(x)[::-1]]
    return np.array(reversed_mat)


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    if x.ndim == 1:
        x = x.reshape(1, 2)

    assert x.shape[1] == 2, "Affine transform supports only 2D points"

    alpha = np.deg2rad(alpha_deg)

     M = np.array([
        [scale[0]*np.cos(alpha) + shear[0]*np.sin(alpha),
         -np.sin(alpha)*scale[1] + shear[0]*np.cos(alpha)],
        [np.sin(alpha)*scale[0] + shear[1]*np.sin(alpha),
         np.cos(alpha)*scale[1] + shear[1]*np.cos(alpha)]
    ])

     transformed = []

     for point in x:
        new_point = scipy_affine_transform(
            input=point,          
            matrix=M,             
            offset=translate,     
            output_shape=(2,),    
            order=1               
        )
        transformed.append(new_point)

    return np.array(transformed)
