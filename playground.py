import numpy as np
import math


def multi_gaussian(x: np.ndarray, miu: np.ndarray, sigma: np.ndarray):
    """

    :param x: (N, 1)
    :param miu: (N, 1)
    :param sigma: (N, N)
    :return:
    """
    dim = len(x.shape[0])
    inverse_sigma = np.linalg.inv(sigma)
    comp = np.matmul(np.matmul((x - miu).T, inverse_sigma), (x - miu))  # (1)
    k_of_2pi = (2 * math.pi) ** dim
    det_sigma = np.linalg.det(sigma)

    result = ((k_of_2pi * det_sigma) ** -0.5) * np.exp(comp)

    return result


