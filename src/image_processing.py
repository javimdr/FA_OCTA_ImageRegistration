import math
from typing import Tuple

import cv2
import numpy as np
import scipy
from skimage import exposure, filters, morphology
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def imadjust(img: np.ndarray, contrast_limits: Tuple[int, int] = (1, 99)) -> np.ndarray:
    """Adjust image intensity values.
    (Based on: https://es.mathworks.com/help/images/ref/imadjust.html?lang=en)

    J = imadjust(I) maps the intensity values in grayscale image I to new
    values in J. By default, imadjust saturates the bottom 1% and the top 1%
    of all pixel values. This operation increases the contrast of the output
    image J.


    Args:
        img (np.ndarray): Input image (np.uint8).
        contrast_limits (Tuple[int, int], optional): Contrast limits.
            Defaults to (1, 99).

    Returns:
        np.ndarray: Normalized image.
    """
    p_inf, p_sup = np.percentile(img, contrast_limits)
    norm_img = exposure.rescale_intensity(img, in_range=(p_inf, p_sup))  # type: ignore
    return np.array(norm_img, dtype=np.uint8)


def top_hat(img: np.ndarray, disk_radius: int) -> np.ndarray:
    """
    Brightness correction applying the morphological operation Top-Hat.

    Parameters
    ----------
    img : np.ndarray
        Input image (np.uint8)
    disk_radius : int
        Radius of the disk structuring element.

    Returns
    -------
    img : np.ndarray
        Output image (np.uint8)
    """

    kernel = morphology.disk(disk_radius)
    im_bg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # type: ignore
    im_bg = cv2.blur(im_bg, (disk_radius * 2, disk_radius * 2))  # type: ignore
    out_img = cv2.subtract(img, im_bg)  # type: ignore
    return out_img


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_copy = img.copy()
    mask_copy = mask.copy()
    if mask_copy.dtype != bool:
        mask_copy = mask_copy.astype(bool)

    img_copy[~mask_copy] = 0
    return img_copy


def hessian_ridges(
    img: np.ndarray, sigma: float, contrast_limits: Tuple[int, int] = (1, 99)
):
    """
    Calculate ridges using Hessian matrix.

    Parameters
    ----------
    img : np.ndarray
        Input image (np.uint8)
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as weighting function for
        the auto-correlation matrix.
    contrast_limits
        Saturate values for normalization. Default is (1, 99). Use (0,100) to avoid this step.
    Returns
    -------
    img : np.ndarray
        Output image of ridges (np.uint8)
    """
    hessian_elems = hessian_matrix(img, sigma=sigma, mode="mirror", order="xy")  # type: ignore
    _, ridges = hessian_matrix_eigvals(hessian_elems)
    p_low, p_high = np.percentile(ridges, contrast_limits)
    ridges = exposure.rescale_intensity(ridges, in_range=(p_low, p_high))  # type: ignore
    ridges_norm = cv2.normalize(ridges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)  # type: ignore
    ridges_norm = cv2.bitwise_not(ridges_norm)  # type: ignore
    return ridges_norm


def hysteresis_dyn(
    img: np.ndarray, high_th: int, expected_percent: float
) -> Tuple[np.ndarray, float]:
    """
    Calcula el límite inferior del filtro de histéresis de forma
    iterativa, comenzando con el mismo valor que el límite superior
    y descendiendo hasta que se cumpla la condición de parada.

    La condición de parada se satisface cuando más de un tanto porcentaje
    de la imagen es activada tras aplicar el filtro; o cuando la cota inferior
    sea igual a 0.

    Parameters
    ----------
    img : np.ndarray
        Imagen (np.uint8) sobre la que se realizará el proceso.
    high_th : int
        Cota superior del filtro histéresis.
        Valor en el intervalo [0, 255].
    expected_percent : float
        Porcentaje de información mínimo esperado (condición de parada).
        Valor en el intervalo [0,1].

    Returns
    -------
    binarizated_img : np.ndarray
        Imagen binarizada por histéresis.
    low_threshold : float
        Umbral inferior calculado
    """

    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError("Image must be a ndarray of uint8.")
    if not (0 <= expected_percent <= 1):
        raise ValueError("Threshold must be a value between 0 and 1.")

    expected_percent *= img.size
    memo = dict()

    top = high_th
    bottom = 0

    while bottom <= top:
        mid = (top + bottom) // 2

        img_bin = filters.apply_hysteresis_threshold(img, mid, high_th)
        percent = np.count_nonzero(img_bin)
        memo[mid] = (img_bin, percent)

        if percent == expected_percent:
            return img_bin, mid
        elif percent > expected_percent:
            bottom = mid + 1
        else:
            top = mid - 1

    return memo[top][0], top
