from typing import Sequence

import cv2
import numpy as np
from scipy.signal import fftconvolve


def RMSE(x, y):
    return np.sqrt(np.mean(np.subtract(x, y) ** 2))


def MED(points_a, points_b):
    """ Mean of Euclidean distances """
    distances = [euclidean_dist(a, b) for a, b in zip(points_a, points_b)]
    return np.mean(distances)


def euclidean_dist(p: Sequence, q: Sequence) -> float:
    """Compute euclidean distance between the point P and Q.

    Parameters
    ----------
    p : Sequence
        First point
    q : Sequence
        Second point

    Returns
    -------
    float
        Euclidean distance
    """
    return np.linalg.norm(np.subtract(p, q))


def ccorr_normed(x, y):
    xf = x.astype(float)
    yf = y.astype(float)

    num = (xf * yf).sum()
    den = np.sqrt((xf ** 2).sum() * (yf ** 2).sum())
    if den == 0:
        return 0

    return num / den


def zncc(x, y, precission=10):
    """
    Zero-normalized cross-correlation (ZNCC)
    https://es.mathworks.com/help/images/ref/normxcorr2.html
    """
    assert x.shape == y.shape

    xf = x.astype(float)
    yf = y.astype(float)

    x_subs_mean = xf - np.mean(xf)
    y_subs_mean = yf - np.mean(yf)
    # x_std = np.std(xf)
    # y_std = np.std(yf)

    num = np.sum(x_subs_mean * y_subs_mean)
    den = (np.sum(x_subs_mean ** 2) * np.sum(y_subs_mean ** 2)) ** 0.5

    return np.round(num / den, precission) if den != 0 else 0


def normxcorr2(image, template, mode="valid", precission=10):
    """
    Fast Template Matching usign ZNCC function.
    https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py
    """

    # If this happens, it is probably a mistake
    if (
        np.ndim(template) > np.ndim(image)
        or len(
            [i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]
        )
        > 0
    ):
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - np.square(
        fftconvolve(image, a1, mode=mode)
    ) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    den = np.sqrt(image * template)

    if den.size == 1 and den == 0:
        return 0

    with np.errstate(divide="ignore", invalid="ignore"):
        out /= den

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    out = out.round(precission)

    if den.size == 1:
        return float(out)
    return out


def impair_ccorr(src_image, dst_image, transform, ccorr_function="ZNCC"):
    """

    Parameters
    ----------
    src_image
    dst_image
    transform
    mask
    ccorr_function: {CV, ZNCC}

    Returns
    -------

    """
    H, W = dst_image.shape[:2]
    mask = np.ones(src_image.shape[:2], np.uint8)

    src_transform = cv2.warpPerspective(src_image, transform, (W, H))
    mask_transform = cv2.warpPerspective(mask, transform, (W, H))
    indexes = mask_transform > 0

    if ccorr_function == "CV":
        ccorr = ccorr_normed(src_transform[indexes], dst_image[indexes])
    elif ccorr_function == "ZNCC":
        ccorr = zncc(src_transform[indexes], dst_image[indexes])
    else:
        raise ValueError("Invalid correlation funcion. Use 'CV' or 'ZNCC'")

    return ccorr

