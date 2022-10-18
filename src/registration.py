import time
from math import radians
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.transform import AffineTransform

import evolutionary
import image_processing
import metrics
import plots


# Preprocessing functions
def denoise_octa(
    img: np.ndarray, median_ks: int = 3, tophat_radius: int = 15
) -> np.ndarray:
    """Applies the denoise process to OCTA images. This process consists of an
    adjustment of the intensity values of the original image, followed by a
    median filter and, finally, the top hat operator is applied.

    Args:
        img (np.ndarray): original OCTA image.
        median_ks (int, optional): kernel size for median filter.
            Defaults to 3.
        tophat_radius (int, optional): radius of the structuring element used
            for the top-hat operator. Defaults to 15.

    Returns:
        np.ndarray: OCTA denoised image.
    """
    normalized = image_processing.imadjust(img)
    denoised = cv2.medianBlur(normalized, median_ks)  # type: ignore
    denoised = image_processing.top_hat(denoised, tophat_radius)
    return denoised


def denoise_agf(
    img: np.ndarray, gaussian_ks: Tuple[int, int] = (11, 11), tophat_radius: int = 31
) -> np.ndarray:
    """Applies the denoise process to FA images. This process consists of an
    adjustment of the intensity values of the original image, followed by a
    gaussian blur filter and, finally, the top hat operator is applied.

    Args:
        img (np.array): original FA image.
        gaussian_ks (Tuple[int, int], optional): kernel size for gaussian blur
            filter. Defaults to (11, 11).
        tophat_radius (int, optional): radius of the structuring element used
            for the top-hat operator. Defaults to 31.

    Returns:
        np.ndarray: FA denoised image.
    """
    denoised = image_processing.imadjust(img)
    denoised = cv2.GaussianBlur(denoised, gaussian_ks, 0)  # type: ignore
    denoised = image_processing.top_hat(denoised, tophat_radius)
    return denoised


def detect_vessels(
    image: np.ndarray, sigma: float, hyst_th_high: int, hyst_th_percent: float
) -> np.ndarray:
    """Stage of detecting the blood vessels of an image.

    Args:
        image (np.ndarray): input image.
        sigma (float): Standard deviation used for the Gaussian kernel, which
            is used as weighting function for the auto-correlation matrix.
        hyst_th_high (int): Hysteresis filter upper bound.
        hyst_th_percent (float): Minimum information percentage expected (stop
            condition).

    Returns:
        np.ndarray: Mask showing the vessels detected in the input image.
    """
    ridges = image_processing.hessian_ridges(image, sigma)
    mask, _ = image_processing.hysteresis_dyn(ridges, hyst_th_high, hyst_th_percent)
    return mask.astype(bool)


def resize(img: np.ndarray, scale_factor: float) -> np.ndarray:
    """Resizes an image using a specific scale factor.

    Args:
        img (np.ndarray): input image.
        scale_factor (float): scale factor along both axes.

    Returns:
        np.ndarray: resized image.
    """
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor)  # type: ignore


def preprocess_agf(
    agf_image: np.ndarray,
    gaussian_ks: Tuple[int, int] = (19, 19),
    tophat_radius: int = 31,
    sigma: float = 3.5,
    hyst_th_high: int = 200,
    hyst_th_percent: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete preprocessing stage for Fluorescein Angiography images.
    Returns the denoised FA image, the blood vessel mask, and the final
    segmented image.

    Args:
        agf_image (np.ndarray): original FA image.
        gaussian_ks (Tuple[int,int], optional): kernel size for gaussian blur
            filter. Defaults to (19, 19).
        tophat_radius (int, optional): radius of the structuring element used
            for the top-hat operator. Defaults to 31.
        sigma (float, optional): Standard deviation used for the Gaussian
            kernel, which is used as weighting function for the
            auto-correlation matrix. Defaults to 3.5.
        hyst_th_high (int, optional): Hysteresis filter upper bound.
            Defaults to 200.
        hyst_th_percent (float, optional): Maximum information percentage
        expected (stop condition). Defaults to 0.20.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: denoised FA image, the
            blood vessel mask, and the final segmented image.
    """
    agf_pp = denoise_agf(agf_image, gaussian_ks, tophat_radius)
    agf_mask = detect_vessels(agf_pp, sigma, hyst_th_high, hyst_th_percent)
    agf_seg = image_processing.apply_mask(agf_pp, agf_mask)
    return agf_pp, agf_mask, agf_seg


def preprocess_octa(
    octa_image: np.ndarray,
    median_ks: int = 3,
    tophat_radius: int = 15,
    sigma: float = 4.0,
    hyst_th_high: int = 220,
    hyst_th_percent: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete preprocessing stage for OCTA images.
    Returns the denoised OCTA image, the blood vessel mask, and the final
    segmented image.

    Args:
        octa_image (np.ndarray): original OCTA image.
        median_ks (int, optional): kernel size for median filter. Defaults to 3.
        tophat_radius (int, optional): radius of the structuring element used
            for the top-hat operator. Defaults to 15.
        sigma (float, optional): Standard deviation used for the Gaussian
            kernel, which is used as weighting function for the
            auto-correlation matrix. Defaults to 4.0.
        hyst_th_high (int, optional): Hysteresis filter upper bound.
            Defaults to 220.
        hyst_th_percent (float, optional): Maximum information percentage
            expected (stop condition). Defaults to 0.15.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: denoised OCTA image, the
            blood vessel mask, and the final segmented image.
    """
    octa_pp = denoise_octa(octa_image, median_ks, tophat_radius)
    octa_mask = detect_vessels(octa_pp, sigma, hyst_th_high, hyst_th_percent)
    octa_seg = image_processing.apply_mask(octa_pp, octa_mask)
    return octa_pp, octa_mask, octa_seg


# First aproximation: Template Matching process
def template_matching(
    fixed: np.ndarray, template: np.ndarray, corr_func: str = "ZNCC"
) -> Tuple[float, Tuple[int, int]]:
    """Performs the template matching process using the function selected
    correlation.

    Args:
        fixed (np.ndarray): fixed image.
        template (np.ndarray): template image.
        corr_func (str, optional): Correlation function. Avaliable functions:
            'CV' (cv2.TM_CCORR_NORMED) and 'ZNCC' (Zero Normalized Cross
            Correlation). Defaults to 'ZNCC'.

    Raises:
        ValueError: Invalid correlation function selected.

    Returns:
        Tuple[float, Tuple[int, int]]: Maximum correlation value and its
        position (top-left corner) in the form (x, y).
    """
    _CCORR_FUNCTIONS = ["CV", "ZNCC"]

    if corr_func == "CV":
        ccorr_matrix = cv2.matchTemplate(fixed, template, cv2.TM_CCORR_NORMED)  # type: ignore
    elif corr_func == "ZNCC":
        ccorr_matrix = metrics.normxcorr2(fixed, template, "valid")
    else:
        raise ValueError(
            f"Invalid correlation function. Use one of this: {_CCORR_FUNCTIONS}"
        )

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ccorr_matrix)  # type: ignore
    return max_val, max_loc


# Fine tuning: Differential Evolution
def create_individual(
    translation: Tuple[int, int] = (0, 0),
    scale: Tuple[float, float] = (1, 1),
    rotation: float = 0,
    shear: float = 0,
    deg2rad: bool = True,
) -> np.ndarray:
    """Create an individual (vector of 6 values) using the basic values of
    affine transformation.

    Args:
        translation (Tuple[int, int], optional): Traslation (x,y) of the
            transformation. Defaults to (0, 0).
        scale (Tuple[float, float], optional): scale (x,y) of the
            transformation. Defaults to (1, 1).
        rotation (float, optional): rotation of the transformation. Defaults
            to 0.
        shear (float, optional): shear of the transformation. Defaults to 0.
        deg2rad (bool, optional): If true, converts the rotation and shear
            angles from degrees to radians. Defaults to True.

    Returns:
        np.ndarray: Individual in format [tx, ty, sx, sy, rot, shear]
    """
    if deg2rad:
        rotation = radians(rotation)
        shear = radians(shear)

    return np.hstack((translation, scale, rotation, shear)).astype(float)


def individual_to_affine(individual: Sequence[float]) -> np.ndarray:
    """Convert an individual to its corresponding affine transformation
    matrix (3x3 matrix).

    Args:
        individual (Sequence[float]): vector of 6 values.

    Returns:
        np.ndarray: affine transformation matrix.
    """
    t = AffineTransform(
        translation=individual[:2],
        scale=individual[2:4],
        rotation=individual[4],
        shear=individual[5],
    )
    return t.params


def affine_to_individual(affine_matrix: np.ndarray) -> np.ndarray:
    """Convert an affine transformation matrix (3x3 matrix) to its
    corresponding individual vector (vector of 6 values).

    Args:
        affine_matrix (np.ndarray): affine transformation matrix.

    Returns:
        np.ndarray: individual vector.
    """
    t = AffineTransform(matrix=affine_matrix)
    idv = np.hstack((t.translation, t.scale, t.rotation, t.shear)).astype(float)
    return idv


def affine_bounds(
    base_loc: Tuple[int, int],
    trans_bound: int = 30,
    scale_bound: float = 0.05,
    rot_bound: float = 15.0,
    shear_bound: float = 5.0,
) -> np.ndarray:
    """Generate the bounds of the individuals for optimization algortihms.

    Args:
        base_loc (Tuple[int, int]): Translation bounds are applied around
            this location.
        trans_bound (int, optional): Freedom for translation. Defaults to 30.
        scale_bound (float, optional): Freedom for scale. Defaults to 0.05.
        rot_bound (float, optional): Freedom for rotation in degrees.
            Defaults to 15.0.
        shear_bound (float, optional): Freedom for shear in degrees.
            Defaults to 5.0.

    Returns:
        np.ndarray: lower and upper bounds for each freedom degree:
        (tx, ty, sx, sy, rot, shear)
    """
    x, y = base_loc
    sx, sy = (1, 1)  # base_scale

    translation_x_bounds = (x - trans_bound, x + trans_bound)
    translation_y_bounds = (y - trans_bound, y + trans_bound)
    scale_x_bounds = (sx - scale_bound, sx + scale_bound)
    scale_y_bounds = (sy - scale_bound, sy + scale_bound)
    rotation_bounds = (radians(-rot_bound), radians(rot_bound))
    shear_bounds = (radians(-shear_bound), radians(shear_bound))

    return np.array(
        [
            translation_x_bounds,
            translation_y_bounds,
            scale_x_bounds,
            scale_y_bounds,
            rotation_bounds,
            shear_bounds,
        ]
    ).astype(float)


def objective_function(
    individual: np.ndarray, src_image: np.ndarray, dst_image: np.ndarray
) -> float:
    """Objective function used in optimization algorithms to solve the image
    registration problem by maximizing the correlation value.

    Args:
        individual (np.ndarray): affine transformation expressed as a vector
            of 6 values (tx, ty, sx, sy, rotation, shear).
        src_image (np.ndarray): moving image.
        dst_image (np.ndarray): fixed image.

    Returns:
        float: correlation value (ZNCC).
    """
    # T = individual_to_affine(individual)
    T = AffineTransform(
        translation=individual[:2],
        scale=individual[2:4],
        rotation=individual[4],
        shear=individual[5],
    ).params

    # Realize rotation from center of image, not top-left corner
    h2, w2 = np.array(src_image.shape[:2]) / 2
    comp_tf = (
        AffineTransform(translation=(w2, h2)).params
        @ T
        @ AffineTransform(translation=(-w2, -h2)).params
    )

    H, W = dst_image.shape[:2]
    mask = np.ones(src_image.shape[:2], np.uint8)
    src_transform = cv2.warpPerspective(src_image, comp_tf, (W, H))  # type: ignore
    mask_transform = cv2.warpPerspective(mask, comp_tf, (W, H))  # type: ignore

    idx = mask_transform > 0
    ccorr = metrics.zncc(src_transform[idx], dst_image[idx])
    return ccorr


def warp_image(
    src_image: np.ndarray,
    dst_size: Tuple[int, int],
    affine_matrix: np.ndarray,
    src_scale: float = 1,
    dst_scale: float = 1,
) -> np.ndarray:
    """Transforms an image by applying an affine transformation.

    Args:
        src_image (np.ndarray): image to transform (moving).
        dst_size (Tuple[int, int]): fixed image. Used for the output shape.
        affine_matrix (np.ndarray): affine transform.
        src_scale (float, optional): pre-scaling factor to apply to the
            src image before transforming it. Defaults to 1.
        dst_scale (float, optional): pre-scaling factor to apply to the
            dst image. Defaults to 1.

    Returns:
        np.ndarray: src_image transformed.
    """
    dst_shape = np.array(dst_size)
    matrix = np.array(affine_matrix)
    assert len(dst_shape) == 2
    assert matrix.shape == (3, 3)

    src_resized = resize(src_image, src_scale)
    dst_shape = (dst_shape * dst_scale).astype(int)
    src_transformed = cv2.warpPerspective(src_resized, affine_matrix, tuple(dst_shape))  # type: ignore
    return src_transformed


def warp_and_combine(
    src_image: np.ndarray,
    dst_image: np.ndarray,
    affine_matrix: np.ndarray,
    src_scale: float = 1,
    dst_scale: float = 1,
    combine_mode: str = "falsecolor",
    contour_size: int = 8,
    contour_color: Tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """Transforms the source image and combines it with the destination image.
    Additionally you can emphasize the edges of the input image.

    Args:
        src_image (np.ndarray): image to transform (moving).
        dst_image (np.ndarray): fixed image (fixed).
        affine_matrix (np.ndarray): ffine transform.
        src_scale (float, optional): pre-scaling factor to apply to the
            src image before transforming it. Defaults to 1.
        dst_scale (float, optional): pre-scaling factor to apply to the
            dst image. Defaults to 1.
        combine_mode (str, optional): Image combination method.
            Defaults to "falsecolor".
        contour_size (int, optional): Edge emphasis width. Defaults to 8.
        contour_color (Tuple[int, int, int], optional): Color of the edges.
            Defaults to (255, 255, 0).

    Returns:
        np.ndarray: combined image.
    """
    src_transformed = warp_image(
        src_image,
        dst_image.shape[:2],
        affine_matrix,
        src_scale=src_scale,
        dst_scale=dst_scale,
    )
    dst_image = resize(dst_image, dst_scale)
    combined_image = plots.impair(src_transformed, dst_image, combine_mode)
    if contour_size > 0:
        combined_image = plots.draw_contour(
            combined_image,
            resize(src_image, src_scale).shape,
            affine_matrix,
            color=contour_color,
        )
    return combined_image


def initialize_population(
    popsize: int,
    tm_loc: Tuple[int, int],
    bounds: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Creates the initial population. The initial population is composed of
    `popsize` affine transformations created randomly by the LHS method and
    an extra transformation that represents only the translation of the image.
    For this reason, the output has size `popsize` + 1.

    Args:
        popsize (int): population size.
        tm_loc (Tuple[int, int]): translation of the image.
        bounds (np.ndarray): bounds of individuals.
        seed (Optional[int], optional): random number generator to reproduce
            results. Defaults to None.

    Returns:
        np.ndarray: initial population normalized
    """
    tm_individual = create_individual(tm_loc)
    tm_individual_norm = evolutionary.normalize_population(
        tm_individual, bounds
    )  # [0.5] * 6
    pop_norm_lhc = evolutionary.init_population_lhs(popsize, 6, seed)
    pop_norm = np.append(pop_norm_lhc, tm_individual_norm)
    pop_norm = pop_norm.reshape(popsize + 1, 6)
    return pop_norm


# Main function
def run(
    agf_image: np.ndarray,
    octa_image: np.ndarray,
    octa_scale: float,
    info_th: float,
    seed: Optional[int] = None,
    half_resolution=False,
    cores: int = -1,
    display_progress_bar: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """It registers the FA and OCTA images directly, performing all the stages
    directly. Returns the estimated affine transformation matrix to transform
    the OCTA image.

    Keep in mind that to apply this transformation, it is first necessary to
    scale the OCTA image using the `octa_scale` factor

    Args:
        agf_image (np.ndarray): FA image as fixed image.
        octa_image (np.ndarray): OCTA image as moving image.
        octa_scale (float): scale factor to apply to the OCTA image.
        info_th (float): Maximum blood vessel information expected to be found
            on the OCTA image.
        seed (Optional[int], optional): random number generator to reproduce
            results. Defaults to None.
        half_resolution (bool, optional): Perform the process using the images
            with half their original size. This can speed up the process.
            Defaults to False.
        cores (int, optional): Number of cores used in the evolutionary
            algorithm. If set to -1, then it uses all available cores.
            Defaults to -1.
        display_progress_bar (bool, optional): display progress bar of the
            process. Defaults to False.
        verbose (bool, optional): Prints information at the end of the process.
            Defaults to False.

    Returns:
        np.ndarray: estimated affine transformation matrix.
    """
    start_time = time.time()

    _, _, agf_seg = preprocess_agf(agf_image)
    _, _, octa_seg = preprocess_octa(octa_image, hyst_th_percent=info_th)
    octa_seg = resize(octa_seg, octa_scale)

    _, tm_loc = template_matching(agf_seg, octa_seg, "ZNCC")

    if half_resolution:
        loc = (np.asarray(tm_loc, float) * 0.5).astype(int)
        translation_margin = 15
        octa_seg_resize = resize(octa_seg, 0.5)
        agf_seg = resize(agf_seg, 0.5)
    else:
        loc = np.array(tm_loc)
        translation_margin = 30
        octa_seg_resize = octa_seg.copy()

    # fine tuning
    bounds = affine_bounds(tuple(loc), translation_margin, 0.05, 15, 5)
    pop_size = 20
    initial_pop = initialize_population(pop_size, tuple(loc), bounds, seed)

    sol = evolutionary.differential_evolution(
        objective_function,
        bounds=bounds,
        args=(octa_seg_resize, agf_seg),
        mutation=0.5,
        recombination=0.75,
        generations=150,
        popsize=pop_size + 1,
        initial_population=initial_pop,
        max_dist=0.01,
        seed=seed,
        cores=cores,
        display_progress_bar=display_progress_bar,
    )

    best_idv = sol.best_individual()
    if half_resolution:
        best_idv[:2] *= 1 / 0.5

    matrix = individual_to_affine(best_idv)
    h2, w2 = np.array(octa_seg.shape[:2]) / 2
    matrix = (
        AffineTransform(translation=(w2, h2)).params
        @ matrix
        @ AffineTransform(translation=(-w2, -h2)).params
    )

    end_time = time.time()
    total_time = end_time - start_time

    if verbose:
        print(f"Resume of process:")
        print(f" * Template Matching location: {tm_loc}")
        print(f" * Evolutive information: ")
        print(f"     * Fitness value:         {sol.best_fitness_value():.4f}")
        print(f"     * Number of generations: {sol.total_gens_done} / {sol.max_gens}")
        print(f"     * Execution time:        {sol.execution_time:.2f} sec.")
        print(f" * Total process time: {total_time:.2f} sec.")
        print(f" * Estimated Affine transform: ")
        print(
            "      ",
            np.array2string(
                matrix,
                precision=2,
                suppress_small=True,
                separator=", ",
                prefix="       ",
            ),
        )

    return matrix
