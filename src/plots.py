import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np


def impair(
    img_a: np.ndarray, img_b: np.ndarray, method: str = "falsecolor", **kargs
) -> np.ndarray:
    """[summary]

    Parameters
    ----------
    img_a : np.ndarray
        Image A
    img_b : np.ndarray
        Image B
    method : str, optional
        Disponibles:
            - 'falsecolor':
            - 'diff':
            - 'blend':
            - 'chess': custom arg N, by default N=5
        By default 'falsecolor'.

    Returns
    -------
    [type]
        [description]
    """
    A = np.asarray(img_a, np.uint8)
    B = np.asarray(img_b, np.uint8)

    assert len(A.shape) == 2, "Image A must be grayscale"
    assert len(B.shape) == 2, "Image B must be grayscale"
    assert A.shape == B.shape

    if method == "falsecolor":
        return np.dstack((A, B, A))

    if method == "diff":
        return ((B.astype(np.int16) - A.astype(np.int16)) / 2 + 128).astype(np.uint8)

    if method == "blend":
        alpha = kargs.get("alpha", 0.5)
        beta = kargs.get("beta", 0.5)
        gamma = kargs.get("gamma", 0)
        return cv2.addWeighted(B, alpha, A, beta, gamma)

    if method == "chess":
        assert A.shape[0] == A.shape[1], "Images must be squared"
        N = kargs.get("N", 5)
        out = A.copy()
        bsize = A.shape[0] // N
        for rb in range(N):
            for cb in range(N):
                rs = bsize * rb
                re = bsize * (rb + 1)
                cs = bsize * cb
                ce = bsize * (cb + 1)
                if (rb + cb) % 2 == 0:
                    out[rs:re, cs:ce] = B[rs:re, cs:ce]
        return out
    raise ValueError()


def imshowpair(img_a, img_b, method="falsecolor", plot_base_size=6, **kargs):
    title = kargs.get("title", None)
    if method == "montage":
        plot_mult([img_a, img_b], None, 1, 2, title, base_size=plot_base_size)
    else:
        out_im = impair(img_a, img_b, method, **kargs)
        plot_mult([out_im], None, 1, 1, title, base_size=plot_base_size)


def draw_contour(im, contour_size, transform=None, line_width=4, color=(255, 255, 0)):
    out_image = im.copy()
    if len(out_image.shape) == 2:
        out_image = np.dstack((out_image, out_image, out_image))

    if transform is None:
        transform = np.identity(3)
    assert transform.shape == (3, 3)

    h, w = contour_size[:2]

    corners = np.array([[0, 0], [0, h], [w, 0], [w, h]], float).reshape(-1, 1, 2)
    scene_corners = (
        cv2.perspectiveTransform(corners, transform).reshape(-1, 2).astype(int)
    )

    cv2.line(
        out_image, tuple(scene_corners[0]), tuple(scene_corners[1]), color, line_width
    )
    cv2.line(
        out_image, tuple(scene_corners[0]), tuple(scene_corners[2]), color, line_width
    )
    cv2.line(
        out_image, tuple(scene_corners[1]), tuple(scene_corners[3]), color, line_width
    )
    cv2.line(
        out_image, tuple(scene_corners[2]), tuple(scene_corners[3]), color, line_width
    )

    return out_image


def dash_line(image, point_a, point_b, color, line_width, n_dashes=4):
    lenght = np.linalg.norm(np.subtract(point_a, point_b))
    seg_lenght = lenght / ((n_dashes * 2) - 1)

    vdir = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]], float) / lenght

    points = [np.array(point_a)]
    for i in range(1, n_dashes * 2 - 1):
        new_point = np.array(points[i - 1]) + vdir * seg_lenght
        points.append(new_point)
    points.append(np.array(point_b))

    out_image = image.copy()
    for i in range(0, n_dashes * 2, 2):
        start_point = tuple(points[i].astype(int))
        end_point = tuple(points[i + 1].astype(int))

        cv2.line(out_image, start_point, end_point, color, line_width)

    return out_image


def draw_dashed_contour(
    im, contour_size, transform=None, dashes=4, line_width=4, color=(255, 255, 0)
):

    out_image = im.copy()
    if len(out_image.shape) == 2:
        out_image = np.dstack((out_image, out_image, out_image))

    if transform is None:
        transform = np.identity(3)
    assert transform.shape == (3, 3)

    h, w = contour_size[:2]

    corners = np.array([[0, 0], [0, h], [w, 0], [w, h]], float).reshape(-1, 1, 2)
    scene_corners = (
        cv2.perspectiveTransform(corners, transform).reshape(-1, 2).astype(int)
    )
    scene_corners = [tuple(corner) for corner in scene_corners]

    out_image = dash_line(
        out_image, scene_corners[0], scene_corners[1], color, line_width, dashes
    )
    out_image = dash_line(
        out_image, scene_corners[0], scene_corners[2], color, line_width, dashes
    )
    out_image = dash_line(
        out_image, scene_corners[1], scene_corners[3], color, line_width, dashes
    )
    out_image = dash_line(
        out_image, scene_corners[2], scene_corners[3], color, line_width, dashes
    )

    return out_image


def _fig_mult(
    images,
    labels=None,
    rows: int = 1,
    cols: int = 1,
    title: str = None,
    display_axis=False,
    base_size=6,
    subplots_space=0.01,
):
    assert len(images) <= rows * cols

    fig = plt.figure(figsize=(base_size * cols, base_size * rows))

    if title:
        fig.suptitle(title)

    for i in range(len(images)):
        ax = plt.subplot(rows, cols, i + 1)
        im = images[i]

        if not display_axis:
            ax.axis("off")

        if len(im.shape) == 2:
            ax.imshow(im, cmap="gray", aspect="equal")
        else:
            ax.imshow(im, aspect="equal")

        if labels is not None:
            ax.set_title(labels[i])

    if labels is None:
        plt.subplots_adjust(
            left=0.01,
            bottom=0,
            right=0.99,
            top=1,
            hspace=subplots_space,
            wspace=subplots_space,
        )
    else:
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.tight_layout()
    return fig


def plot_mult(
    images,
    labels=None,
    rows: int = 1,
    cols: int = 1,
    title: str = None,
    display_axis=False,
    base_size=6,
    subplots_space=0.03,
):
    _fig_mult(
        images, labels, rows, cols, title, display_axis, base_size, subplots_space
    )
    plt.show()


def save_mult(
    filename,
    images,
    labels=None,
    rows: int = 1,
    cols: int = 1,
    title: str = None,
    base_size=6,
    subplots_space=0.03,
):
    fig = _fig_mult(images, labels, rows, cols, title, base_size, subplots_space)
    fig.savefig(filename, bbox_inches="tight", facecolor="w")

    fig.clear()
    plt.close(fig)
    gc.collect()


def plot_mult_hist(
    images, labels, rows: int, cols: int, title: str = None, base_size=6
):
    fig = plt.figure(figsize=(base_size * cols, base_size * rows))
    if title:
        fig.suptitle(title)

    for i, (im, label) in enumerate(zip(images, labels)):
        ax = plt.subplot(rows, cols, i + 1)
        ax.hist(im.ravel(), 256)
        ax.set_title(label)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
