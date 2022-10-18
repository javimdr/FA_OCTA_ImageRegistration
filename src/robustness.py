import numpy as np
import metrics
import plots
import registration
import cv2
from skimage.transform import AffineTransform


PX_TO_MICRAS = np.round(200/19, 4)

def pixels_to_micras(pixels):
    return pixels * PX_TO_MICRAS

def micras_to_pixels(micras):
    return micras / PX_TO_MICRAS


def check_robustness(octa3_size, octa3_scale, 
                     tf_octa3_fa, tf_octa6_fa, tf_octa3_octa6, 
                     d_max=19.): 
    r"""
    Check robustness

    Parameters
    ----------
    octa3_shape : Sequence
        Size (HxW) of original OCTA $3 \times 3$ image
    octa3_scale : Float
        Scale factor between OCTA $3 \times 3$ and FA images.
    tf_octa3_fa : np.ndarray
        Affine transform between OCTA $3 \times 3$  and FA images.
    tf_octa6_fa : np.ndarray
        Affine transform between OCTA $6 \times 6$ and FA images.
    tf_octa3_octa6 : np.ndarray
        Affine transform between OCTA $3 \times 3$ and $6 \times 6$ images.
    d_max : Float, optional
        Distance in pixels used as tolerance to detect robustness, by default 19.

    Returns
    -------
    Boolean, Float
        Return if distance is less than d_max and the distance
    """
    # Use central point of octa3 image instead top-left corner to compare
    central_point = (np.array(octa3_size)*octa3_scale) // 2
    central_point = np.append(central_point, [1])  # homogeneus (x, y, 1)
        
    point_a = tf_octa3_fa @ central_point.T
    point_b = (tf_octa6_fa@tf_octa3_octa6) @ central_point.T
    
    dist = metrics.euclidean_dist(point_a, point_b)
    return dist <= d_max, dist


def transform_between_octas(octa_3x3_image, octa_3x3_scale, octa_3x3_hyst_th_percent,
                            octa_6x6_image, octa_6x6_scale, octa_6x6_hyst_th_percent):

    _, _, octa3_seg = registration.preprocess_octa(
        octa_3x3_image, hyst_th_percent=octa_3x3_hyst_th_percent
    )
    _, _, octa6_seg = registration.preprocess_octa(
        octa_6x6_image, hyst_th_percent=octa_6x6_hyst_th_percent
    )

    octa3_seg = registration.resize(octa3_seg, octa_3x3_scale)
    octa6_seg = registration.resize(octa6_seg, octa_6x6_scale)

    # Template matching between OCTA 3x3 and OCTA 6x6
    _, loc_octa6_octa3 = registration.template_matching(octa6_seg, octa3_seg)
    return AffineTransform(translation=loc_octa6_octa3).params


def plot_transforms(agf_image, octa3_image, octa6_image, octa3_scale, octa6_scale,
                    tf_octa3_fa, tf_octa6_fa, tf_octa3_octa6,
                    plot_de=True, plot_tm=True, plot_robustness=True,
                    line_width=8, dashes=6, title=None, base_size=6):

    RED =    (255,   0,   0)
    GREEN =  (  0, 255,   0)
    BLUE =   (  0,   0, 255)
    YELLOW = (255, 255,   0)

    octa3 = registration.resize(octa3_image, octa3_scale)
    octa6 = registration.resize(octa6_image, octa6_scale)

    titles = []
    images = []

    if plot_de:                    
        # Image 2: Regsitrations 3x3 and 6x6 over AGF
        title2 = r"BLUE: $\Psi_{\mathregular{DE}}\ (\mathregular{OCTA}\ 3 \times 3)$" + \
                 "\n" + \
                 r"RED:  $\Psi_{\mathregular{DE}}\ (\mathregular{OCTA}\ 6 \times 6)$"
        img2_agf = plots.draw_contour(agf_image, octa3.shape, tf_octa3_fa, 
                                    line_width=line_width, color=BLUE)
        img2_agf = plots.draw_contour(img2_agf, octa6.shape, tf_octa6_fa, 
                                    line_width=line_width, color=RED)
        img2_agf = plots.draw_contour(img2_agf, img2_agf.shape, np.identity(3), 
                                    line_width=line_width*2, color=GREEN)
        
        titles.append(title2)
        images.append(img2_agf)


    if plot_tm:
        # Image 1: TM between OCTAS
        title1 = r"BLUE DOTTED:  $\Psi_{\mathregular{TM}}\ (\mathregular{OCTA}\ 3 \times 3)$"
        img1_octa3t = cv2.warpPerspective(octa3, tf_octa3_octa6, octa6.shape)
        img1_octas = plots.impair(img1_octa3t, octa6, "blend")
        img1_octas = plots.draw_dashed_contour(
            img1_octas, octa3.shape, transform=tf_octa3_octa6, dashes=dashes, 
            line_width=line_width-2, color=YELLOW
        )
        img1_octas = plots.draw_contour(
            img1_octas, img1_octas.shape, transform=np.identity(3),
            line_width=line_width*2, color=RED
        )

        titles.append(title1)
        images.append(img1_octas)


    if plot_robustness:
        
        # Image 3: Difference between error sizes
        title3 = r"BLUE: $\Psi_{\mathregular{DE}}\ (\mathregular{OCTA}\ 3 \times 3)$" + \
                "\n" + \
                r"BLUE DOTTED:  $\Psi_{\mathregular{DE}}\ (\Psi_{\mathregular{TM}}\ (\mathregular{OCTA}\ 3 \times 3))$"
        img3_agf = plots.draw_contour(agf_image, octa3.shape, tf_octa3_fa, 
                                    line_width=line_width+2, color=BLUE)
        # img3_agf = plots.draw_contour(img3_agf, octa6.shape, tf_octa6_fa, 
        #                               line_width=line_width, color=RED)
        img3_agf = plots.draw_dashed_contour(img3_agf, octa3.shape, tf_octa6_fa@tf_octa3_octa6, 
                                            dashes=dashes, line_width=line_width-2, color=YELLOW)
        img3_agf = plots.draw_contour(img3_agf, img3_agf.shape, np.identity(3), 
                                    line_width=line_width*2, color=GREEN)
                                            
        titles.append(title3)
        images.append(img3_agf)

    
    plots.plot_mult(images, titles, 1, len(images), 
                    title=title, base_size=base_size)

                         