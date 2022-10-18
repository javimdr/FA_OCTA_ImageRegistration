import os
import re
from typing import List, Tuple

import cv2
import numpy as np

# OCTA sizes in dataset: 3x3mm and 6x6mm
OCTA_SIZES = ["3X3", "6X6"]

# Fluorescein Angiography (AGF), OCTA 3x3mm and OCTA 6x6mm
IMAGE_TYPES = ["AGF", "3X3", "6X6"]

# Posibles temporal capture time of images in dataset: Initial, Final
IMAGE_TIMES = ["INICIAL", "FINAL"]

# Possible clinical eye states from the dataset: Healthy, Pathological
IMAGE_EYE_STATES = ["SANO", "ENF"]


class TestCase:
    """Represents the data required for an image registration case.
    It contains two images OCTA (3x3 and 6x6mm) and FA image."""

    def __init__(
        self,
        patient_id: str,
        time_instant: str,
        eye_state: str,
        dataset_path: str,
        img_ext: str = ".png",
    ):

        if not time_instant in IMAGE_TIMES:
            raise ValueError(f"Select one of this: {IMAGE_TIMES}")
        if not eye_state in IMAGE_EYE_STATES:
            raise ValueError(f"Select one of this: {IMAGE_EYE_STATES}")

        self.patient_id = patient_id
        self.time_instant = time_instant
        self.eye_state = eye_state
        self.layer = "SUPERF"
        self._dataset_path = dataset_path
        self._IMG_EXTENSION = img_ext

    def label(self) -> str:
        """Unique label representing the registration case

        Returns:
            str: unique label
        """
        return f"{self.patient_id}_{self.time_instant}_{self.eye_state}"

    def filename_octa(self, size: str) -> str:
        """Returns the filepath of the OCTA image.

        Args:
            size (str): OCTA size: '3x3' or '6x6'.

        Returns:
            str: OCTA image filepath
        """
        assert size.upper() in OCTA_SIZES
        filename = f"{self.time_instant} {size.upper()} {self.layer} {self.eye_state}{self._IMG_EXTENSION}"
        return os.path.join(self._dataset_path, self.patient_id, filename)

    def filename_agf(self) -> str:
        """Returns the filepath of the FA image.

        Returns:
            str: FA image filepath
        """
        filename = f"AGF {self.time_instant} {self.eye_state}{self._IMG_EXTENSION}"
        filename = os.path.join(self._dataset_path, self.patient_id, filename)
        return filename

    def exists_all_images(self) -> bool:
        """Check if FA, OCTA 3x3 and OCTA 6x6 images exist in the dataset for
        the given configuration.

        Returns:
            bool: images exist.
        """
        exists_agf = os.path.exists(self.filename_agf())
        exists_octas = all(
            [os.path.exists(self.filename_octa(size)) for size in OCTA_SIZES]
        )

        return exists_agf and exists_octas

    @staticmethod
    def _read_image(filename: str, grayscale: bool = True) -> np.ndarray:
        """Load an image from local storage.

        Args:
            filename (str): Filepath of the image
            grayscale (bool, optional): Loaded in grayscale if true.
                Defaults to True.

        Raises:
            FileNotFoundError: Image filepath not found.

        Returns:
            np.ndarray: Read image.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found. Path: {filename}")

        color_flag = 0 if grayscale else 1
        return cv2.imread(filename, color_flag)  # type: ignore

    def load_image(self, image_type: str, grayscale: bool = True) -> np.ndarray:
        """Load an image from local storage. It can load the AGF, OCTA 3x3 and
        OCTA 6x6 images.

        Args:
            image_type (str): Image to load. Can be: AGF, 3x3 or 6x6.
            grayscale (bool, optional): Loaded in grayscale if true.
                Defaults to True.

        Raises:
            ValueError: selected image type is not valid.

        Returns:
            np.ndarray: Loaded image
        """
        if str.upper(image_type) == "AGF":
            return self._read_image(self.filename_agf(), grayscale)
        if str.upper(image_type) in OCTA_SIZES:
            return self._read_image(self.filename_octa(image_type), grayscale)

        raise ValueError(
            f"Image type '{image_type}' is not valid. "
            f"Select one of the following: {IMAGE_TYPES}"
        )

    def read_control_points(
        self, groundtruth_path: str, octa_size: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read control points of the registratrion.

        Args:
            groundtruth_path (str): filepath of groundtruth of points.
            octa_size (str): Load points of the especific OCTA size.

        Raises:
            FileNotFoundError: file not found.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Points of OCTA and FA.
        """
        filename = f"{self.label()}_{octa_size}.csv"
        filename = os.path.join(groundtruth_path, filename)

        if not os.path.exists(filename):
            raise FileNotFoundError("Control points file does not found.")

        data = np.genfromtxt(filename, comments="#", delimiter=",")
        src_pts = data[:, :2]
        dst_pts = data[:, 2:]
        return src_pts, dst_pts


class Dataset:
    """It load the dataset, storing all image registration cases as test cases."""

    def __init__(self, dataset_path: str, img_ext: str = ".png"):
        if not os.path.exists(dataset_path) or os.path.isfile(dataset_path):
            raise NotADirectoryError(f"Dataset path not valid")

        self._dataset_path = dataset_path

        self._img_ext = img_ext
        self._test_cases = dict()
        self._read()

    def _read(self) -> None:
        """Find and load all registration cases in the dataset folder."""
        directory = os.listdir(self._dataset_path)
        directory.sort(key=lambda f: int(re.sub(r"\D", "", f)))
        for patient_id in directory:
            for time_instant in IMAGE_TIMES:
                for eye_state in IMAGE_EYE_STATES:
                    test_case = TestCase(
                        patient_id,
                        time_instant,
                        eye_state,
                        self._dataset_path,
                        self._img_ext,
                    )
                    if test_case.exists_all_images():
                        self._test_cases[test_case.label()] = test_case

    def get_test_cases(self) -> List[TestCase]:
        """Returns a list of all loaded registration cases.

        Returns:
            List[TestCase]:  list of all registration cases.
        """
        return [case for case in self._test_cases.values()]

    def get_test_labels(self) -> List[str]:
        """Returns a list with the labels of each loaded registration cases.

        Returns:
            List[str]: List of labels.
        """
        return [case.label() for case in self.get_test_cases()]

    def get_case(self, label: str) -> TestCase:
        """Returns, if it exists, a test case selected by its label.

        Args:
            label (str): Test case label.

        Returns:
            TestCase: Test case if exists, else None.
        """
        return self._test_cases.get(label, None)
