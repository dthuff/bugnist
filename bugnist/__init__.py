import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops


def get_features():
    return ['class', 'area', 'solidity', 'axis_major_length', 'axis_minor_length']


def segment_bugs(image: np.ndarray, threshold: int = 30):
    # Segment one or more bug instances given an input image volume
    return np.array(image > threshold).astype(int)


def extract_features(mask: np.ndarray, image: np.ndarray, a_class: str) -> pd.Series:
    r = regionprops(mask, intensity_image=image)
    my_features = get_features()
    my_features.remove('class')
    my_dict = {'class': a_class}
    for f in my_features:
        my_dict[f] = r[0][f]

    return pd.Series(my_dict)


def load_volume(file_path: str) -> np.ndarray:
    """
    Read a 3D TIFF image and return it as a NumPy array with axis order (z, x, y).

    Parameters
    ----------
    file_path : str
        The path to the 3D TIFF file.

    Returns
    -------
    numpy.ndarray
        The 3D TIFF image data as a NumPy array.
    """
    _, image = cv2.imreadmulti(file_path, flags=cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read file: {file_path}")

    return np.array(image)
