import cv2
import numpy as np
import pandas as pd
import tifffile as tif
from skimage.measure import regionprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

NEGATIVE_SAMPLES = ['maddi_1_090', 'bank_2_026', 'boffel_2_159', 'bank_6_116', 'boffel_4_095', 'guld_5_114']
TRAINING_FEATURES = ['area', 'solidity', 'extent', 'intensity_max', 'intensity_mean', 'axis_major_length', 'axis_minor_length']
SMALL_BUG_THRESHOLD = 1000
BUG_INTENSITY_THRESHOLD = 50
BUG_DISTANCE_MAP_THRESHOLD = 3
MINIMUM_MARKER_SIZE = 10


def fit_knn(csv_path, n_neighbors: int) -> tuple[KNeighborsClassifier, StandardScaler]:
    df = pd.read_csv(csv_path)
    y = df["class"]
    X = df[TRAINING_FEATURES]
    scaler = StandardScaler()
    scaler.fit(X)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(scaler.transform(X), y)
    return knn, scaler


def segment_bugs(image: np.ndarray, threshold: int = BUG_INTENSITY_THRESHOLD):
    # Segment one or more bug instances given an input image volume
    return np.array(image > threshold).astype(int)


def extract_features(mask: np.ndarray, image: np.ndarray, file_path: str) -> pd.Series:
    """
    Compute features from a segmented bug instance and return as a pd.Series
    Parameters
    ----------
    mask: np.ndarray
        Binary mask identifying the bug
    image: np.ndarray
        Bug image
    file_path:
        Full path to this bug image. Used to extract class and case name.

    Returns
    -------
    A pd.Series containing fields class, case, centroid, and all features defined by TRAINING_FEATURES
    """
    r = regionprops(mask, intensity_image=image)
    *_, a_class, file_name = file_path.split("/")
    my_dict = {'class': a_class.lower(), 'case': file_name, 'centroid': r[0]['centroid']}
    for f in TRAINING_FEATURES:
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


def save_volume(image: np.ndarray, path: str):
    tif.imwrite(path, image)
