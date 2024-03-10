import argparse
import os
import time
from glob import glob

import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, iterate_structure
from skimage.measure import label
from skimage.morphology import binary_erosion, dilation, remove_small_objects, remove_small_holes, binary_closing
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from bugnist import load_volume, segment_bugs, extract_features, SMALL_BUG_THRESHOLD, TRAINING_FEATURES, \
    BUG_INTENSITY_THRESHOLD, BUG_DISTANCE_MAP_THRESHOLD, MINIMUM_MARKER_SIZE, fit_knn, save_volume

STRUCT = iterate_structure(generate_binary_structure(3, 1), 2)


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_csv")
    parser.add_argument("--test_data_root")
    parser.add_argument("--debug", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    cl_args = parse_cl_args()

    # Load the training csv and fit a kNN
    knn, scaler = fit_knn(cl_args.feature_csv, n_neighbors=50)

    # Set up output dataframe
    df = pd.DataFrame(columns=['filename', 'centerpoints'])

    test_images = sorted(glob(f"{cl_args.test_data_root}/*.tif"))
    test_images = [t for t in test_images if 'labelled' not in t]

    for test_image in test_images:
        img = load_volume(test_image)

        # Initial segmentation and clean up of all bugs
        bugs_mask = segment_bugs(img, BUG_INTENSITY_THRESHOLD)
        bugs_mask = remove_small_objects(bugs_mask, SMALL_BUG_THRESHOLD)
        bugs_mask = remove_small_holes(bugs_mask, 50000)

        # Watershed to separate bugs
        distance = ndi.distance_transform_edt(bugs_mask)
        distance_thresholded = distance > BUG_DISTANCE_MAP_THRESHOLD
        markers = label(distance_thresholded)
        markers = remove_small_objects(markers, MINIMUM_MARKER_SIZE)
        bugs_labelled = watershed(-distance, markers, mask=bugs_mask)
        save_volume(bugs_labelled.astype('uint8'), test_image.replace(".tif", "_labelled.tif"))

        if cl_args.debug:  # Extra saving to debug watershed segmentation
            save_volume(distance.astype('uint8'), test_image.replace(".tif", "_distance.tif"))
            save_volume(markers.astype('uint8'), test_image.replace(".tif", "_markers.tif"))

        centerpoints_string = ""

        for idx in np.unique(bugs_labelled[bugs_labelled > 0]):
            this_bug_mask = np.asarray(bugs_labelled == idx, int)

            # Compute the feature representation of this bug in the test image
            features_for_this_bug = extract_features(this_bug_mask, img, test_image)

            # Apply kNN to the test-bug feature representation to assign bug class
            predicted_class = knn.predict(scaler.transform(pd.DataFrame(features_for_this_bug[TRAINING_FEATURES]).T))

            # Append to centerpoint string using kNN assigned class and centroid from feature representation
            this_centerpoint_string = (f"{predicted_class[0]};"
                                       f"{round(features_for_this_bug['centroid'][1], 2)};"
                                       f"{round(features_for_this_bug['centroid'][2], 2)};"
                                       f"{round(features_for_this_bug['centroid'][0], 2)};")
            centerpoints_string += this_centerpoint_string

        centerpoints_string = centerpoints_string[:-1]  # Drop trailing ;
        row = pd.Series({'filename': os.path.basename(test_image),
                         'centerpoints': centerpoints_string})
        df.loc[len(df)] = row

    df.to_csv(os.path.join(cl_args.test_data_root, f"predictions.csv"), index=False)
