import argparse
from glob import glob

import numpy as np
from skimage.measure import label
from skimage.morphology import binary_erosion, dilation, remove_small_objects
from scipy.ndimage import generate_binary_structure, iterate_structure

from bugnist import load_volume, segment_bugs, extract_features, SMALL_BUG_THRESHOLD, fit_knn, save_volume

STRUCT = iterate_structure(generate_binary_structure(3, 1), 2)


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_csv")
    parser.add_argument("--test_data_root")
    return parser.parse_args()


if __name__ == "__main__":
    cl_args = parse_cl_args()

    # Load the training csv and fit a kNN
    knn = fit_knn(cl_args.feature_csv, n_neighbors=5)

    test_images = sorted(glob(f"{cl_args.test_data_root}/*.tif"))

    # For each test image:
    for test_image in test_images:
        #   Load the image
        img = load_volume(test_image)

        #   Segment and divide up bugs
        bugs_mask = segment_bugs(img, 30)

        # Remove small volumes, Erode to separate, then label, then dilate back
        bugs_mask = remove_small_objects(bugs_mask, SMALL_BUG_THRESHOLD)
        bugs_eroded = binary_erosion(bugs_mask, STRUCT)
        bugs_labelled = label(bugs_eroded)
        bugs_labelled = dilation(bugs_labelled, STRUCT)
        save_volume(bugs_labelled.astype('uint8'), test_image.replace(".tif", "_labelled.tif"))

        centerpoints_string = ""
        #   For each bug
        for idx in np.unique(bugs_labelled[bugs_labelled > 0]):
            this_bug_mask = np.asarray(bugs_labelled == idx, int)

            if np.sum(this_bug_mask) > SMALL_BUG_THRESHOLD:
                # Compute the feature representation of this bug in the test image
                features_for_this_bug = extract_features(this_bug_mask, img, test_image)

                # Apply kNN to the test-bug feature representation to assign bug class
                predicted_class = knn.predict(features_for_this_bug)

                # Append to centerpoint string using kNN assigned class and centroid from feature representation
                this_centerpoint_string = (f"{predicted_class};"
                                           f"{features_for_this_bug['centroid'][1]};"
                                           f"{features_for_this_bug['centroid'][2]};"
                                           f"{features_for_this_bug['centroid'][0]};")
                centerpoints_string += this_centerpoint_string
