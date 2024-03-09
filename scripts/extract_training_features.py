import argparse
import os
import time
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from bugnist import load_volume, segment_bugs, extract_features, TRAINING_FEATURES, NEGATIVE_SAMPLES


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root")
    return parser.parse_args()


if __name__ == "__main__":
    cl_args = parse_cl_args()

    columns = ['case', 'class'] + TRAINING_FEATURES
    df = pd.DataFrame(columns=columns)
    list_of_dirs = [f for f in os.listdir(cl_args.data_root) if os.path.isdir(os.path.join(cl_args.data_root, f))]

    for a_dir in list_of_dirs:
        bug_class = a_dir.lower()
        print(f"Starting class: {bug_class}")
        tif_glob = sorted(glob(os.path.join(cl_args.data_root, a_dir, '*.tif')))
        for a_file in tqdm(tif_glob):
            if not [True for f in NEGATIVE_SAMPLES if f in a_file]:
                bug_img = load_volume(a_file)
                bug_segm = segment_bugs(bug_img, 30)
                if np.any(bug_segm):
                    row = extract_features(bug_segm, bug_img, a_file)
                    df.loc[len(df)] = row
            else:
                print(f"Skipped case: {a_file}")

    t = time.localtime()
    current_time = time.strftime("%Y%m%d_%H%M%S", t)
    df.to_csv(os.path.join(cl_args.data_root, f"features_{current_time}.csv"), index=False)
