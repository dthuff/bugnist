import argparse
import os
import time
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from bugnist import load_volume, segment_bugs, extract_features, get_features


# Learn representations of each target class in training data
# Produce a csv of bug features with class labels


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root")
    return parser.parse_args()


if __name__ == "__main__":
    cl_args = parse_cl_args()

    df = pd.DataFrame(columns=get_features())
    list_of_dirs = [f for f in os.listdir(cl_args.data_root) if os.path.isdir(os.path.join(cl_args.data_root, f))]

    for a_dir in list_of_dirs:
        bug_class = a_dir.lower()
        print(f"Starting class: {bug_class}")
        tif_glob = sorted(glob(f"{os.path.join(cl_args.data_root, a_dir)}/*.tif"))
        for a_file in tqdm(tif_glob):
            bug_img = load_volume(a_file)
            bug_segm = segment_bugs(bug_img, 30)
            if np.any(bug_segm):
                row = extract_features(bug_segm, bug_img, bug_class)
                df.loc[len(df)] = row

    t = time.localtime()
    current_time = time.strftime("%Y%m%d_%H%M%S", t)
    df.to_csv(os.path.join(cl_args.data_root, f"features_{current_time}.csv"))
