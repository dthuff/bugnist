import argparse
import os
import time
from glob import glob

import numpy as np
import pandas as pd
from skimage.measure import regionprops
from tqdm import tqdm

from bugnist import load_volume, segment_bugs, save_volume


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root")
    return parser.parse_args()


if __name__ == '__main__':
    cl_args = parse_cl_args()

    columns = ['case', 'class', 'centroid', 'bbox']
    df = pd.DataFrame(columns=columns)
    list_of_dirs = [f for f in os.listdir(cl_args.data_root) if os.path.isdir(os.path.join(cl_args.data_root, f))]

    for a_dir in list_of_dirs:
        bug_class = a_dir.lower()
        print(f"Starting class: {bug_class}")
        tif_glob = sorted(glob(os.path.join(cl_args.data_root, a_dir, '*.tif')))
        for a_file in tqdm(tif_glob):
            bug_img = load_volume(a_file)
            bug_segm = segment_bugs(bug_img)
            if np.any(bug_segm):
                r = regionprops(bug_segm)
                *_, file_name = a_file.split("/")
                df.loc[len(df)] = pd.Series({'case': file_name,
                                             'class': a_dir.lower(),
                                             'centroid': r[0]['centroid'],
                                             'bbox': r[0]['bbox']})
                bbox_img = np.zeros(bug_img.shape)
                bbox_img[r[0]['bbox'][0]:r[0]['bbox'][3],
                            r[0]['bbox'][1]:r[0]['bbox'][4],
                            r[0]['bbox'][2]:r[0]['bbox'][5]] = 1
                save_volume(bbox_img.astype('uint8'), a_file.replace('.tif', '_bbox.tif'))
            else:
                print(f"Skipped case: {a_file}")

    t = time.localtime()
    current_time = time.strftime("%Y%m%d_%H%M%S", t)
    df.to_csv(os.path.join(cl_args.data_root, f"bounding_boxes.csv"), index=False)
