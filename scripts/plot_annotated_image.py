"""This is an example python script for loading image and segmentation into the correct format
   Requirements: skimage, numpy
"""
import argparse
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.io import imread


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir')
    parser.add_argument('--labels_sheet')
    return parser.parse_args()


def save_annotated_figure(image_path: str, labels: str, save_img_path: Path):
    """ Loads image and displa labels centerpoints

    Args:
        image_path (str): path to .tif file
        labels (str): string containing the labels formatted as described in the kaggle dataset
        save_img_path (str): path to save slices as .png
    """
    # Image is loading into array with orientation (z-aixs, x-axis,y-axis) so dim: (128,92,92)
    img = imread(image_path)
    print(f"img dim: {img.shape}")

    # Important labels are in format (x-axis, y-axis, z-aixs)
    pred_centerpoints = labels.replace(" ", "").rstrip(";").split(";")  # Removes whitespace and last ';' if applicable
    filtered_centerpoints = [float(item) for item in pred_centerpoints if not re.search("[a-zA-Z]", item)]
    pred_centers = np.array(
        list(zip(filtered_centerpoints[::3], filtered_centerpoints[1::3], filtered_centerpoints[2::3])))

    print("Centerpoints: ", pred_centers)

    # Plots the first 16 centerpoints
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))

    for i in range(min(16, len(pred_centers))):
        axs[i // 4, i % 4].imshow(img[int(pred_centers[i][2])])  # third coordinate of labels are z-axis
        axs[i // 4, i % 4].plot(int(pred_centers[i][0]), int(pred_centers[i][1]), "xr",
                                markersize=10)  # first and second coordinate is x and y axis
        axs[i // 4, i % 4].axis('off')  # Remove axis on images

    fig.suptitle("Center points of bugs in volume", fontsize='xx-large')
    fig.tight_layout()
    fig.savefig(save_img_path, bbox_inches='tight')


if __name__ == "__main__":
    cl_args = parse_cl_args()

    img_glob = sorted(glob(os.path.join(cl_args.image_dir, '*.tif')))
    img_glob = [i for i in img_glob if 'label' not in i]

    df = pd.read_csv(cl_args.labels_sheet)

    if not os.path.isdir(os.path.join(cl_args.image_dir, 'annotated')):
        os.makedirs(os.path.join(cl_args.image_dir, 'annotated'))

    for img_path in img_glob:
        case_name = os.path.basename(img_path)
        pred_centerpoints_str = df['centerpoints'][df['filename'] == case_name].values[0]
        save_annotated_figure(img_path, pred_centerpoints_str, Path(Path(img_path).parent, 'annotated', case_name))