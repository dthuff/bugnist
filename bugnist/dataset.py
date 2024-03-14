import torch
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from bugnist import load_volume


class BugNistDataset(Dataset):
    def __init__(self, root, transforms, bbox_csv):
        self.root = root
        self.transforms = transforms
        self.bbox_csv = bbox_csv
        # load all image files, sorting them to
        # ensure that they are aligned
        all_tifs = glob(f'{root}/*/*.tif')
        self.imgs = sorted([i for i in all_tifs if 'bbox' not in i])
        self.masks = sorted([i for i in all_tifs if 'bbox' in i])

    def __getitem__(self, idx):
        img = load_volume(self.imgs[idx])
        mask = load_volume(self.masks[idx])
        return img, mask

class BugNistDataLoader(DataLoader):
    def __init__(self):