import os
import torch
import numpy as np
from PIL import Image
import logging
from Dataloader import Kittiloader
from Transformer import Transformer
from torch.utils.data import Dataset, DataLoader


class KittiDataset(Dataset):

    # param kittiDir - path to kitti_data
    # param mode - 'train', 'test' or 'val'
    # param splits - which split to use kitti' or 'eigen' (eigen contains depth maps whereas kitti does not)
    # param transform - what transforms should be applied to the data
    def __init__(self,
                 kittiDir,
                 mode,
                 splits='kitti',
                 transform=None):
        self.mode = mode
        self.kitti_root = kittiDir
        self.transform = transform

        # use cam=2 the left image by default
        self.kittiloader = Kittiloader(kittiDir, mode, splits, cam=2)

    def __getitem__(self, idx):
        # load an item according to the given index
        data_item = self.kittiloader.load_item(idx)
        data_trans = self.transform(data_item)
        return data_trans


    def __len__(self):
        return self.kittiloader.data_length()

class DataGenerator(object):
    def __init__(self,
                 KittiDir,
                 phase,
                 splits='kitti'):

        self.phase = phase
        self.split = splits

        transformer = Transformer(self.phase)
        self.dataset = KittiDataset(KittiDir, phase, splits, transformer.get_transform())

    def create_data(self, batch_size, nthreads=0):
        # use page locked gpu memory by default
        return DataLoader(self.dataset,
                          batch_size,
                          shuffle=(self.phase=='train'),
                          num_workers=nthreads,
                          pin_memory=True)
