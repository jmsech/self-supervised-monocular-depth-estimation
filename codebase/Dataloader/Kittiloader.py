#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import logging
from PIL import Image
from .bin2depth import get_depth, get_focal_length_baseline



class Kittiloader(object):
    # param kittiDir - path to kitti_data
    # param mode - 'train', 'test' or 'val'
    # param splits - which split to use 'kitti' or 'eigen' (eigen contains depth maps whereas kitti does not)

    # param cam - camera id. 2 represents the left cam, 3 represents the right one
    def __init__(self, kittiDir, mode, splits='kitti',cam=2):
        self.mode = mode
        self.cam = cam
        self.files = []
        self.kitti_root = kittiDir
        self.split = splits
        # read filenames files
        filepath = os.path.dirname(os.path.realpath(__file__)) + \
                   '/filenames/{}_{}_files.txt'.format(self.split, self.mode)
        logging.info('Loading {} {}'.format(self.split, self.mode))
        if self.split == 'kitti':
            with open(filepath, 'r') as f:
                data_list = f.read().split('\n')
                for data in data_list:
                    if len(data) == 0:
                        continue
                    data_info = data.split(' ')
                    self.files.append({
                        "l_rgb": data_info[0],
                        "r_rgb": data_info[1]
                    })
        elif self.split == 'eigen':
            with open(filepath, 'r') as f:
                data_list = f.read().split('\n')
                for data in data_list:
                    if len(data) == 0:
                        continue
                    data_info = data.split(' ')
                    self.files.append({
                        "l_rgb": data_info[0],
                        "r_rgb": data_info[1],
                        "cam_intrin": data_info[2],
                        "depth": data_info[3],
                    })
        elif True:
            logging.error('Unrecognized split in Kittiloader')

    def data_length(self):
        return len(self.files)

    def _check_path(self, filename, err_info):
        file_path = os.path.join(self.kitti_root, filename)
        assert os.path.exists(file_path), err_info
        return file_path


    def _read_data(self, item_files):
        data = {}
        l_rgb_path = self._check_path(item_files['l_rgb'],
                                      "????::Cannot find Left Image at {}".format(item_files['l_rgb']))
        r_rgb_path = self._check_path(item_files['r_rgb'],
                                      "????::Cannot find Right Image at {}".format(item_files['r_rgb']))

        l_rgb = Image.open(l_rgb_path).convert('RGB')
        r_rgb = Image.open(r_rgb_path).convert('RGB')

        data['left_img'] = l_rgb
        data['right_img'] = r_rgb

        if self.split == 'kitti':
            return data

        elif self.split == 'eigen':
            cam_path = self._check_path(item_files['cam_intrin'], "Panic::Cannot find Camera Info")
            depth_path = self._check_path(item_files['depth'], "Panic::Cannot find depth file")
            w, h = l_rgb.size
            focal_length, baseline = get_focal_length_baseline(cam_path, cam=self.cam)
            depth, depth_interp = get_depth(cam_path, depth_path, [h,w], cam=self.cam, interp=True, vel_depth=True)
            data['focal_length'] = focal_length
            data['baseline'] = baseline
            data['depth'] = depth
            data['depth_interp'] = depth_interp
            data['fb'] = np.array(focal_length * baseline)
            return data
        else:
            logging.error('Unrecognized split in _read_data')

    def load_item(self, idx):
        # param idx - loads the data at idx
        item_files = self.files[idx]
        data_item = self._read_data(item_files)

        return data_item
