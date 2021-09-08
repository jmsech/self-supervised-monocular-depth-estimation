import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


"""
This file defines some example transforms.
Each transform method is defined by using BaseMethod class
"""

class TransToPIL():
    """
    Transform method to convert images as PIL Image.
    """
    def __init__(self):
        self.to_pil = transforms.ToPILImage()

    def __call__(self, data_item):
        self.left_img = data_item['left_img']
        self.right_img = data_item['right_img']

        if not self._is_pil_image(self.left_img):
            data_item['left_img'] = self.to_pil(self.left_img)
        if not self._is_pil_image(self.right_img):
            data_item['right_img'] = self.to_pil(self.right_img)
        return data_item


class Scale():
    def __init__(self, mode, size):
        self.scale = transforms.Resize(size, Image.BILINEAR)
        self.mode = mode

    def __call__(self, data_item):
        self.left_img = data_item['left_img']
        self.right_img = data_item['right_img']

        if self.mode in ["pair", "Img"]:
            data_item['left_img'] = self.scale(self.left_img)
            data_item['right_img'] = self.scale(self.right_img)

        return data_item


class RandomHorizontalFlip():
    def __init__(self, mode=""):
        self.mode = mode

    def __call__(self, data_item):
        self.left_img = data_item['left_img']
        self.right_img = data_item['right_img']

        if random.random() < 0.5:
            data_item['left_img'] = self.left_img.transpose(Image.FLIP_LEFT_RIGHT)
            data_item['right_img'] = self.right_img.transpose(Image.FLIP_LEFT_RIGHT)

        return data_item


class RandomRotate():
    def __init__(self, mode=""):
        self.mode = mode

    @staticmethod
    def rotate_pil_func():
        degree = random.randrange(-500, 500)/100
        return (lambda pil, interp : F.rotate(pil, degree, interp))

    def __call__(self, data_item):
        self.left_img = data_item['left_img']
        self.right_img = data_item['right_img']

        if random.random() < 0.5:
            rotate_pil = self.rotate_pil_func()
            data_item['left_img'] = rotate_pil(self.left_img, Image.BICUBIC)
            data_item['right_img'] = rotate_pil(self.right_img, Image.BICUBIC)

        return data_item


class ImgAug():
    def __init__(self, mode=""):
        self.mode = mode

    @staticmethod
    def adjust_pil(pil):
        brightness = random.uniform(0.8, 1.0)
        contrast = random.uniform(0.8, 1.0)
        saturation = random.uniform(0.8, 1.0)

        pil = F.adjust_brightness(pil, brightness)
        pil = F.adjust_contrast(pil, contrast)
        pil = F.adjust_saturation(pil, saturation)

        return pil

    def __call__(self, data_item):
        self.left_img = data_item['left_img']
        self.right_img = data_item['right_img']

        data_item['left_img'] = self.adjust_pil(self.left_img)
        data_item['right_img'] = self.adjust_pil(self.right_img)

        return data_item


class ToTensor():
    def __init__(self, mode=""):
        self.mode = mode
        self.totensor = transforms.ToTensor()

    def __call__(self, data_item):
        self.left_img = data_item['left_img']
        self.right_img = data_item['right_img']

        if self.mode == "Img":
            data_item['left_img'] = self.totensor(self.left_img)
            data_item['right_img'] = self.totensor(self.right_img)
        if self.mode == "depth":
            data_item['depth'] = self.totensor(self.depth)
            data_item['depth_interp'] = self.totensor(self.depth_interp)

        return data_item


class ImgNormalize():
    def __init__(self, mean, std, mode=""):
        self.mode = mode
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, data_item):
        self.left_img = data_item['left_img']
        self.right_img = data_item['right_img']

        data_item['left_img'] = self.normalize(self.left_img)
        data_item['right_img'] = self.normalize(self.right_img)

        return data_item


class Transfb():
    def __init__(self, mode=""):
        self.mode = mode

    def __call__(self, data_item):
        self.left_img = data_item['left_img']
        self.right_img = data_item['right_img']

        data_item['fb'] = torch.from_numpy(self.fb)
        return data_item
