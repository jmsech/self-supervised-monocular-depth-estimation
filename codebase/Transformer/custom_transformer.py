import torch
import numpy as np
import torchvision.transforms as transforms
import Transformer.custom_methods as augmethods

class CustTransformer():
    """
    An example of Custom Transformer.
    This class should work with custom transform methods which defined in custom_methods.py
    """
    def __init__(self, phase):
        self.phase = phase

    def get_joint_transform(self):
        if self.phase == "train":
            return transforms.Compose([augmethods.Scale("pair", [256, 512]),
                                       augmethods.RandomHorizontalFlip(),
                                       augmethods.RandomRotate()])
        else:
            return transforms.Compose([augmethods.TransToPIL()])

    def get_img_transform(self):
        if self.phase == "train":
            return transforms.Compose([augmethods.ImgAug(),
                                       augmethods.ToTensor("Img"),
                                       augmethods.ImgNormalize([.5, .5, .5], [.5, .5, .5])])
        else:
            return transforms.Compose([augmethods.Scale("Img", [256, 512]),
                                       augmethods.ToTensor("Img"),
                                       augmethods.ImgNormalize([.5, .5, .5], [.5, .5, .5])])

    def get_depth_transform(self):
        return transforms.Compose([augmethods.ToTensor("depth"), augmethods.Transfb()])

    def get_transform(self):
        """
        Total transform processing
        """
        joint_transform = self.get_joint_transform()
        img_transform = self.get_img_transform()
        if self.phase == 'train':
            return transforms.Compose([joint_transform, img_transform])
        return transforms.Compose([img_transform])