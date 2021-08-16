from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torch
from PIL import Image
import glob
import numpy as np
from numpy import random
import torchvision.transforms.functional as TF
from tqdm import tqdm
from typing import List, Callable, Tuple
from functools import partial


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, input: bool = True, target: bool = False, *args, **kwargs):
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, inp, tar):
        if self.input: inp = self.function(inp)
        if self.target: tar = self.function(tar)

        return inp, tar


class AugmentationDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, *args, **kwargs):
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp, tar):
        inp, tar = self.function(inp, tar)
        return inp, tar


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, txt_images, txt_masks, train=False, p=0.5, use_cache=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            txt_images (string): Path to the txt file containing the datasplit's image paths.
            txt_masks (string): Path to the txt file containing the datasplit's masks paths.
            train (bool): Apply training transforms if true.
            p (float): Probability for data augmentations.
            use_cache (bool): Cache data in main memory before starting the training
        """
        self.root_dir = root_dir
        self.txt_path = [txt_images, txt_masks]
        self.train = train
        self.p = p
        self.img_list = []
        self.use_cache = use_cache
        self.cached_data = []

        im_p, ma_p = self.txt_path
        self.img_list = [[os.path.join(r, "images", im), os.path.join(r, "masks", ma)]
                          for r in self.root_dir for im, ma in zip(read_txt(r + im_p), read_txt(r + ma_p))]

        if train:
            self.transform = self.get_train_transforms(p=self.p)
        else:
            self.transform = self.get_test_transforms()

        if self.use_cache:
            self.cache()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index idx.
        If caching is enabled the cached data will be used. Otherwise the data
        has to be retrieved from the disk.

        Returns:
            sample (dict): Dictionary with training data ("img") and ground
                           truth ("label").
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.use_cache:
             image, label = self.cached_data[idx]
             sample = {"img": image,
                       "label": label}
             return sample
        else:
            image = Image.open(self.img_list[idx][0]).convert("RGB")
            label = Image.open(self.img_list[idx][1]).convert("RGB")

            image, label = self.transform(image, label)

            sample = {"img": image,
                      "label": label}
            return sample

    def get_train_transforms(self, p=0.5):
        """
        Creates composed transformation function for training data

        Returns:
            test_transform (ComposeDouble): Composed function performing
                                            required transformation for
                                            training data
        """
        train_transform = ComposeDouble([
            FunctionWrapperDouble(self.resize, input=True, target=True, dim=(256, 256)),
            FunctionWrapperDouble(self.to_tensor, input=True, target=True),
            # FunctionWrapperDouble(normalize, input=True, target=True),
            AugmentationDouble(self.random_rotate, p=p),
            FunctionWrapperDouble(self.create_binary_label, input=False, target=True),
            FunctionWrapperDouble(self.normalize_to_range_0_1, input=True, target=False)
        ])

        return train_transform

    def get_test_transforms(self):
        """
        Creates composed transformation function for test/validation data

        Returns:
            test_transform (ComposeDouble): Composed function performing
                                            required transformation for
                                            test/validation data
        """
        test_transform = ComposeDouble([
            FunctionWrapperDouble(self.resize, input=True, target=True, dim=(256, 256)),
            FunctionWrapperDouble(self.to_tensor, input=True, target=True),
            # FunctionWrapperDouble(normalize, input=True, target=True),
            FunctionWrapperDouble(self.create_binary_label, input=False, target=True),
            FunctionWrapperDouble(self.normalize_to_range_0_1, input=True, target=False)
        ])

        return test_transform

    def resize(self, image, dim):
        """
        Resize input image to given dimensions

        Args:
            image (PIL.Image): Image to be resized
            dim (tuple): Integer tuple with dimensions

        Returns:
            A (tensor): Resized image
        """
        _resize = transforms.Resize(dim)

        return _resize(image)


    def to_tensor(self, A):
        """
        Convert image to tensor

        Args:
            A (PIL.Image): Image to be converted

        Returns:
            A (tensor): Converted image
        """
        _to_tensor = transforms.ToTensor()

        return _to_tensor(A)


    def normalize(self, A):
        """
        Normalize tensor with precomputed mean and std

        Args:
            A (tensor): Tensor to be normalized

        Returns:
            A (tensor): Normalized tensor
        """
        _normalize = transforms.Normalize(mean=[0.5833, 0.5833, 0.5833],
                                          std=[0.2649, 0.2649, 0.2649])

        return _normalize(A)


    def normalize_to_range_0_1(self, A):
        """
        Normalize array to values within range [0, 1]

        Args:
            A (tensor): Tensor with values in range [0, 255]

        Returns:
            B (tensor): Tensor with values in range [0, 1]
        """
        B = A - A.min()
        B /= B.max()

        return B


    def create_binary_label(self, label):
        """
        Create a binary ground truth mask.
        1 pixel is contained in mask, 0 pixel is not contrained in mask.

        Args:
            label (tensor): Tensor with values in range [0, 255]

        Returns:
            mask (tensor): Binary tensor with values 1 or 0.
        """
        zero = torch.zeros_like(label, dtype=torch.long)
        one = torch.ones_like(label, dtype=torch.long)
        mask = torch.where(label != 0, one, zero)

        return mask[None,0,:,:]


    def random_rotate(self, image, label, p):
        """
        Randomly rotates image and label by the same degree with given probability p

        Args:
            image (tensor): input image
            label (tensor): target mask
            p (float): Probability to rotate ([0., 1.])

        Returns:
            image (tensor): Randomly rotated image
            label (tensor): Randomly rotated label
        """
        angles=[90, 180, 270]

        if p < random.random():
            angle_idx = random.choice([0, 1 , 2])

            image = TF.rotate(image, int(angles[angle_idx]))
            label = TF.rotate(label, int(angles[angle_idx]))

        return image, label

    def cache(self):
        """
        Caches data in RAM for faster access during training and testing.
        Can require a lot of free RAM (depending on the dataset size)!
        """
        progressbar = tqdm(range(len(self.img_list)), desc="Caching")
        for i, data in zip(progressbar, self.img_list):
            img_path, label_path = data

            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('RGB')

            self.cached_data.append((self.transform(image, label)))
