from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torch
from PIL import Image, ImageEnhance
import glob
from numpy import random
import torchvision.transforms.functional as TF
from tqdm import tqdm


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class CovidDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, train=False, unet=None, overlay=False,
                 p=0.5, use_cache=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            txt_COVID (string): Path to the txt file containing the datasplit's COVID positive image paths.
            txt_NonCOVID (string): Path to the txt file containing the datasplit's COVID negative image paths.
            train (bool): Apply training transforms if true.
            unet (class UNet): Trained UNet for lung segmentation.
            overlay (bool): If true overlay ROI mask will be applied. Otherwise last image's last channel
                            will be replaced with ROI mask.
            p (float): Probability for data augmentations.
            use_cache (bool): Cache data in main memory before starting the training.
        """
        self.root_dir = root_dir
        self.txt_path = [txt_NonCOVID, txt_COVID]
        self.classes = ['NonCOVID', 'COVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        self.overlay = overlay
        self.use_cache = use_cache
        self.p = p
        self.cached_data = []
        self.unet = unet

        for c in range(self.num_cls):
            cls_list = [[os.path.join(r + "images/", self.classes[c], item), c]
                        for r in self.root_dir for item in read_txt(r + self.txt_path[c])]
            self.img_list += cls_list

        if train:
             self.transform = self.get_train_transforms()
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
             sample = {'img': image,
                       'label': int(label)}
             return sample
        else:
            img_path = self.img_list[idx][0]
            image = Image.open(img_path).convert('RGB')

            sample = {'img': self.transform(image),
                      'label': int(self.img_list[idx][1])}
            return sample

    def get_train_transforms(self):
        """
        Creates composed transformation function for training data.

        Returns:
            test_transform (ComposeDouble): Composed function performing
                                            required transformation for
                                            training data
        """
        train_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(self.p),
            transforms.RandomVerticalFlip(self.p),
            self.rotation_transform,
            # self.normalize,
            self.normalize_to_range_0_1,
            self.mask_transform
        ])

        return train_transform

    def get_test_transforms(self):
        """
        Creates composed transformation function for test/validation data.

        Returns:
            test_transform (ComposeDouble): Composed function performing
                                            required transformation for
                                            test/validation data
        """
        test_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            self.normalize_to_range_0_1
            # self.normalize
        ])

        return test_transform

    def rotation_transform(self, x):
        """
        Rotate by one of the defined angles with probability p.

        Args:
            x (tensor): Tensor to be rotated.

        Returns:
            A (tensor): Rotated tensor.
        """
        angles = [90, 180, 270]

        if self.p < random.random():
            angle = random.choice(angles)
            x = TF.rotate(x, int(angle))
        return x

    def mask_transform(self, x):
        """
        Set pixels outside of mask to zero.
        Overlay: set pixels outside of mask to min value.
        Replace channel: replace last channel with mask values.

        Args:
            x (tensor): Tensor to be masked.

        Returns:
            x (tensor): Masked tensor.
        """
        if self.unet:
            threshold = 0.5
            mask = self.unet(x[None,:,:,:])
            mask = self.normalize_to_range_0_1(mask)
            zero = torch.zeros_like(mask, dtype=torch.long)
            one = torch.ones_like(mask, dtype=torch.long)
            mask = torch.where(mask >= threshold, one, zero)
            if self.overlay:
                x[mask[0,:,:,:].repeat(3,1,1) == 0] = torch.min(x)
            else:
                x[-1,:,:] = mask[0,0,:,:]
        return x

    def normalize(self, A):
        """
        Normalize tensor with precomputed (training dataset) mean and std.

        Args:
            A (tensor): Tensor to be normalized.

        Returns:
            A (tensor): Normalized tensor.
        """
        _normalize = transforms.Normalize(mean=[0.2099, 0.2098, 0.2098],
                                          std=[0.7148, 0.7148, 0.7148])

        return _normalize(A)

    def normalize_to_range_0_1(self, A):
        """
        Normalize array to values within range [0, 1].

        Args:
            A (tensor): Tensor with values in range [0, 255].

        Returns:
            B (tensor): Tensor with values in range [0, 1].
        """
        B = A - A.min()
        B /= B.max()

        return B

    def cache(self):
        """
        Caches data in RAM for faster access during training and testing.
        Can require a lot of free RAM (depending on the dataset size)!
        """
        progressbar = tqdm(range(len(self.img_list)), desc="Caching")
        for i, data in zip(progressbar, self.img_list):
            img_path, label = data
            image = Image.open(img_path).convert('RGB')

            self.cached_data.append((self.transform(image), label))
