from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torch
from PIL import Image
import glob
from numpy import random
import torchvision.transforms.functional as TF
from tqdm import tqdm


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, p, angles):
        self.angles = angles

    def __call__(self, x, angle_idx):
        if angle_idx:
            x = TF.rotate(x, int(self.angles[angle_idx]))
        return x


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
            pre_transform (bool): Apply image transformation before starting training
                                  (only when using caching)
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

        # Mean and std are calculated from the train dataset
        self.normalize = transforms.Normalize(mean=[0.2099, 0.2098, 0.2098],
                                              std=[0.7148, 0.7148, 0.7148])

        self.transform = self.get_transforms()
        self.rotation_transform = RotationTransform(p, angles=[90, 180, 270])

        if self.use_cache:
            self.cache()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
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

            if self.train:
                angle_idx = None
                if self.p < random.random():
                    angle_idx = random.choice([0, 1 , 2])
                image = self.rotation_transform(self.transform(image), angle_idx)
                label = self.rotation_transform(self.transform(label), angle_idx)
            else:
                image = self.transform(image)
                label = self.transform(label)

            sample = {"img": image,
                      "label": label}

            return sample

    def get_train_transforms(self, p=0.5, angle_idx=None):
        rotation_transform = RotationTransform(p, angles=[90, 180, 270])

        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            rotation_transform,
            self.normalize
        ])

        return train_transform

    def get_transforms(self):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            self.normalize
        ])

        return transform

    def cache(self):
        progressbar = tqdm(range(len(self.img_list)), desc="Caching")
        for i, data in zip(progressbar, self.img_list):
            img_path, label_path = data

            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('RGB')

            if self.train:
                angle_idx = None
                if self.p < random.random():
                    angle_idx = random.choice([0, 1 , 2])
                image = self.rotation_transform(self.transform(image), angle_idx)
                label = self.rotation_transform(self.transform(label), angle_idx)
            else:
                image = self.transform(image)
                label = self.transform(label)
            self.cached_data.append((image, label))
