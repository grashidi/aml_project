from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torch
from PIL import Image
import glob
from numpy import random
import torchvision.transforms.functional as TF


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, p, angles):
        self.p = p
        self.angles = angles

    def __call__(self, x):
        if self.p < random.random():
            angle = random.choice(self.angles)
            x = TF.rotate(x, int(angle))
        return x


class CovidDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, train=False, p=0.5):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            txt_COVID (string): Path to the txt file with annotations for COVID positive images.
            txt_NonCOVID (string): Path to the txt file with annotations for COVID negative images.
            train (bool): Apply training transforms if true.
            p (float): Probability for data augmentations.

        """
        self.root_dir = root_dir
        self.txt_path = [txt_NonCOVID, txt_COVID]
        self.classes = ['NonCOVID', 'COVID']
        self.num_cls = len(self.classes)
        self.img_list = []

        for c in range(self.num_cls):
            cls_list = [[os.path.join(r + "images/", self.classes[c], item), c]
                        for r in self.root_dir for item in read_txt(r + self.txt_path[c])]
            self.img_list += cls_list

        # Mean and std are calculated from the train dataset
        self.normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                              std=[0.33165374, 0.33165374, 0.33165374])

        if train:
             self.transform = self.get_train_transforms(p)
        else:
            self.transform = self.get_test_transforms(p)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample

    def get_train_transforms(self, p=0.5):
        rotation_transform = RotationTransform(p, angles=[0, 90, 180, 270])

        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p),
            transforms.RandomVerticalFlip(p),
            rotation_transform,
            self.normalize
        ])

        return train_transform

    def get_test_transforms(self, p=0.5):
        test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            self.normalize
        ])

        return test_transform