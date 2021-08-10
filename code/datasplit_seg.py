import torch
from torch.utils.data import random_split
import os
from os import listdir
from os.path import isfile, join
from math import ceil

root_dir = "../data/segmentation/"
images = "images/"
masks = "masks/"

im_paths = [i_p for i_p in os.listdir(root_dir + images)]
ma_paths = [m_p for m_p in os.listdir(root_dir + masks)]

# sort to align images and masks
sorted_im_paths = sorted(im_paths)
sorted_ma_paths = sorted(ma_paths)

# pack into tuple
data = [(i_p, m_p) for i_p, m_p in zip(sorted_im_paths, sorted_ma_paths)]

# generate split
num_total = len(data)
num_train = ceil((len(data)/10)*7)
num_val = ceil((len(data)/10)*1)
num_test = num_total - (num_train + num_val)

split = random_split(data, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(42))

# write to file
for i, im_ma in enumerate([images, masks]):
    if not os.path.exists(root_dir + "data_split/" + im_ma):
        os.makedirs(root_dir + "data_split/" + im_ma)

    for sp, phase, num in zip(split, ["train.txt", "val.txt", "test.txt"], [num_train, num_val, num_test]):
        with open(root_dir + "data_split/" + im_ma + phase, "w") as f:
            for path in sp:
                f.write(path[i] + "\n")
            print(im_ma + phase, "number of images", num)
