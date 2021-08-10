import torch
from torch.utils.data import random_split
import os
from os import listdir
from os.path import isfile, join
from math import ceil


root_dir = "../data/"
covid = "COVID/"
non_covid = "NonCOVID/"
xray = "xray/"
ct = "ct_scan/"

for medical_imaging_technique in [ct, xray]:
    for c in [covid, non_covid]:
        # collect paths
        path = root_dir + medical_imaging_technique + "images/" + c
        data = [f for f in listdir(path) if isfile(join(path, f))]

        # generate split
        num_total = len(data)
        num_train = ceil((len(data)/10)*7)
        num_val = ceil((len(data)/10)*1)
        num_test = num_total - (num_train + num_val)

        split = random_split(data, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(42))

        # write to file
        if not os.path.exists(root_dir + medical_imaging_technique + "data_split/" + c):
            os.makedirs(root_dir + medical_imaging_technique + "data_split/" + c)

        for sp, phase, num in zip(split, ["train.txt", "val.txt", "test.txt"], [num_train, num_val, num_test]):
            with open(root_dir + medical_imaging_technique + "data_split/" + c + phase, "w") as f:
                for path in sp:
                    f.write(path + "\n")
                print(medical_imaging_technique + c + phase, "number of images", num)
