import os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import  DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
from datetime import datetime
import segmentation_models_pytorch as smp
from util.segmentation_dataset import SegmentationDataset
from util.train_util import fit, test, DiceLoss, compute_mean_std
from util.check_out_images import check_out_images_unet


if __name__ == "__main__":
    # adpated from https://www.kaggle.com/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset
    BATCH_SIZE = 16
    NUM_EPOCHS = 10

    USE_CACHE = False

    root_dir = ["../../data/segmentation/xray/", "../../data/segmentation/ct_scan/"]
    txt_images = "data_split/images/"
    txt_masks = "data_split/masks/"

    trainset = SegmentationDataset(root_dir=root_dir,
                                   txt_images=txt_images + "train.txt",
                                   txt_masks=txt_masks + "train.txt",
                                   train=True,
                                   use_cache=USE_CACHE)
    valset = SegmentationDataset(root_dir=root_dir,
                                 txt_images=txt_images + "val.txt",
                                 txt_masks=txt_masks + "val.txt",
                                 use_cache=USE_CACHE)
    testset = SegmentationDataset(root_dir=root_dir,
                                  txt_images=txt_images + "test.txt",
                                  txt_masks=txt_masks + "test.txt",
                                  use_cache=USE_CACHE)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)

    # # check out some of the images
    # check_out_images_unet(train_loader, num_images=5)

    # print(compute_mean_std(train_loader))

    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)

    # train...
    if not os.path.exists("model_backup/"):
        os.makedirs("model_backup/")

    time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    stats_path = "model_backup/stats_unet_e{}_bs{}_{}.json".format(NUM_EPOCHS,
                                                              BATCH_SIZE,
                                                              time)

    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=1,
                                                     threshold=0.0001,
                                                     cooldown=2,
                                                     min_lr=1e-6,
                                                     verbose=1)

    criterion = DiceLoss()


    fit(unet, optimizer, scheduler, criterion, train_loader, val_loader, NUM_EPOCHS, stats_path)
    test(unet, criterion, test_loader)

    torch.save(unet.state_dict(),
               "model_backup/unet_e{}_bs{}_{}.pt".format(NUM_EPOCHS,
                                                         BATCH_SIZE,
                                                         time))
