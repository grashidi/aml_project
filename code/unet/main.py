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
from util.train_util import fit, test, DiceLoss


if __name__ == "__main__":
    # adpated from https://www.kaggle.com/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset
    BATCH_SIZE = 16
    NUM_EPOCHS = 20

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
    # for batch_index, batch_samples in enumerate(train_loader):
    #     im, labels = batch_samples['img'], batch_samples['label']
    #     plt.figure()
    #     c1 = plt.imshow(im[0,1,:,:].numpy(), alpha=1.0)
    #     plt.colorbar(c1)
    #     plt.savefig("test_" + str(batch_index) + "_im.png")
    #     plt.figure()
    #     c2 = plt.imshow(labels[0,0,:,:].numpy(), alpha=1.0)
    #     plt.colorbar(c2)
    #     plt.savefig("test_" + str(batch_index) + "_label.png")
    #
    #     if batch_index > 3:
    #         break

    # # compute mean and std for dataset
    # mean = 0.
    # std = 0.
    # for batch_samples in train_loader:
    #     images, labels = batch_samples['img'], batch_samples['label']
    #     samples = images.size(0) # batch size (the last batch can have smaller size!)
    #     images = images.view(samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    #     std += images.std(2).sum(0)
    #
    # mean /= len(train_loader.dataset)
    # std /= len(train_loader.dataset)
    #
    # print(mean, std)

    # unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #     in_channels=3, out_channels=1, init_features=32, pretrained=True)

    unet = smp.Unet(encoder_name="se_resnext50_32x4d",
                    encoder_weights="imagenet",
                    classes=1,
                    activation=None)

    # train...
    if not os.path.exists("model_backup/"):
        os.makedirs("model_backup/")

    time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    stats_path = "model_backup/stats_unet_e{}_bs{}_{}".format(NUM_EPOCHS,
                                                              BATCH_SIZE,
                                                              time)

    optimizer = optim.Adam(unet.parameters(), lr=2e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                  mode='min',
    #                                                  factor=0.5,
    #                                                  patience=3,
    #                                                  threshold=0.0001,
    #                                                  cooldown=2,
    #                                                  min_lr=1e-6,
    #                                                  verbose=1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[3,5,6,7,8,9,10,11,13,15],
                                                     gamma=0.75)
    criterion = DiceLoss()

    fit(unet, optimizer, scheduler, criterion, train_loader, val_loader, NUM_EPOCHS, stats_path)
    test(unet, criterion, test_loader)

    torch.save(unet.state_dict(),
               "model_backup/unet_e{}_bs{}_{}.pt".format(NUM_EPOCHS,
                                                         BATCH_SIZE,
                                                         time))
