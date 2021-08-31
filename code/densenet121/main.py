import os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import  DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
from datetime import datetime
from util.covid_dataset import CovidDataset
from util.train_util import fit, test

if __name__ == "__main__":
    BATCH_SIZE = 10 # adapted from paper
    NUM_EPOCHS = 10 # adapted from paper
    USE_CACHE = False # Make sure you have enough RAM available

    root_dir = ["data/ct_scan/", "data/xray/"]
    txt_COVID = "data_split/COVID/"
    txt_NonCOVID = "data_split/NonCOVID/"

    trainset = CovidDataset(root_dir=root_dir,
                            txt_COVID=txt_COVID + "train.txt",
                            txt_NonCOVID=txt_NonCOVID + "train.txt",
                            train=True,
                            use_cache=USE_CACHE)
    valset = CovidDataset(root_dir=root_dir,
                          txt_COVID=txt_COVID + "val.txt",
                          txt_NonCOVID=txt_NonCOVID + "val.txt",
                          use_cache=USE_CACHE)
    testset = CovidDataset(root_dir=root_dir,
                           txt_COVID=txt_COVID + "test.txt",
                           txt_NonCOVID=txt_NonCOVID + "test.txt",
                           use_cache=USE_CACHE)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)

    # # check out some of the images
    # for batch_index, batch_samples in enumerate(train_loader):
    #     im, labels = batch_samples['img'], batch_samples['label']
    #     plt.imshow(im[0,1,:,:].numpy(), alpha=1.0)
    #     plt.savefig("test_" + str(batch_index) + ".png")
    #
    #     if batch_index > 18:
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


    # load model pretrained on ImageNet
    densenet121 = models.densenet121(pretrained=True)

    # replace fully connected layer
    densenet121.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    # freeze all layers
    for param in densenet121.parameters():
        param.requires_grad = True

    # unfreeze last three layers
    #for layer in [densenet121.layer4, densenet121.avgpool, densenet121.fc]:
    #    for param in layer.parameters():
    #        param.requires_grad = True

    #train ...
    time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    stats_path = "model_backup/stats_densenet121_e{}_bs{}_{}.pt".format(NUM_EPOCHS,
                                                                     BATCH_SIZE,
                                                                     time)

    optimizer = optim.Adam(densenet121.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=1)
    criterion = nn.CrossEntropyLoss()

    fit(densenet121, optimizer, scheduler, criterion, train_loader, val_loader, NUM_EPOCHS, stats_path)
    test(densenet121, criterion, test_loader)

    if not os.path.exists("model_backup/"):
        os.makedirs("model_backup/")

    torch.save(densenet121.state_dict(),
               "model_backup/densenet121_e{}_bs{}_{}.pt".format(NUM_EPOCHS,
                                                             BATCH_SIZE,
                                                             time))
