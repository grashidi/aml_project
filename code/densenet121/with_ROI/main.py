import os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import  DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
from datetime import datetime
import segmentation_models_pytorch as smp
from util.covid_dataset import CovidDataset
from util.train_util import fit, test


if __name__ == "__main__":
    # Adpated from Deep CNN models for predicting COVID-19 in CT and x-ray images
    BATCH_SIZE = 10
    NUM_EPOCHS = 10

    USE_CACHE = True # Make sure you have enough RAM available

    root_dir = ["../../../data/ct_scan/", "../../../data/xray/"]
    txt_COVID = "data_split/COVID/"
    txt_NonCOVID = "data_split/NonCOVID/"

    # load trained unet
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    unet.load_state_dict(torch.load("unet_e10_bs16_09-09-2021_10:17:50.pt"))

    unet.eval()

    trainset = CovidDataset(root_dir=root_dir,
                            txt_COVID=txt_COVID + "train.txt",
                            txt_NonCOVID=txt_NonCOVID + "train.txt",
                            unet=unet,overlay=True,
                            train=True,
                            use_cache=USE_CACHE)
    valset = CovidDataset(root_dir=root_dir,
                          txt_COVID=txt_COVID + "val.txt",
                          txt_NonCOVID=txt_NonCOVID + "val.txt",overlay=True,
                          use_cache=USE_CACHE)
    testset = CovidDataset(root_dir=root_dir,
                           txt_COVID=txt_COVID + "test.txt",
                           txt_NonCOVID=txt_NonCOVID + "test.txt",overlay=True,
                           use_cache=USE_CACHE)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)

 
    # load model pretrained on ImageNet
    densenet121 = models.densenet121(pretrained=True)

    # replace fully connected layer
    densenet121.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)

    # freeze all layers
    for param in densenet121.parameters():
        param.requires_grad = False

    # unfreeze last three layers of last block
    denseblock4 = densenet121.features.denseblock4
    for layer in [denseblock4.denselayer14,denseblock4.denselayer15,denseblock4.denselayer16]:
        for param in layer.parameters():
            param.requires_grad = True

    #train ...
    if not os.path.exists("model_backup/"):
        os.makedirs("model_backup/")

    time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    stats_path = "model_backup/stats_densenet121_e{}_bs{}_{}.json".format(NUM_EPOCHS,
                                                                       BATCH_SIZE,
                                                                       time)
    stats_test_path = "model_backup/stats_densenet121_e{}_bs{}_{}_test.json".format(NUM_EPOCHS,
                                                                     BATCH_SIZE,
                                                                     time)
    optimizer = optim.Adam(densenet121.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=1)
    criterion = nn.CrossEntropyLoss()

    fit(densenet121, optimizer, scheduler, criterion, train_loader, val_loader,
        NUM_EPOCHS, stats_path, additional_stats_enabled=True)
    test(densenet121, criterion, test_loader, stats_path = stats_test_path, additional_stats_enabled=True)

    torch.save(densenet121.state_dict(),
               "model_backup/densenet121_e{}_bs{}_{}.pt".format(NUM_EPOCHS,
                                                             BATCH_SIZE,
                                                             time))
