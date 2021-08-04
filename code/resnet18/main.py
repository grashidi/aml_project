import torchvision.models as models
import torch.nn as nn
from torch.utils.data import  DataLoader
from matplotlib import pyplot as plt
from covid_dataset import CovidDataset


if __name__ == "__main__":
    BATCH_SIZE = 10 # adpated from paper
    NUM_EPOCHS = 10 # adpated from paper

    root_dir = ["../../data/ct_scan/", "../../data/xray/"]
    txt_COVID = "data_split/COVID/"
    txt_NonCOVID = "data_split/NonCOVID/"

    trainset = CovidDataset(root_dir=root_dir,
                            txt_COVID=txt_COVID + "/train.txt",
                            txt_NonCOVID=txt_NonCOVID + "/train.txt",
                            train=True)
    valset = CovidDataset(root_dir=root_dir,
                          txt_COVID=txt_COVID + "/val.txt",
                          txt_NonCOVID=txt_NonCOVID + "/val.txt")
    testset = CovidDataset(root_dir=root_dir,
                           txt_COVID=txt_COVID + "/test.txt",
                           txt_NonCOVID=txt_NonCOVID + "/test.txt")

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)

    for batch_index, batch_samples in enumerate(train_loader):
        im, labels = batch_samples['img'], batch_samples['label']
        plt.imshow(im[0,1,:,:].numpy(), alpha=1.0)
        plt.savefig("test_" + str(batch_index) + ".png")

        if batch_index > 18:
            break


    # load model pretrained on ImageNet
    resnet18 = models.resnet18(pretrained=True)

    # replace fully connected layer
    resnet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    # freeze all layers
    for param in resnet18.parameters():
        param.requires_grad = False

    # unfreeze last two layers
    for layer in [resnet18.avgpool, resnet18.fc]:
        for param in layer.parameters():
            param.requires_grad = True

    #train ...
