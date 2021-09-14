import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.covid_dataset import CovidDataset


def get_label_and_prediction(label, model_out):
    """
    Converts binary label into a string

    Args:
        label (int): 1 for COVID positive 0 for COIVD negative

    Returns:
    (string) The corresponding label.
    """
    label = "COVID_positive" if label else "COVID_negative"
    pred = "COVID_positive" if model_out.argmax(dim=1, keepdim=True)[0].item() else "COVID_negative"

    return label, pred


def get_activations(name):
    """
    Wrapper function for registering activation hook

    Args:
        label (string): Layer name.

    Returns:
    (function) Hook.
    """
    def hook(module, input, output):
    	activations[name] = output.detach()
    return hook


def get_grads(name):
    """
    Wrapper function for registering gradient hook

    Args:
        label (string): Layer name.

    Returns:
    (function) Hook.
    """
    def hook(module, grad_input, grad_output):
    	grads[name] = grad_output # is a tuple
    return hook




BATCH_SIZE = 1

device = "cuda"

model_name = "resnet18_e10_bs10_12-09-2021_10:46:17.pt"
model_path = "model_backup/" + model_name

root_dir = ["../../../data/ct_scan/", "../../../data/xray/"]
txt_COVID = "data_split/COVID/"
txt_NonCOVID = "data_split/NonCOVID/"

grad_cam_path = 'grad_cam/' + "_sum_" + model_path.split('/')[1][:-3]+'/'
if not os.path.exists(grad_cam_path):
    os.makedirs(grad_cam_path)

testset = CovidDataset(root_dir=root_dir,
                       txt_COVID=txt_COVID + "/test.txt",
                       txt_NonCOVID=txt_NonCOVID + "/test.txt")

test_loader = DataLoader(testset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)

resnet18 = models.resnet18(pretrained=False)
resnet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)

resnet18.load_state_dict(torch.load(model_path))

# register hooks in las convolutional layer
resnet18.layer4.register_forward_hook(get_activations("layer4"))
resnet18.layer4.register_full_backward_hook(get_grads("layer4"))

resnet18.eval()
resnet18.to(device)

for batch_index, batch_samples in tqdm(enumerate(test_loader)):
    activations = dict()
    grads = dict()

    images, labels = batch_samples["img"].to(device), batch_samples["label"].to(device)

    model_out = resnet18(images)
    model_out[:, model_out.argmax(dim=1)[0]].backward()

    activations = activations["layer4"]
    grads = grads["layer4"][0]

    grads = torch.mean(grads, dim=(-2, -1), keepdims=True) # gradients are pooled
    grad_cam = activations * grads
    # grad_cam = torch.mean(activations, dim=1).squeeze()
    grad_cam = torch.sum(grad_cam, dim=-3) # weighted sum
    grad_cam = nn.functional.relu(grad_cam) # relu'd weighted sum
    grad_cam /= grad_cam.max() # normalization
    # grad_cam = grad_cam[0] # squeezing the batch of one

    y, x =  get_label_and_prediction(labels.item(), model_out)

    plt.imshow(images[0,1,:,:].cpu().numpy(), alpha=1.0)
    plt.imshow(torchvision.transforms.ToPILImage()(grad_cam.cpu()).resize((256, 256),resample=Image.BILINEAR),
               cmap='jet', alpha=0.5)
    plt.title("Label: " + y + " / Predicted: " + x)
    plt.savefig(grad_cam_path + "predicted " + x + "_" + str(batch_index)+".png")
