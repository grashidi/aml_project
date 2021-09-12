import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from collections import OrderedDict


def fit(model, optimizer, scheduler, criterion, train_loader, val_loader,
        epochs, stats_path, additional_stats_enabled=False):
    """
    Trains the given model

    Args:
        model (model class): Model to be trained.
        optimizer (optimizer class): Optimizer to be used.
        scheduler (scheduler class): Scheduler to be used.
        criterion (loss class): Loss to be used.
        train_loader (data loader class): Data loader containing trainig data.
        val_loader (data loader class):  Data loader containing validation data.
        epochs (int): Number epochs.
        stats_path (string): Path for writing statistics to file.
        additional_stats_enabled (bool): If true precision, recall, accuracy,
                                         AUC, F1 are computed and logged.
                                         Turn off those statistics for unet training!
    """
    model, device = move_model_to_device(model)

    vote_num = len(train_loader) // 10
    vote_num = vote_num if vote_num > 0 else 1

    train_loss = 0.
    val_loss = 0.
    precision = 0.
    recall = 0.
    acc = 0.
    auc = 0.

    stat_vars = OrderedDict()
    stat_vars["epoch"] = 1
    stat_vars["batch_number"] = 1
    stat_vars["train_loss"] = 0.
    stat_vars["val_loss"] = 0.

    if additional_stats_enabled:
        additional_stats = ["precision", "recall", "accuracy", "AUC", "F1"]
        for stat in additional_stats: stat_vars[stat] = 0.

    stats_history = {stat: [] for stat in stat_vars.keys()}

    for e in range(epochs):
        progressbar = tqdm(range(len(train_loader)), desc='Training...')
        progressbar.set_postfix(stat_vars)

        train_loss = 0.0
        val_loss = 0.0

        model.train()

        for batch_index, batch_samples in zip(progressbar, train_loader):
            images, labels = batch_samples["img"].to(device), batch_samples["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # evaluate
            if (batch_index + 1) % vote_num == 0:
                target_list, score_list, pred_list, curr_val_loss = validate(model,
                                                                             criterion,
                                                                             val_loader,
                                                                             device,
                                                                             additional_stats_enabled)

                val_loss += curr_val_loss
                if additional_stats_enabled:
                    precision, recall, acc, auc, f1 = compute_statistics(pred_list, score_list, target_list)
                    stat_vars["precision"] = precision
                    stat_vars["recall"] = recall
                    stat_vars["accuracy"] = acc
                    stat_vars["AUC"] = auc
                    stat_vars["F1"] = f1

                stat_vars["epoch"] = e + 1
                stat_vars["batch_number"] = batch_index + 1
                stat_vars["train_loss"] = train_loss / vote_num
                stat_vars["val_loss"] = val_loss.item() / ((batch_index + 1) / vote_num)

                progressbar.set_postfix(stat_vars)

                for stat, val in stat_vars.items(): stats_history[stat].append(val)

                train_loss = 0.0

        val_loss = val_loss.item() / (len(train_loader) / vote_num)
        scheduler.step(val_loss)

    with open(stats_path, "w") as f:
        json.dump(stats_history, f)


def validate(model, criterion, val_loader, device, additional_stats_enabled):
    """
    Validates the given model

    Args:
        model (model class): Model to be trained.
        criterion (loss class): Loss to be used.
        val_loader (data loader class):  Data loader containing validation data.
        device (string): Device to be used.
        additional_stats_enabled (bool): If true target_list, score_list and
                                         pred_list are computed.

    Returns:
        target_list (list): list containig the true labels
                            (0 for covid negative label)
                            (1 for covid positive label)
        score_list (list): list containing the softmax values of the predicted logits
        pred_list (list): list containig the model's predictions
                          (0 for covid negative prediction)
                          (1 for covid positive prediction)
        val_loss (float): average validation loss

    """
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        pred_list = []
        target_list = []
        score_list = []

        for batch_index, batch_samples in enumerate(val_loader):
            images, labels = batch_samples["img"].to(device), batch_samples["label"].to(device)

            output = model(images)

            val_loss += criterion(output, labels.long())

            if additional_stats_enabled:
                score = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(labels.long().view_as(pred)).sum().item()
                targetcpu = [l.item() for l in labels.long().cpu()]

                pred_list += [p.item() for p in pred.cpu()]
                score_list += [s.item() for s in score.cpu()[:,1]]
                target_list += targetcpu

    return target_list, score_list, pred_list, (val_loss / len(val_loader))


def test(model, criterion, test_loader, additional_stats_enabled=False):
    """
    Tests the given model

    Args:
        model (model class): Model to be trained.
        criterion (loss class): Loss to be used.
        test_loader (data loader class):  Data loader containing test data.
        additional_stats_enabled (bool): If true precision, recall, accuracy,
                                         AUC, F1 are computed. Turn off those
                                         statistics for unet testing!
    """
    model.eval()

    model, device = move_model_to_device(model)

    vote_num = len(test_loader) // 10
    vote_num = vote_num if vote_num > 0 else 1

    test_loss = 0.
    correct = 0

    with torch.no_grad():
        # see validation returns
        pred_list = []
        target_list = []
        score_list = []

        stat_vars = OrderedDict()
        stat_vars["batch_number"] = 1
        stat_vars["test_loss"] = 0.

        if additional_stats_enabled:
            additional_stats = ["precision", "recall", "accuracy", "AUC", "F1"]
            for stat in additional_stats: stat_vars[stat] = 0.

        progressbar = tqdm(range(len(test_loader)), desc='Testing...')
        progressbar.set_postfix(stat_vars)


        for batch_index, batch_samples in zip(progressbar, test_loader):
            images, labels = batch_samples["img"].to(device), batch_samples["label"].to(device)

            output = model(images)

            test_loss += criterion(output, labels.long())

            if additional_stats_enabled:
                score = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(labels.long().view_as(pred)).sum().item()
                targetcpu = [l.item() for l in labels.long().cpu()]

                pred_list += [p.item() for p in pred.cpu()]
                score_list += [s.item() for s in score.cpu()[:,1]]
                target_list += targetcpu

            if (batch_index + 1) % vote_num == 0:
                if additional_stats_enabled:
                    precision, recall, acc, auc, f1 = compute_statistics(pred_list, score_list, target_list)
                    stat_vars["precision"] = precision
                    stat_vars["recall"] = recall
                    stat_vars["accuracy"] = acc
                    stat_vars["AUC"] = auc
                    stat_vars["F1"] = f1

                stat_vars["batch_number"] = batch_index + 1
                stat_vars["test_loss"] = test_loss.item() / vote_num

                progressbar.set_postfix(stat_vars)

                test_loss = 0.


def compute_statistics(pred_list, score_list, target_list):
    """
    Computes precision, recall, accuracy and AUC (area under the curve)

    Args:
        pred_list (list): list containig the model's predictions
                          (0 for covid negative prediction)
                          (1 for covid positive prediction)
        score_list (list): list containing the softmax values of the predicted logits
        target_list (list): list containig the true labels
                            (0 for covid negative label)
                            (1 for covid positive label)
    Returns:
        precision (float): precision computed from the given inputs
        recall (float): recall computed from the given inputs
        acc (float): accuracy computed from the given inputs
        auc (float): AUC computed from the given inputs
        f1 (float): F1 score compute from given inputs
    """
    pred_list = np.array(pred_list, dtype=np.float32)
    score_list = np.array(score_list, dtype=np.float32)
    target_list = np.array(target_list, dtype=np.float32)

    TP = ((pred_list == 1) & (target_list == 1)).sum()
    TN = ((pred_list == 0) & (target_list == 0)).sum()
    FN = ((pred_list == 0) & (target_list == 1)).sum()
    FP = ((pred_list == 1) & (target_list == 0)).sum()

    precision = (TP / (TP + FP)) if (TP + FP) > 0. else 0.
    recall = (TP / (TP + FN)) if (TP + FN) > 0. else 0.
    acc = (TP + TN) / (TP + TN + FP + FN)
    auc = roc_auc_score(target_list, score_list)
    f1 = 2 * recall * precision / (recall + precision)

    return precision, recall, acc, auc, f1


def move_model_to_device(model):
    """
    Checks out available devices and moves model to devices

    Args:
        model (model class): model
    Returns:
        model (model class): model at device
        device (string): device
    """
    device = "cuda"

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Devices in use:")
        for dc in range(device_count):
            print(torch.cuda.get_device_name(device=dc))
        model = nn.DataParallel(model, device_ids=list(range(device_count)))
    elif device_count == 1:
        print("Device in use:")
        print(torch.cuda.get_device_name(device=0))
    else:
        print("No acclerator available. Using CPU.")
        device = "cpu"

    model.to(device)

    return model, device


# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
