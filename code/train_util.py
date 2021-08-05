import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def fit(model, optimizer, scheduler, criterion, train_loader, val_loader, epochs):
    device = "cuda"
    device_count = torch.cuda.device_count()

    if device_count > 1:
        print("Devices in use:")
        for dc in device_count:
            print(torch.cuda.get_device_name(device=dc))
        model = nn.DataParallel(model)
    elif device_count == 1:
        print("Device in use:")
        print(torch.cuda.get_device_name(device=0))
    else:
        print("No acclerator available. Using CPU.")
        device = "cpu"

    model.to(device)

    vote_num = 100

    for e in range(epochs):
        running_loss = 0.0
        val_loss = 0.0
        for batch_index, batch_samples in enumerate(train_loader, 0):
            images, labels = batch_samples['img'].to(device), batch_samples['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # val_loss += loss.item()
            if batch_index % vote_num == 99:    # print every 100 mini-batches
                target_list, score_list, pred_list, curr_val_loss = validate(model, criterion, val_loader, device)

                val_loss += curr_val_loss
                precision, recall, acc, auc = compute_statistics(pred_list, score_list, target_list)

                print(("epoch: %d\t"
                       "batch_number: %5d\t"
                       "train_loss: %.3f\t"
                       "precision: %.3f\t"
                       "recall: %.3f\t"
                       "accuracy: %.3f\t"
                       "AUC: %.3f") %
                      (e + 1,
                       batch_index + 1,
                       running_loss / vote_num,
                       precision,
                       recall,
                       acc,
                       auc))

                running_loss = 0.0

        print("val_loss: ", val_loss.item() / (len(train_loader) / vote_num))
        scheduler.step(val_loss)


def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    results = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            images, labels = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(images)

            val_loss += criterion(output, labels.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(labels.long().view_as(pred)).sum().item()

            targetcpu = [l.item() for l in labels.long().cpu()]
            predlist += [p.item() for p in pred.cpu()]
            scorelist += [s.item() for s in score.cpu()[:,1]]
            targetlist += targetcpu

    return targetlist, scorelist, predlist, (val_loss / len(val_loader))


def compute_statistics(vote_pred, vote_score, targetlist):
    vote_pred = np.asarray(vote_pred)
    vote_score = np.asarray(vote_score)
    targetlist = np.asarray(targetlist)

    TP = ((vote_pred == 1) & (targetlist == 1)).sum()
    TN = ((vote_pred == 0) & (targetlist == 0)).sum()
    FN = ((vote_pred == 0) & (targetlist == 1)).sum()
    FP = ((vote_pred == 1) & (targetlist == 0)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = (TP + TN) / (TP + TN + FP + FN)
    auc = roc_auc_score(targetlist, vote_score)

    return precision, recall, acc, auc
