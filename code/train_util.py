import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def fit(model, optimizer, scheduler, criterion, train_loader, val_loader, epochs):
    device = "cuda"

    # Check out available devices
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

            # collect loss
            running_loss += loss.item()

            # evaluate every 100 minibatches
            if batch_index % vote_num == 99:
                target_list, score_list, pred_list, curr_val_loss = validate(model,
                                                                             criterion,
                                                                             val_loader,
                                                                             device)

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

        # print val loss and adjust learning rate according to val loss
        print("val_loss: ", val_loss.item() / (len(train_loader) / vote_num))
        scheduler.step(val_loss)


def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0

    TP = 0 # true positive
    TN = 0 # true negative
    FN = 0 # false negative
    FP = 0 # false positive

    # Don't update model
    with torch.no_grad():
        # list containing prediction of the model
        # 0 for covid negative prediction
        # 1 for covid positive prediction
        pred_list = []

        # list containing the true values
        # 0 for covid negative
        # 1 for covid positive
        target_list = []

        # list containing the softmax values of the predicted logits
        score_list = []


        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            images, labels = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(images)

            val_loss += criterion(output, labels.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(labels.long().view_as(pred)).sum().item()

            targetcpu = [l.item() for l in labels.long().cpu()]
            pred_list += [p.item() for p in pred.cpu()]
            score_list += [s.item() for s in score.cpu()[:,1]]
            target_list += targetcpu

    return target_list, score_list, pred_list, (val_loss / len(val_loader))


def test(model, criterion, test_loader):
    device = "cuda"

    # Check out available devices
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

    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0

    TP = 0 # true positive
    TN = 0 # true negative
    FN = 0 # false negative
    FP = 0 # false positive

    # Don't update model
    with torch.no_grad():
        # list containing prediction of the model
        # 0 for covid negative prediction
        # 1 for covid positive prediction
        pred_list = []

        # list containing the true values
        # 0 for covid negative
        # 1 for covid positive
        target_list = []

        # list containing the softmax values of the predicted logits
        score_list = []


        # Predict
        print("Testing ...")
        for batch_samples in tqdm(test_loader):
            images, labels = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(images)

            test_loss += criterion(output, labels.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(labels.long().view_as(pred)).sum().item()

            targetcpu = [l.item() for l in labels.long().cpu()]
            pred_list += [p.item() for p in pred.cpu()]
            score_list += [s.item() for s in score.cpu()[:,1]]
            target_list += targetcpu

        precision, recall, acc, auc = compute_statistics(pred_list, score_list, target_list)

        print(("test_loss: %.3f\t"
               "precision: %.3f\t"
               "recall: %.3f\t"
               "accuracy: %.3f\t"
               "AUC: %.3f") %
              (test_loss / len(test_loader),
               precision,
               recall,
               acc,
               auc))



def compute_statistics(pred_list, score_list, target_list):
    pred_list = np.asarray(pred_list)
    score_list = np.asarray(score_list)
    target_list = np.asarray(target_list)

    TP = ((pred_list == 1) & (target_list == 1)).sum()
    TN = ((pred_list == 0) & (target_list == 0)).sum()
    FN = ((pred_list == 0) & (target_list == 1)).sum()
    FP = ((pred_list == 1) & (target_list == 0)).sum()

    precision = (TP / (TP + FP)) if (TP + FP) > 0. else 0.
    recall = (TP / (TP + FN)) if (TP + FN) > 0. else 0.
    acc = (TP + TN) / (TP + TN + FP + FN)
    auc = roc_auc_score(target_list, score_list)

    return precision, recall, acc, auc
