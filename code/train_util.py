import torch
import torch.nn as nn


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
            val_loss += loss.item()
            if batch_index % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (e + 1, batch_index + 1, running_loss / 100))
                running_loss = 0.0

        print("loss: ", val_loss / len(train_loader))
        scheduler.step(val_loss)
