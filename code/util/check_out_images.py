from matplotlib import pyplot as plt

def check_out_images_resnet(data_loader, num_images=20):
    """
    Plots num_images of the given dataloader.

    Args:
        data_loader (DataLoader class): Data loader containing x-rays or CT-scans
        num_images: (int): Number of images to be plotted.
    """
    for batch_index, batch_samples in enumerate(data_loader):
        im, labels = batch_samples['img'], batch_samples['label']
        plt.figure()
        c1 = plt.imshow(im[0,0,:,:].numpy(), alpha=1.0)
        plt.colorbar(c1)
        plt.savefig("test_" + str(batch_index) + ".png")

        if batch_index > (num_images - 2):
            break
