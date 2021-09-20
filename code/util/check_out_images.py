from matplotlib import pyplot as plt

def check_out_images_resnet(data_loader, num_images=20):
    """
    Plots num_images of the given dataloader.

    Args:
        data_loader (DataLoader class): Dataloader containing x-rays or CT-scans
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


def check_out_images_unet(data_loader, num_images=20):
    """
    Plots num_images of the given dataloader.

    Args:
        data_loader (DataLoader class): Dataloader containing x-rays or CT-scans
        num_images: (int): Number of images to be plotted.
    """
    for batch_index, batch_samples in enumerate(data_loader):
        im, labels = batch_samples['img'], batch_samples['label']
        plt.figure()
        c1 = plt.imshow(im[0,0,:,:].numpy(), alpha=1.0)
        plt.colorbar(c1)
        plt.savefig("test_" + str(batch_index) + "_im.png")
        plt.figure()
        c2 = plt.imshow(labels[0,0,:,:].numpy(), alpha=1.0)
        plt.colorbar(c2)
        plt.savefig("test_" + str(batch_index) + "_label.png")

        if batch_index > (num_images - 2):
            break
