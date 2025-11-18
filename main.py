import torch
import torchvision

DOWNLOAD_DATA = True



if __name__ == '__main__':
    batch_size = 16

    resize = torchvision.transforms.Resize([32, 32])

    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=DOWNLOAD_DATA)
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, transform=resize)

    mnist_test = torchvision.datasets.MNIST(root="./data", train=True, download=DOWNLOAD_DATA)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, transform=resize)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
