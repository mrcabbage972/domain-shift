import torch
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Function


DOWNLOAD_DATA = True

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None  # reverse gradient

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class Model(nn.Module):
    def __init__(self, lambd_grl):
        super(Model, self).__init__()
        # shared feature extractor
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)

        # 64 x 14 x 14 = 12544 after conv+pool
        self.fc1 = nn.Linear(12544, 128)

        # digit classifier head (10 classes)
        self.fc_digit = nn.Linear(128, 10)

        # domain classifier head (2 domains: 0 = MNIST, 1 = SVHN)
        self.fc_domain = nn.Linear(128, 2)

        self.dropout2 = nn.Dropout(0.5)

        self.lambd_grl = lambd_grl 

    def extract_features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)           # [B, 12544]
        x = self.fc1(x)                   # [B, 128]
        x = F.relu(x)
        return x

    def forward(self, x):
        """
        lambd_grl = 0.0  → no gradient reversal (baseline / simple domain confusion by -loss)
        lambd_grl > 0.0  → use gradient reversal for the domain head
        """
        features = self.extract_features(x)       # shared features [B, 128]

        # digit classification head
        digit_logits = self.fc_digit(self.dropout2(features))
        digit_log_probs = F.log_softmax(digit_logits, dim=1)

        # domain classification head
        if self.lambd_grl > 0.0:
            rev_features = grad_reverse(features, lambd_grl)
        else:
            rev_features = features

        domain_logits = self.fc_domain(rev_features)
        domain_log_probs = F.log_softmax(domain_logits, dim=1)

        return digit_log_probs, domain_log_probs


def train(log_interval, model, device, train_loader, optimizer, epoch, dry_run):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        if dry_run:
            return


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train_stage1(model):
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, mnist_train_loader, optimizer, epoch, dry_run)
        #test(model, device, mnist_test_loader)
        scheduler.step()

    return model

def train_stage2(model):
    pass
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    # for epoch in range(1, epochs + 1):
    #     train(log_interval, model, device, mnist_train_loader, optimizer, epoch, dry_run)
    #     #test(model, device, mnist_test_loader)
    #     scheduler.step()

    # return model

if __name__ == '__main__':
    batch_size = 65
    test_batch_size = 1000
    epochs = 1
    lr = 1.0
    gamma = 0.7
    log_interval = 10
    device = "cpu"
    dry_run = True


    transform=transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Resize([32, 32])
        ])

    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=DOWNLOAD_DATA, transform=transform)
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)

    mnist_test = torchvision.datasets.MNIST(root="./data", train=True, download=DOWNLOAD_DATA, transform=transform)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_train, batch_size=test_batch_size, shuffle=False, num_workers=2)

    base_model = Model(lambd_grl=0.0)
    
    train_stage1(base_model)

    #train_stage2(model)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
