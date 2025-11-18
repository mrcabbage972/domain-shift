import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Function

DOWNLOAD_DATA = True


# -------------------------
# Gradient Reversal Layer
# -------------------------

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


# -------------------------
# Model: feature extractor + two heads
# -------------------------

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # shared feature extractor (for 1-channel 32x32)
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

    def forward(self, x, lambd_grl=0.0):
        """
        lambd_grl = 0.0  → no gradient reversal (baseline / domain confusion via -loss)
        lambd_grl > 0.0  → use gradient reversal for the domain head
        """
        features = self.extract_features(x)       # shared features [B, 128]

        # digit classification head
        digit_logits = self.fc_digit(self.dropout2(features))
        digit_log_probs = F.log_softmax(digit_logits, dim=1)

        # domain classification head
        if lambd_grl > 0.0:
            rev_features = grad_reverse(features, lambd_grl)
        else:
            rev_features = features

        domain_logits = self.fc_domain(rev_features)
        domain_log_probs = F.log_softmax(domain_logits, dim=1)

        return digit_log_probs, domain_log_probs


# -------------------------
# Training loop (baseline + domain adaptation)
# -------------------------

def train(model,
          device,
          mnist_loader,
          optimizer,
          epoch,
          log_interval=10,
          use_domain_adaptation=False,
          svhn_loader=None,
          alpha=0.1,
          lambd_grl=0.0,
          beta=0.1,
          dry_run=False):
    """
    use_domain_adaptation = False -> baseline, train on MNIST only
    use_domain_adaptation = True  -> train on MNIST + SVHN (need svhn_loader)

    alpha     -> weight for domain confusion loss (if GRL is OFF)
    lambd_grl -> GRL strength; if > 0, use gradient reversal
    """
    model.train()

    if not use_domain_adaptation or svhn_loader is None:
        # ---------- BASELINE: MNIST ONLY ----------
        for batch_idx, (data, target) in enumerate(mnist_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            digit_log_probs, _ = model(data, lambd_grl=0.0)
            loss = F.nll_loss(digit_log_probs, target)

            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} "
                    f"[{batch_idx * len(data)}/{len(mnist_loader.dataset)} "
                    f"({100. * batch_idx / len(mnist_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}"
                )

            if dry_run:
                break
        return

    # ---------- DOMAIN ADAPTATION: MNIST + SVHN ----------
    svhn_iter = iter(svhn_loader)

    for batch_idx, (x_mnist, y_mnist) in enumerate(mnist_loader):
        # get SVHN batch (loop around if needed)
        try:
            x_svhn, _ = next(svhn_iter)
        except StopIteration:
            svhn_iter = iter(svhn_loader)
            x_svhn, _ = next(svhn_iter)

        x_mnist, y_mnist = x_mnist.to(device), y_mnist.to(device)
        x_svhn = x_svhn.to(device)

        # combine MNIST + SVHN for domain classifier
        x_all = torch.cat([x_mnist, x_svhn], dim=0)

        # domain labels: MNIST = 0, SVHN = 1
        domain_labels = torch.cat([
            torch.zeros(len(x_mnist), dtype=torch.long),
            torch.ones(len(x_svhn), dtype=torch.long)
        ]).to(device)

        optimizer.zero_grad()

        # forward (GRL if lambd_grl > 0)
        digit_log_probs, domain_log_probs = model(x_all, lambd_grl=lambd_grl)

        # digit loss only on MNIST part
        digit_log_probs_mnist = digit_log_probs[:len(x_mnist)]
        loss_digit = F.nll_loss(digit_log_probs_mnist, y_mnist)

        # domain loss on MNIST + SVHN
        loss_domain = F.nll_loss(domain_log_probs, domain_labels)

        # combine losses
        if lambd_grl > 0.0:
            # gradient reversal already flips gradient of domain head
            loss = loss_digit + beta * loss_domain
        else:
            # domain confusion: explicitly negate domain loss
            loss = loss_digit - alpha * loss_domain

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(x_mnist)}/{len(mnist_loader.dataset)} "
                f"({100. * batch_idx / len(mnist_loader):.0f}%)]\t"
                f"Digit Loss: {loss_digit.item():.6f}\t"
                f"Domain Loss: {loss_domain.item():.6f}"
            )

        if dry_run:
            break


# -------------------------
# Evaluation (digit classification only)
# -------------------------

def test(model, device, test_loader, name="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            digit_log_probs, _ = model(data, lambd_grl=0.0)
            test_loss += F.nll_loss(digit_log_probs, target, reduction='sum').item()
            pred = digit_log_probs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f'\n{name} set: Average loss: {test_loss:.4f}, '
        f'Accuracy: {correct}/{len(test_loader.dataset)} '
        f'({100. * correct / len(test_loader.dataset):.0f}%)\n'
    )


# -------------------------
# Stage 1 & Stage 2 helpers
# -------------------------

def train_stage1(model, device, mnist_train_loader, mnist_test_loader,
                 lr, gamma, epochs, log_interval, dry_run):
    print("=== Stage 1: Baseline MNIST training ===")
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, mnist_train_loader, optimizer, epoch,
              log_interval=log_interval,
              use_domain_adaptation=False,
              svhn_loader=None,
              dry_run=dry_run)
        test(model, device, mnist_test_loader, name="MNIST")
        scheduler.step()

    return model


def train_stage2(model, device, mnist_train_loader, svhn_train_loader,
                 mnist_test_loader, svhn_test_loader,
                 lr, gamma, epochs, log_interval, dry_run,
                 use_grl=True, alpha=0.1, lambd_grl=1.0):
    print("=== Stage 2: Domain adaptation MNIST ↔ SVHN ===")
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, mnist_train_loader, optimizer, epoch,
              log_interval=log_interval,
              use_domain_adaptation=True,
              svhn_loader=svhn_train_loader,
              alpha=alpha,
              lambd_grl=lambd_grl if use_grl else 0.0,
              beta=0.1,
              dry_run=dry_run)
        test(model, device, mnist_test_loader, name="MNIST")
        test(model, device, svhn_test_loader, name="SVHN")
        scheduler.step()

    return model


# -------------------------
# Main
# -------------------------

if __name__ == '__main__':
    batch_size = 65
    test_batch_size = 1000
    epochs = 1
    epochs2 = 5
    lr = 1.0
    gamma = 0.7
    log_interval = 10
    device = "cpu"
    dry_run = False
    checkpoint_path = "base_model.ckpt"

    # Transforms: resize to 32x32, 1 channel
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    svhn_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # MNIST
    mnist_train = torchvision.datasets.MNIST(
        root="./data", train=True, download=DOWNLOAD_DATA, transform=mnist_transform
    )
    mnist_train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=2
    )

    mnist_test = torchvision.datasets.MNIST(
        root="./data", train=False, download=DOWNLOAD_DATA, transform=mnist_transform
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=test_batch_size, shuffle=False, num_workers=2
    )

    # SVHN
    svhn_train = torchvision.datasets.SVHN(
        root="./data", split='train', download=DOWNLOAD_DATA, transform=svhn_transform
    )
    svhn_train_loader = torch.utils.data.DataLoader(
        svhn_train, batch_size=batch_size, shuffle=True, num_workers=2
    )

    svhn_test = torchvision.datasets.SVHN(
        root="./data", split='test', download=DOWNLOAD_DATA, transform=svhn_transform
    )
    svhn_test_loader = torch.utils.data.DataLoader(
        svhn_test, batch_size=test_batch_size, shuffle=False, num_workers=2
    )

    # Model
    base_model = Model().to(device)

    if os.path.exists(checkpoint_path):
        print("### Loading saved baseline MNIST model ###")
        base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("### Training baseline MNIST model (Stage 1) ###")
        base_model = train_stage1(
            base_model, device,
            mnist_train_loader, mnist_test_loader,
            lr, gamma, epochs, log_interval, dry_run
        )
        print("### Saving baseline model checkpoint ###")
        torch.save(base_model.state_dict(), checkpoint_path)

    test(base_model, device, mnist_test_loader, name="MNIST")
    test(base_model, device, svhn_test_loader, name="SVHN")

    # Stage 2: domain adaptation (uncomment when ready)
    base_model = train_stage2(
        base_model, device,
        mnist_train_loader, svhn_train_loader,
        mnist_test_loader, svhn_test_loader,
        lr, gamma, epochs2, log_interval, dry_run,
        use_grl=True,   # set False to use confusion loss with -alpha * domain_loss
        alpha=np.log(10) / np.log(2.0),
        lambd_grl=0.1, #
    )
