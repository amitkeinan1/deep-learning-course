import torch
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
    train_dataset = datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8)
    return train_loader, test_loader
