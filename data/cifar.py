import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout


class Cifar:
    def __init__(self, batch_size, threads, cifar100=False):
        self.cifar100 = cifar100
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        if cifar100:
            train_set = torchvision.datasets.CIFAR100(root='/data/DataSets/cifar-100', train=True, download=False, transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root='/data/DataSets/cifar-100', train=False, download=False, transform=test_transform)
        else:
            train_set = torchvision.datasets.CIFAR10(root='/data/DataSets/cifar-10', train=True, download=False, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root='/data/DataSets/cifar-10', train=False, download=False, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        if self.cifar100:
            train_set = torchvision.datasets.CIFAR100(root='/data/DataSets/cifar-100', train=True, download=True, transform=transforms.ToTensor())
        else:
            train_set = torchvision.datasets.CIFAR10(root='/data/DataSets/cifar-10', train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


class ImageNet:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.ImageNet(root='/data/DataSets/imagenet', split='train', download=False, transform=train_transform)
        test_set = torchvision.datasets.ImageNet(root='/data/DataSets/imagenet', split='val', download=False, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        train_set = torchvision.datasets.ImageNet(root='/data/DataSets/imagenet', split='train', download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])