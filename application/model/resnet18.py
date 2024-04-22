import torch
import torch.nn as nn
import torchvision.models as models


def resnet18():
    net = models.resnet18()
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 120),
        )
    return net

def resnet50():
    net = models.resnet50()
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Sequential(
        nn.Linear(2048, 256), nn.ReLU(),
        nn.Linear(256, 120),
        )
    return net
    