import torch.nn as nn
import torch
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*13*13, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

def get_model(name, in_channels, num_classes):
    if name == 'simplecnn':
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    elif name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {name}")