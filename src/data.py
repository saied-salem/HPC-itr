import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_flag: str, batch_size: int, download: bool = True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = DataClass(split='train', transform=transform, download=download)
    test_dataset = DataClass(split='test', transform=transform, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return info, train_loader, test_loader