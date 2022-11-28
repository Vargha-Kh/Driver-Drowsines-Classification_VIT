import torch
from torchvision import transforms, datasets


def get_dataset(directory="./fer2013", batch_size=128, img_size=48):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    transform_train = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(0.3),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    transform_val = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(0.3),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    train_dataset = datasets.ImageFolder(directory + '/train', transform=transform_train)
    val_dataset = datasets.ImageFolder(directory + '/validation', transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    return train_loader, val_loader,
