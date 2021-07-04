from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
 
def get_loaders(data_dir, transform, batch_size, val_ratio, num_workers):
    ds = ImageFolder(data_dir, transform=transform)
    train_indices, val_indices = train_test_split(list(range(len(ds.targets))), test_size=val_ratio, stratify=ds.targets)
    train_dataset = Subset(ds, train_indices)
    val_dataset = Subset(ds, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


