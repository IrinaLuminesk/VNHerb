from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class DatasetLoader(Dataset):
    """
    Hàm dùng để load dữ liệu từ Folder có kèm transform
        ...
    
    Args:
        root_dir (str): Path to dataset root directory.
        transform (callable, optional): Transformations to apply to the dataset.
    """
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.data = ImageFolder(root_dir, transform=transform)
    def __len__(self):
        return len(self)
    def __getitem__(self, index):
        return self.data[index]
    
    @property
    def classes(self):
        return self.data.classes
        