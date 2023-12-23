from torch.utils.data import Dataset
from enum import Enum
from PIL import Image
import os
# Define the Dataset class
class PACSDataset(Dataset):

    class DatasetType(Enum):
        PAINTING = 1
        CARTOON = 2
        PHOTO = 3
        SKETCH = 4

    def __init__(self,root : str = "PACS",split : str = "train",dataset : DatasetType = DatasetType.PAINTING, transform=None):
        self.root = root
        self.split = split
        self.dataset = dataset
        self.transform = transform
        self.data = []
        self.label = []
        # Load the dataset
        match self.dataset:
            case self.DatasetType.PAINTING:
                label_file = os.path.join(self.root,"art_painting.txt")
            case self.DatasetType.CARTOON:
                label_file = os.path.join(self.root,"cartoon.txt")
            case self.DatasetType.PHOTO:
                label_file = os.path.join(self.root,"photo.txt")
            case self.DatasetType.SKETCH:
                label_file = os.path.join(self.root,"sketch.txt")
            case _:
                raise ValueError("Invalid dataset type")
        with open(label_file) as f:
            self.data = [line.split(" ") for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data , label = self.data[index]
        data = os.path.join(self.root,data)
        label = int(label)
        data = Image.open(data).convert("RGB")
        if self.transform:
            data = self.transform(data)
        return data, label