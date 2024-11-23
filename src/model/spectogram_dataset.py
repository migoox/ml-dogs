import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load all images and labels
        for label in ['0', '1']:
            folder_path = os.path.join(data_dir, label)
            for file in os.listdir(folder_path):
                if file.endswith('.png'):
                    self.data.append(os.path.join(folder_path, file))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label