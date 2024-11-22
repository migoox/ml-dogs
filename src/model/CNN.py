import os

from PIL import Image
from sklearn.metrics import f1_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

dir_path = os.path.dirname(os.path.realpath(__file__))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # output size (6, 82, 509)
        self.pool = nn.MaxPool2d(2, 2)  # output size (6, 41, 254)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # output size (16, 37, 250)

        # After pooling layer, we get (16, 18, 125)
        self.fc1 = nn.Linear(16 * 18 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


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

def test():
    transform = transforms.Compose([
        transforms.Resize((513, 86)),  # Resize to the target spectrogram size (H, W)
        transforms.ToTensor()
    ])

    # Hyperparameters
    batch_size = 32

    test_dataset = SpectrogramDataset(data_dir=os.path.join(dir_path, "..", "dataset", "test"),
                                      transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = torch.load("simple_cnn_full_model.pth", weights_only=False)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on test images: {100 * correct // total} %')


