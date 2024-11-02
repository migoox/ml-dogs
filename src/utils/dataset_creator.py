from enum import Enum
import os
import numpy as np
import torch
from torch import Tensor
from utils.daps_explorer import DapsExplorer
from torchaudio import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class DatasetType(Enum):
    Train = 1
    Validation = 2
    Test = 3

class DatasetCreator:
    def __init__(self, class0: list[DapsExplorer], class1: list[DapsExplorer], dataset_path: str, dataset_type: DatasetType):
        self.class0 = class0
        self.class1 = class1
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
    
    @staticmethod
    def get_image(specgram: Tensor, img_height: int, img_width: int) -> Image.Image:
        ch, cw = specgram.shape

        # fit into the image
        if ch > img_height:
            specgram = specgram[:img_height, :]
        if cw > img_width:
            specgram = specgram[:, :img_width]

        # pad if too small
        if cw < img_height or cw < img_width:
            pad_height = max(0, img_height - ch)
            pad_width = max(0, img_width - cw)

            specgram = torch.nn.functional.pad(specgram, (0, pad_width, 0, pad_height), value=0)
        
        image = Image.fromarray((specgram.numpy() * 255).astype(np.uint8), mode='L')
        return ImageOps.flip(image)
    
    @staticmethod
    def save_image(dir: str, name: str, image: Image.Image):
        os.makedirs(dir, exist_ok=True)
        image.save(os.path.join(dir, name))

    def export_dataset(self, interval_duration: float = 2):
        n_fft = 512
        specgram_transform = transforms.Spectrogram(n_fft=n_fft)
        img_width = DapsExplorer.get_time_bins_len(duration=interval_duration, n_fft=n_fft)
        img_height = DapsExplorer.get_freq_bins_len(n_fft=n_fft)
        dataset_type_name = "train" if self.dataset_type == DatasetType.Train else (
            "validation" if self.dataset_type == DatasetType.Validation else "test"
        )

        def thread_work(s, directory, name):
            image = DatasetCreator.get_image(s, img_height, img_width)
            DatasetCreator.save_image(directory, name, image)

        for file_ind, c in enumerate(self.class0 + self.class1):
            file_name = c.get_file_name()
            specgrams =  c.load_specgram_splitted_tensors(interval_duration=interval_duration, normalize=True, specgram_transform=specgram_transform)
            for i, s in enumerate(specgrams):
                class_name = "0" if file_ind < len(self.class0) else "1"
                directory = os.path.join(self.dataset_path, dataset_type_name, class_name)
                name = f"chunk_{i}_{os.path.splitext(file_name)[0]}.png"
                thread_work(s, directory, name)
