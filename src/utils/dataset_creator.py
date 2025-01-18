from enum import Enum
import os
import time
import numpy as np
import torch
import random
from torch import Tensor
from torchaudio import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import shutil
from concurrent.futures import ThreadPoolExecutor

from utils.daps_explorer import DapsExplorer, DataSetType

class SpecgramsSilentFilter:
    def __init__(self, k: float = 0.75):
        self.k = k
        
    def filter(self, specgrams: list[Tensor]) -> list[Tensor]:
        # Compute average energy using vectorized operations
        total_energy = torch.tensor([specgram.sum() for specgram in specgrams]).sum()
        avg_energy = total_energy / len(specgrams)

        # Filter spectrograms based on the average energy
        specgrams_new = [specgram for specgram in specgrams if specgram.sum() >= self.k*avg_energy]

        return specgrams_new

class SpecgramsRandomFilter:
    def filter(self, specgrams: list[Tensor]) -> list[Tensor]:
        return [specgrams[random.randint(0, len(specgrams) - 1)]]

class DatasetCreator:
    def __init__(
        self, class0: list[DapsExplorer], class1: list[DapsExplorer], parent_path: str, dataset_type: DataSetType, 
        specgram_filters: list
    ):
        self.class0 = class0
        self.class1 = class1
        self.parent_path = parent_path
        self.dataset_type = dataset_type
        self.specgram_filters = specgram_filters

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

        image = Image.fromarray((specgram.numpy() * 255).astype(np.uint8), mode="L")
        return ImageOps.flip(image)

    @staticmethod
    def save_image(directory: str, name: str, image: Image.Image):
        image.save(os.path.join(directory, name))
    
    def export_dataset(self, folder_name: str = "dataset", n_fft=1024, interval_duration: float = 2, n_mels=86, multithreading=True) -> tuple[int, int]:
        """Returns image width and height."""

        print("DatasetCreator: Exporting the dataset with the following parameters:")
        print(f"    n_fft={n_fft}")
        print(f"    n_mels={n_mels}")
        print(f"    interval_duration={interval_duration}s")
        print(f"    multithreading={multithreading}")
        print(f"Class 0 recordings count: {len(self.class0)}")
        print(f"Class 1 recordings count: {len(self.class1)}")

        specgram_transform = transforms.MelSpectrogram(n_fft=n_fft, n_mels=n_mels, sample_rate=DapsExplorer.get_samplerate())
        img_width = DapsExplorer.get_time_bins_len(duration=interval_duration, n_fft=n_fft)
        img_height = n_mels

        dataset_type_name = (
            "train"
            if self.dataset_type == DataSetType.Training
            else ("validation" if self.dataset_type == DataSetType.Validation else "test")
        )
        files_count = len(self.class0) + len(self.class1)
        print("Image properties:")
        print(f"    width={img_width}px")
        print(f"    height={img_height}px")

        class0_path = os.path.join(self.parent_path, folder_name, dataset_type_name, "0")
        class1_path = os.path.join(self.parent_path, folder_name, dataset_type_name, "1")

        if os.path.exists(class0_path):
            print(f"Removing directory {class0_path}")
            shutil.rmtree(class0_path)

        if os.path.exists(class1_path):
            print(f"Removing directory {class1_path}")
            shutil.rmtree(class1_path)

        os.makedirs(class0_path)
        os.makedirs(class1_path)

        start_time = time.time()
        def thread_work(specgrams: list[torch.Tensor], file_name: str, file_ind: int):
            for filter in self.specgram_filters:
                specgrams = filter.filter(specgrams)

            for i, s in enumerate(specgrams):
                class_name = "0" if file_ind < len(self.class0) else "1"

                image = DatasetCreator.get_image(s, img_height, img_width)
                DatasetCreator.save_image(
                    os.path.join(
                        self.parent_path,
                        folder_name,
                        dataset_type_name,
                        class_name,
                    ),
                    f"chunk_{i}_{os.path.splitext(file_name)[0]}.png",
                    image,
                )

            print("\r", end="")
            print(f"Finished [{file_ind + 1}/{files_count}]", end="")

        min_db, max_db = (
            min(specgram.min().item() for c in self.class0 + self.class1 for specgram in [c.load_specgram_tensor()]),
            max(specgram.max().item() for c in self.class0 + self.class1 for specgram in [c.load_specgram_tensor()])
        )

        with ThreadPoolExecutor() as executor:
            futures = []
            for file_ind, c in enumerate(self.class0 + self.class1):
                file_name = c.get_file_name()
                specgrams = c.load_specgram_split_tensors(
                    interval_duration=interval_duration, 
                    min_db=min_db,
                    max_db=max_db,
                    specgram_transform=specgram_transform
                )

                if multithreading:
                    futures.append(executor.submit(thread_work, specgrams, file_name, file_ind))
                else:
                    thread_work(specgrams, file_name, file_ind)

            if multithreading:
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"An error occurred: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Dataset has been exported. Elapsed time: {elapsed_time}s.")

        return img_height, img_width

