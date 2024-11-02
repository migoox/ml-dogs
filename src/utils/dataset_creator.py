from enum import Enum
import os
import time
import numpy as np
import torch
from torch import Tensor
from utils.daps_explorer import DapsExplorer
from torchaudio import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


class DatasetType(Enum):
    Train = 1
    Validation = 2
    Test = 3


class DatasetCreator:
    def __init__(
        self, class0: list[DapsExplorer], class1: list[DapsExplorer], dataset_path: str, dataset_type: DatasetType
    ):
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

        image = Image.fromarray((specgram.numpy() * 255).astype(np.uint8), mode="L")
        return ImageOps.flip(image)

    @staticmethod
    def save_image(directory: str, name: str, image: Image.Image):
        image.save(os.path.join(directory, name))

    def export_dataset(self, n_fft=1024, interval_duration: float = 2, multithreading=True):
        print("DatasetCreator: Exporting the dataset with the following parameters:")
        print(f"    n_fft={n_fft}")
        print(f"    interval_duration={interval_duration}s")
        print(f"    multithreading={multithreading}")
        print(f"Class 0 recordings count: {len(self.class0)}")
        print(f"Class 1 recordings count: {len(self.class1)}")

        specgram_transform = transforms.Spectrogram(n_fft=n_fft)
        img_width = DapsExplorer.get_time_bins_len(duration=interval_duration, n_fft=n_fft)
        img_height = DapsExplorer.get_freq_bins_len(n_fft=n_fft)
        dataset_type_name = (
            "train"
            if self.dataset_type == DatasetType.Train
            else ("validation" if self.dataset_type == DatasetType.Validation else "test")
        )
        files_count = len(self.class0) + len(self.class1)

        os.makedirs(os.path.join(self.dataset_path, dataset_type_name, "0"), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, dataset_type_name, "1"), exist_ok=True)

        start_time = time.time()
        def thread_work(specgrams: list[torch.Tensor], file_name: str, file_ind: int):
            print(f"Processing [{file_ind + 1}/{files_count}]...")

            for i, s in enumerate(specgrams):
                class_name = "0" if file_ind < len(self.class0) else "1"

                image = DatasetCreator.get_image(s, img_height, img_width)
                DatasetCreator.save_image(
                    os.path.join(
                        self.dataset_path,
                        dataset_type_name,
                        class_name,
                    ),
                    f"chunk_{i}_{os.path.splitext(file_name)[0]}.png",
                    image,
                )

            print(f"Finished [{file_ind + 1}/{files_count}]")

        with ThreadPoolExecutor() as executor:
            futures = []
            for file_ind, c in enumerate(self.class0 + self.class1):
                file_name = c.get_file_name()
                specgrams = c.load_specgram_splitted_tensors(
                    interval_duration=interval_duration, normalize=True, specgram_transform=specgram_transform
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
