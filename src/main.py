import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from utils.daps_explorer import DapsExplorer, DataSetType
from utils.dataset_creator import DatasetCreator, SpecgramsSilentFilter, SpecgramsRandomFilter
from prepare_datasets import create_train_data, create_test_data
from PIL import Image
from torchaudio import transforms
import torchvision.transforms as transforms2

dir_path = os.path.dirname(os.path.realpath(__file__))

def daps_expl_use_case_example_1():
    root = DapsExplorer(os.path.join(dir_path, "..", "data", "daps"))

    dev_sc1 = root["device"]["ipad"]["balcony1"]["script1"]  # order doesn't matter

    specgram = dev_sc1["f1"].load_specgram_tensor()
    time_bins, freq_bins  = DapsExplorer.get_freq_and_time_bins(specgram)

    plt.figure(figsize=(30, 4))
    plt.pcolormesh(time_bins, freq_bins, specgram)
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Spectrogram f1")
    plt.show()


def daps_expl_use_case_example_2():
    root = DapsExplorer(os.path.join(dir_path, "..", "data", "daps"))

    speakers = DapsExplorer.get_speakers(gender=True)

    cl_sc1 = root["clean"]["script1"]

    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(30, rows * 5))
    axes = axes.flatten()

    for i, speaker in enumerate(speakers):
        specgram = cl_sc1[speaker].load_specgram_tensor()
        time_bins, freq_bins  = DapsExplorer.get_freq_and_time_bins(specgram)

        ax = axes[i]
        pcm = ax.pcolormesh(time_bins, freq_bins, specgram, shading="auto")
        fig.colorbar(pcm, ax=ax, label="Amplitude")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Spectrogram for Speaker {speaker}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def daps_expl_use_case_example_3():
    root = DapsExplorer(os.path.join(dir_path, "..", "data", "daps"))

    n_ffts = [512, 1024, 2048, 4096]

    cl_sc1 = root["clean"]["script1"]

    cols = 2
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()

    for i, n_fft in enumerate(n_ffts):
        specgram = cl_sc1["f1"].load_specgram_tensor(n_fft)
        time_bins, freq_bins  = DapsExplorer.get_freq_and_time_bins(specgram)

        ax = axes[i]
        pcm = ax.pcolormesh(time_bins, freq_bins, specgram, shading="auto")
        fig.colorbar(pcm, ax=ax, label="Amplitude")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Spectrogram for Speaker f1 with n_fft={n_fft}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def dataset_creator_usage_example():
    root = DapsExplorer(os.path.join(dir_path, "..", "data", "daps"))

    dc = DatasetCreator(
        class0=[
            root["device"]["ipad"]["office1"]["script1"]["f1"],
            root["device"]["ipad"]["office1"]["script1"]["f2"],
            root["device"]["ipad"]["office1"]["script1"]["f3"],
        ],
        class1=[
            root["device"]["ipad"]["office1"]["script2"]["f1"],
            root["device"]["ipad"]["office1"]["script2"]["f2"],
            root["device"]["ipad"]["office1"]["script2"]["f3"],
        ],
        parent_path=os.path.join(dir_path, "..", "datasets"),
        dataset_type=DataSetType.Training,
        specgram_filters=
        [
           SpecgramsSilentFilter(),
        #    SpecgramsRandomFilter()
        ]
    )
    dc.export_dataset()
    
def daps_expl_use_case_example_4():
    for type in DataSetType:
        set = DapsExplorer.get_data_set(type)
        print(type.name, len(set))

def prepare_datasets():
    create_train_data()
    create_test_data()

def data_loss_example():
    root = DapsExplorer(os.path.join(dir_path, "..", "data", "daps"))
    s = root["cleanraw"]["script1"]["f1"].load_specgram_split_tensors(2)[11]

    # save wav file directly
    DapsExplorer.save_wav_from_specgram("test_no_image.wav", s.squeeze())

    # save to image
    min_db = -40
    max_db = 40
    s = (s - min_db) / (max_db - min_db) # convert to [0,1]
    DatasetCreator.save_image("", "test.png", DatasetCreator.get_image(s, 86, 173))

    # load from image
    image = Image.open("test.png")
    transform = transforms2.ToTensor()

    # save wav indirectly
    s2 = transform(image)
    s2 = s2 * (max_db - min_db) + min_db
    DapsExplorer.save_wav_from_specgram("test_from_image.wav", s2.squeeze())

if __name__ == "__main__":
    data_loss_example()
