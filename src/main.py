import os
import matplotlib.pyplot as plt
from utils.daps_explorer import DapsExplorer

dir_path = os.path.dirname(os.path.realpath(__file__))

def daps_expl_use_case_example_1():
    root = DapsExplorer(f"{dir_path}/../data/daps/") # DAPS dataset path relative to the current working directory

    dev_sc1 = root['device']['ipad']['balcony1']['script1'] # order doesn't matter

    wavefrom, sample_rate = dev_sc1['f1'].load_wav()
    print(sample_rate)

    time_bins, freq_bins, specgram = dev_sc1['f1'].load_specgram(1024)

    plt.figure(figsize=(30, 4))
    plt.pcolormesh(time_bins, freq_bins, specgram)
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram f1')
    plt.show()


def daps_expl_use_case_example_2():
    root = DapsExplorer(f"{dir_path}/../data/daps/")

    speakers = DapsExplorer.get_speakers(gender=True)

    cl_sc1 = root['clean']['script1']

    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(30, rows * 5))
    axes = axes.flatten()

    for i, speaker in enumerate(speakers):
        time_bins, freq_bins, specgram = cl_sc1[speaker].load_specgram()
        
        ax = axes[i]
        pcm = ax.pcolormesh(time_bins, freq_bins, specgram, shading='auto')
        fig.colorbar(pcm, ax=ax, label='Amplitude')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Spectrogram for Speaker {speaker}')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def daps_expl_use_case_example_3():
    root = DapsExplorer(f"{dir_path}/../data/daps/")

    n_ffts = [512, 1024, 2048, 4096]

    cl_sc1 = root['clean']['script1']

    cols = 2
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()

    for i, n_fft in enumerate(n_ffts):
        time_bins, freq_bins, specgram = cl_sc1['f1'].load_specgram(n_fft)
        
        ax = axes[i]
        pcm = ax.pcolormesh(time_bins, freq_bins, specgram, shading='auto')
        fig.colorbar(pcm, ax=ax, label='Amplitude')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Spectrogram for Speaker f1 with n_fft={n_fft}')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    daps_expl_use_case_example_3() 
