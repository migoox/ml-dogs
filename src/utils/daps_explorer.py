from collections import defaultdict
import copy
from enum import Enum
import math
import sys
from typing import Optional
from torch import Tensor
import torch
import os
import torchaudio
from torchaudio import transforms
import random
import itertools
from typing import Optional

def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """
    Convert a power spectrogram (amplitude squared) to decibel (dB) units.

    Parameters:
        S (torch.Tensor): Input power spectrogram tensor.
        ref (float or callable): Reference value for scaling. If callable, computes ref(S).
        amin (float): Minimum threshold for S to avoid log of zero.
        top_db (float): Threshold in dB. Values lower than (max_dB - top_db) are set to (max_dB - top_db).

    Returns:
        torch.Tensor: The dB-scaled spectrogram tensor.
    """
    S = torch.maximum(S, torch.tensor(amin, dtype=S.dtype))

    if callable(ref):
        ref_value = ref(S)
    else:
        ref_value = ref

    S_db = 10.0 * torch.log10(S / ref_value)

    if top_db is not None:
        max_db = torch.max(S_db)
        S_db = torch.maximum(S_db, max_db - top_db)

    return S_db

class DapsRouteNodeType(Enum):
    Device = 1
    Surrounding = 2
    RecordingType = 3
    Speaker = 4
    Script = 5

class DataSetType(Enum):
    Training = 1
    Validation = 2
    Test = 3

class DapsExplorer:
    

    @staticmethod
    def get_recording_types() -> list[str]:
        """
        The DAPS dataset contains 4 types of recordings:
            - cleanraw: 20 participants were asked to read 5 scripts in the professional studio,
            - clean: based on cleanraw, but the breaths, lip smacks, etc. were removed by sound engineer,
            - produced: based on clean, but effects were applied by the sound engineer,
            - device: the clean recordings where played through a loudspeaker and recorded by devices, e.g. ipad in several surroundings e.g. on balcony.

        Returns:
            List of keys representing the recording types.
        """

        return [
            "cleanraw",
            "clean",
            "produced",
            "device",
        ]

    @staticmethod
    def get_devices() -> list[str]:
        """
        Returns:
            List of devices keys that were used to record the clean recordings played through a loudspeaker.
        """

        return [
            "ipad",
            "ipadflat",
            "iphone",
        ]

    @staticmethod
    def get_speakers(gender: Optional[bool] = None, sp_class: Optional[bool] = None) -> list[str]:
        """
        Retrieves a list of speaker identifiers based on specified filters for gender and speaker class.

        Parameters:
            gender (bool, optional): Filters speakers by gender.
                                    - True: returns only female speakers.
                                    - False: returns only male speakers.
                                    - None: includes both male and female speakers (no gender filter).

            sp_class (bool, optional): Filters speakers by class.
                                    - True: returns only Class 1 speakers.
                                    - False: returns only Class 0 speakers.
                                    - None: includes both Class 0 and Class 1 speakers (no class filter).

        Returns:
            list[str]: A list of speaker IDs (strings) for speakers who participated in the experiment,
                    filtered by the specified gender and class.
        """

        result = list[str]
        if gender is not None:
            if gender:
                result = [f"f{i}" for i in range(1, 11)]
            else:
                result = [f"m{i}" for i in range(1, 11)]
        else:
            result = [f"f{i}" for i in range(1, 11)] + [f"m{i}" for i in range(1, 11)]

        if sp_class is not None:
            class1_speakers = ["f1", "f7", "f8", "m3", "m6", "m8"]
            if sp_class is True:
                result = list(set(result).intersection(class1_speakers))
            else:
                result = list(set(result).difference(class1_speakers))

        return result

    @staticmethod
    def get_surroundings() -> list[str]:
        """
        Returns:
            List of surroundings keys in which the device recordings where taken.
        """

        return [
            "balcony1",
            "bedroom1",
            "confroom1",
            "confroom2",
            "livingroom1",
            "office1",
            "office2",
            "confroom1",
        ]

    @staticmethod
    def get_scripts() -> list[str]:
        """
        Returns:
            List of 5 scripts keys which were read by the participants.
        """
        return [
            "script1",
            "script2",
            "script3",
            "script4",
            "script5",
        ]

    @staticmethod
    def get_scripts_by_data_set_type(type: DataSetType) -> list[str]:
        """
        Retrieves script keys used by the chosen data set type.

        Parameters:
            type (DataSetType): identifies portion of the data set: training, validation or test. 

        Returns:
            List of scripts keys making up the chosen set.
        """
        scripts = {
            DataSetType.Training: [
                'script1',
                'script3',
                'script5',
            ],
            DataSetType.Validation: [
                'script2',
            ],
            DataSetType.Test: [
                'script4',
            ],
        }

        return scripts[type]

    @staticmethod
    def get_surroundings_by_device(device: str) -> list[str]:
        """
        Retrieves surroundings available for the chosen device.

        Parameters:
            device (str): identifies the device between 'ipad', 'ipadflat' and 'iphone'. 

        Returns:
            List of surroundings avaialable for the chosen device.
        """
        if device not in DapsExplorer.get_devices():
            raise KeyError(f"Provided key '{device}' is invalid in the DAPS dataset.")

        device_to_surrounding = {
            'ipad': [
                'balcony1',
                'bedroom1',
                'confroom1',
                'confroom2',
                'livingroom1',
                'office1',
                'office2',
            ],
            'ipadflat': [
                'confroom1',
                'office1',
            ],
            'iphone': [
                'balcony1',
                'bedroom1',
                'livingroom1',
            ]
        }
        return device_to_surrounding[device]
        
    @staticmethod
    def get_data_set(type: DataSetType, sp_class: Optional[bool] = None) -> list['DapsExplorer']:
        """
        Retrieves either the training, validation or test data set based on the provided parameter.

        Parameters:
            type (DataSetType): identifies portion of the data set: training, validation or test. 

            sp_class (bool, optional): Filters speakers by class.
                                    - True: returns only Class 1 speakers.
                                    - False: returns only Class 0 speakers.
                                    - None: includes both Class 0 and Class 1 speakers (no class filter).

        Returns:
            Data set represented as a list of DapsExplorers.
        """
        result = []

        dir_path = os.path.dirname(os.path.realpath(__file__))
        root = DapsExplorer(os.path.join(dir_path, "..", "..", "data", "daps"))

        random.seed(45)
        
        script_speaker_rectype_lists = [
            DapsExplorer.get_scripts_by_data_set_type(type),
            DapsExplorer.get_speakers(sp_class=sp_class),
            DapsExplorer.get_recording_types(),
        ]

        for element in itertools.product(*script_speaker_rectype_lists):
            script = element[0]
            speaker = element[1]
            recording_type = element[2]

            if recording_type == 'device':
                for device in DapsExplorer.get_devices():
                    for surrounding in DapsExplorer.get_surroundings_by_device(device):
                        file = root[recording_type][device][surrounding][script][speaker]
                        result.append(file)
            else:
                file = root[recording_type][script][speaker]
                result.append(file)
                    
        return result

    def __init__(self, daps_folder_path: str, route: list[str] = []):
        self.folder_path = daps_folder_path
        self.route = route

    def __getitem__(self, key: str) -> "DapsExplorer":
        new_route = copy.deepcopy(self.route)
        new_route.append(key)
        return DapsExplorer(self.folder_path, new_route)

    def load_wav(self) -> Tensor:
        """
        Loads the .wav file and returns a tuple containing wave torch tensor and sample rate.

        Note: The sample_rate may be accessed with `get_samplerate()`.
        """
        waveform, _ = torchaudio.load(self.get_file_path(), normalize=True)
        return waveform

    def load_wav_splitted(self, interval_duration: float) -> list[Tensor]:
        """
        Loads the .wav file and returns audio chunks of the given interval represented as torch tensors.
        If the recording cannot be splitted to the intervals that are exactly interval_duration long, the
        last interval will be omitted.

        Note: The sample_rate may be accessed with `get_samplerate()`.
        """
        waveform = self.load_wav()
        samples_per_interval = int(DapsExplorer.get_samplerate() * interval_duration)
        chunks = torch.split(waveform, samples_per_interval, dim=1)

        # remove chunks that are not exactly interval_duration long
        result: list[Tensor] = [chunk for chunk in chunks if chunk.size(1) == samples_per_interval]

        return result

    def load_specgram_tensor(
        self,
        specgram_transform: transforms.MelSpectrogram = transforms.MelSpectrogram(n_fft=1024, n_mels=86),
    ) -> Tensor:
        """
        Returns a spectrogram represented as a torch tensor. Amplitudes are expressed in dB.
        """
        waveform = self.load_wav()
        amp_transform = transforms.AmplitudeToDB(stype='power', top_db=80.0)
        s = amp_transform(specgram_transform(waveform)[0])

        return s

    @staticmethod
    def save_wav_from_specgram(file_path: str, specgram_db: Tensor, n_fft: int = 1024, n_mels: int = 86) -> list[str]:
        specgram_amp = torchaudio.functional.DB_to_amplitude(specgram_db, ref=1.0, power=0.5)

        inv_mel_transform = torchaudio.transforms.InverseMelScale(n_mels=n_mels, n_stft=int(n_fft // 2) + 1)
        grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft)

        waveform = grifflim_transform(inv_mel_transform(specgram_amp))

        # Ensure waveform is 2D (channels, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add a channel dimension for mono audio
        elif waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if stereo

        assert waveform.dim() == 2, f"Waveform must be 2D, but got shape {waveform.shape}"

        torchaudio.save(
            file_path,
            waveform,
            DapsExplorer.get_samplerate(),
            encoding="PCM_S",
            bits_per_sample=16
        )

    def load_specgram_split_tensors(
        self,
        interval_duration: float,
        min_db: float,
        max_db: float,
        specgram_transform: transforms.MelSpectrogram = transforms.MelSpectrogram(n_fft=1024),
    ) -> list[Tensor]:
        """
        Returns a spectrograms of provided interval represented as a torch tensor. Amplitudes are expressed in dB.
        """
        s = (self.load_specgram_tensor(specgram_transform = specgram_transform) - min_db) / (max_db - min_db)
        interval_length = DapsExplorer.get_time_bins_len(duration = interval_duration, n_fft=specgram_transform.n_fft)

        chunks: list[Tensor] = list()
        rlen = int(s.shape[1] / interval_length) * interval_length 
        for ind in range(0, rlen, interval_length):
            chunks.append(s[:, ind:ind + interval_length])

        return chunks

    def get_file_name(self) -> str:
        """
        Returns the constructed DAPS file name, if the construction is not valid an error is thrown
        """
        DapsExplorer.validate_route(self.route)

        nodes = dict()
        for node in self.route:
            nodes[DapsExplorer.get_route_node_type(node)] = node

        if "device" in self.route:
            return f"{nodes[DapsRouteNodeType.Speaker]}_{nodes[DapsRouteNodeType.Script]}_{nodes[DapsRouteNodeType.Device]}_{nodes[DapsRouteNodeType.Surrounding]}.wav"
        else:
            return f"{nodes[DapsRouteNodeType.Speaker]}_{nodes[DapsRouteNodeType.Script]}_{nodes[DapsRouteNodeType.RecordingType]}.wav"

    def get_file_path(self):
        """
        Returns the constructed DAPS file path, if the construction is not valid an error is thrown
        """
        file = self.get_file_name()
        for root, _, files in os.walk(self.folder_path):
            if file in files:
                return os.path.join(root, file)

        return None

    @staticmethod
    def get_route_node_type(key: str) -> "DapsRouteNodeType":

        if key in DapsExplorer.get_recording_types():
            return DapsRouteNodeType.RecordingType
        elif key in DapsExplorer.get_devices():
            return DapsRouteNodeType.Device
        elif key in DapsExplorer.get_speakers():
            return DapsRouteNodeType.Speaker
        elif key in DapsExplorer.get_surroundings():
            return DapsRouteNodeType.Surrounding
        elif key in DapsExplorer.get_scripts():
            return DapsRouteNodeType.Script
        else:
            raise KeyError(f"Provided key '{key}' is invalid in the DAPS dataset.")

    @staticmethod
    def validate_route(route: list[str]):
        if route is []:
            return

        node_types: list[DapsRouteNodeType] = list(map(DapsExplorer.get_route_node_type, route))

        node_types_count = defaultdict(int)
        for node_type in node_types:
            node_types_count[node_type] += 1

        for key, value in node_types_count.items():
            if value > 1:
                raise KeyError(f"Invalid route, key of type '{key}' appeared more than once")

        if (
            "device" in route
            and DapsRouteNodeType.Device not in node_types
            and DapsRouteNodeType.Surrounding not in node_types
        ):
            raise KeyError(f"Expected device and surrounding because 'device' was found in the route")

    @staticmethod
    def get_samplerate() -> int:
        """
        All DAPS recordings have the same sample rate equal to 44,1kHz.

        Returns:
            Sample rate of all DAPS recordings in Hz.
        """

        return 44100
    
    @staticmethod
    def get_max_freq() -> int:
        """
        All DAPS recordings have the same sample rate equal to 44.1kHz. The sample rate of 44.1 kHz technically allows 
        for audio at frequencies up to 22.05 kHz to be recorded.

        Returns:
            Max audio frequency in Hz.
        """

        return 22050
    
    @staticmethod
    def get_freq_bins_len(n_fft: int) -> int:
        return int(DapsExplorer.get_max_freq() * n_fft / DapsExplorer.get_samplerate()) + 1

    @staticmethod
    def get_time_bins_len(duration: float, n_fft: int) -> int:
        # from https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html
        return int(duration * DapsExplorer.get_samplerate() / n_fft) + 1

    @staticmethod
    def get_freq_and_time_bins(specgram: Tensor) -> tuple[range, range]:
        return range(specgram.shape[1]), range(specgram.shape[0])

