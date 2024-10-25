from collections import defaultdict
import copy
from enum import Enum
import numpy
from torch import Tensor
import torch
import os
import torchaudio
import torchaudio.transforms

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

class DapsExplorer:
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
            'cleanraw',
            'clean',
            'produced',
            'device',
        ]

    def get_devices() -> list[str]:
        """
        Returns:
            List of devices keys that were used to record the clean recordings played through a loudspeaker.
        """

        return [
            'ipad',
            'ipadflat',
            'iphone',
        ]

    def get_speakers(gender: bool = None, sp_class: bool = None) -> list[str]:
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

    def get_surroundings() -> list[str]:
        """
        Returns:
            List of surroundings keys in which the device recordings where taken.
        """

        return [
            'balcony1',
            'bedroom1',
            'confroom1',
            'confroom2',
            'livingroom1',
            'office1',
            'office2',
            'confroom1',
        ]

    def get_scripts() -> list[str]:
        """
        Returns:
            List of 5 scripts keys which were read by the participants.
        """
        return [
            'script1',
            'script2',
            'script3',
            'script4',
            'script5',
        ]

    def __init__(self, daps_folder_path: str, route: list[str] = []):
        self.folder_path = daps_folder_path
        self.route = route
    
    def __getitem__(self, key: str) -> 'DapsExplorer':
        new_route = copy.deepcopy(self.route)
        new_route.append(key)
        return DapsExplorer(self.folder_path, new_route)
    
    def load_wav(self) -> tuple[Tensor, int]:
        """
        Loads the .wav file and returns a tuple containing wave torch tensor and sample rate
        """
        return torchaudio.load(self.get_file_path())

    def load_specgram_tensor(self, n_fft: int = 1024) -> Tensor:
        """
        Returns a spectrogram represented as a torch tensor. Amplitudes are expressed in dB.
        """
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft)
        waveform, _ = self.load_wav()

        return power_to_db(spectrogram(waveform)[0])

    def load_specgram(self, n_fft: int = 1024) -> tuple[int, int, numpy.ndarray]:
        """
        Returns a spectrogram represented as a numpy ndarray. Each element of the ndarray represents
        a row of the spectrogram, i.e. it contains amplitudes for each time_bin. 
        Amplitudes are expressed in dB.

        The result of the following function is a tuple (time_bins, freq_bins, spectrogram).
        """
        specgram = self.load_specgram_tensor(n_fft)
        return range(specgram.shape[1]), range(specgram.shape[0]), specgram

    def get_file_name(self) -> str:
        """
        Returns the constructed DAPS file name, if the construction is not valid an error is thrown  
        """
        DapsExplorer.validate_route(self.route)

        nodes = dict()
        for node in self.route:
            nodes[DapsExplorer.get_route_node_type(node)] = node

        if 'device' in self.route:
            return f"{nodes[DapsRouteNodeType.Speaker]}_{nodes[DapsRouteNodeType.Script]}_{nodes[DapsRouteNodeType.Device]}_{nodes[DapsRouteNodeType.Surrounding]}.wav"
        else:
            return f"{nodes[DapsRouteNodeType.Speaker]}_{nodes[DapsRouteNodeType.Script]}_{nodes[DapsRouteNodeType.RecordingType]}.wav"
    
    def get_file_path(self):
        """
        Returns the constructed DAPS file path, if the construction is not valid an error is thrown
        """
        file = self.get_file_name()
        for root, dirs, files in os.walk(self.folder_path):
            if file in files:
                return os.path.join(root, file)

        return None
    
    def get_route_node_type(key: str) -> 'DapsRouteNodeType':
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
        
        if 'device' in route and DapsRouteNodeType.Device not in node_types and DapsRouteNodeType.Surrounding not in node_types:
            raise KeyError(f"Expected device and surrounding because 'device' was found in the route")
