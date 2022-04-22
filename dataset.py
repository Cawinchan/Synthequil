import torch
from torch.utils.data import Dataset
import torchaudio
import glob
from typing import Callable, Optional, Union
from constants import *

class DemixingAudioDataset(Dataset):
    """Demixing Audio dataset"""

    def __init__(self, root_dir: str, chunk_size: int, transform: Optional[Callable]=None):
        """
        Args:
            root_dir (string): Directory with all the audio.
            transform (callable, optional): Optional transform to be applied on a sample.

        Returns: 
            
            sample (dict): No transform 
                    key (str): 'context',
                    value (torch.Tensor): mixture_waveform
                    key (str): 'target',
                    values (tuple): (bass_waveform, drums_waveform ... )

            sample (dict): With transform
                    key (str): 'context',
                    value (list of torch.Tensor): mixture_stft_batch
                    key (str): 'target',
                    values (tuple of list of torch.Tensor): (bass_stft_batch, drums_stft_batch ... )

            
        """
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        self.transform = transform

    # Get number of complete tracks used for training
    def __len__(self):
        size = len(glob.glob("{}/*".format(self.root_dir)))
        return size

    def __getitem__(self, idx: Union[torch.Tensor, list]):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load mixture waveform
        mixture_waveform, _ = torchaudio.load("{}/{}".format(glob.glob("{}/*".format(self.root_dir))[idx],'mixture.wav'))

        # Find first and last samples with non-zero amplitude, and clip audio to just samples within this range
        nonzero_indices = torch.nonzero(torch.flatten(torch.transpose(mixture_waveform,-1,-2)))
        start_index = nonzero_indices[0].item() // 2
        end_index = nonzero_indices[-1].item() // 2
        mixture_waveform = mixture_waveform[:,start_index:end_index+1]

        # Load other waveforms
        waveforms_dict = dict()
        for i in INSTRUMENTS:
            waveforms_dict[i], _ = torchaudio.load("{}/{}".format(glob.glob("{}/*".format(self.root_dir))[idx],'{}.wav'.format(i)))

        # Perform clipping and chunking
        mixture_waveform = torch.split(mixture_waveform,self.chunk_size,dim=-1)
        for i in INSTRUMENTS:
            waveforms_dict[i] = torch.split(waveforms_dict[i][:,start_index:end_index+1],self.chunk_size,dim=-1)

        sample = (
            mixture_waveform,
            waveforms_dict
        )

        # Perform transform if needed
        if self.transform:
            sample = self.transform(sample)

        return sample