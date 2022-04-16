import torch
from torch.utils.data import Dataset
import torchaudio
import glob

class DemixingAudioDataset(Dataset):
    """Demixing Audio dataset"""

    def __init__(self, root_dir, transform=None):
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
        self.transform = transform

    def __len__(self):
        size = len(glob.glob("{}/*".format(self.root_dir)))
        return size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        mixture_waveform, _ = torchaudio.load("{}/{}".format(glob.glob("{}/*".format(self.root_dir))[idx],'mixture.wav'))
        bass_waveform, _ = torchaudio.load("{}/{}".format(glob.glob("{}/*".format(self.root_dir))[idx],'bass.wav'))
        drums_waveform, _ = torchaudio.load("{}/{}".format(glob.glob("{}/*".format(self.root_dir))[idx],'drums.wav'))
        others_waveform, _ = torchaudio.load("{}/{}".format(glob.glob("{}/*".format(self.root_dir))[idx],'other.wav'))
        vocals_waveform, _ = torchaudio.load("{}/{}".format(glob.glob("{}/*".format(self.root_dir))[idx],'vocals.wav'))

        sample = (
            mixture_waveform,
            {
                "bass": bass_waveform,
                "drums": drums_waveform,
                "vocals": others_waveform,
                "other": vocals_waveform
            }
        )

        if self.transform:
            sample = self.transform(sample)

        return sample