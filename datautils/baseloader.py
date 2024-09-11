import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

#########################
# Set up logging
#########################
import logging
from logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class Dataset_base(Dataset):
    def __init__(self, **kwargs):
        """
        Args:
            datadir (string): Path to the data directory, which contains wav files and scp folder.
        """
        self.args = getattr(self, 'args', None)
        self.datadir = getattr(self, 'datadir', 'DATA/')
        self.labels = getattr(self, 'labels', None)
        self.base_dir = getattr(self, 'base_dir', './')
        self.list_IDs = []
        self.bonafide_dir = os.path.join(self.base_dir, 'bonafide')
        self.vocoded_dir = os.path.join(self.base_dir, 'vocoded')
        self.algo = getattr(self, 'algo', 5)
        self.vocoders = getattr(self, 'vocoders', [])
        print("vocoders:", self.vocoders)
        
        self.augmentation_methods = getattr(self, 'augmentation_methods', ["RawBoost12"])
        self.eval_augment = getattr(self, 'eval_augment', None)
        self.num_additional_real = getattr(self, 'num_additional_real', 2)
        self.num_additional_spoof = getattr(self, 'num_additional_spoof', 2)
        self.trim_length = getattr(self, 'trim_length', 64000)
        self.sample_rate = getattr(self, 'wav_samp_rate', 16000)
        
        self.args.noise_path = getattr(self, 'noise_path', './NOISES/')
        self.args.rir_path = getattr(self, 'rir_path', './RIRS/')
        self.args.aug_dir = getattr(self, 'aug_dir', './AUGMENTED/')
        self.args.online_aug = getattr(self, 'online_aug', False)
        self.repeat_pad = getattr(self, 'repeat_pad', True)
        self.is_train = getattr(self, 'is_train', True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # to be implemented in child classes
        pass
        
