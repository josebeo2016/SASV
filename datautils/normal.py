import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import librosa
from core_scripts.data_io import wav_augmentation as nii_wav_aug
from core_scripts.data_io import wav_tools as nii_wav_tools
from datautils.RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import random
from datautils.dataio import pad, load_audio

# base loader
from datautils.baseloader import Dataset_base

# augwrapper
from datautils.augwrapper import SUPPORTED_AUGMENTATION

# dynamic import of augmentation methods
for aug in SUPPORTED_AUGMENTATION:
    exec(f"from datautils.augwrapper import {aug}")
    
#########################
# Set up logging
#########################
import logging
from logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
def genList(dir_meta, is_train=False, is_eval=False, is_dev=False):
    # bonafide: 1, spoof: 0
    d_meta = {}
    file_list=[]
    protocol = os.path.join(dir_meta, "protocol.txt")

    if (is_train):
        with open(protocol, 'r') as f:
            l_meta = f.readlines()
        for line in l_meta:
            utt, subset, label = line.strip().split()
            if subset == 'train':
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0

        return d_meta, file_list
    if (is_dev):
        with open(protocol, 'r') as f:
            l_meta = f.readlines()
        for line in l_meta:
            utt, subset, label = line.strip().split()
            if subset == 'dev':
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    
    if (is_eval):
        # no eval protocol yet
        with open(protocol, 'r') as f:
            l_meta = f.readlines()
        for line in l_meta:
            utt, subset, label = line.strip().split()
            if subset == 'dev':
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0
        # return d_meta, file_list
        return d_meta, file_list
class Dataset_for(Dataset_base):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[],
                    augmentation_methods=[], eval_augment=None, num_additional_real=2, num_additional_spoof=2,
                    trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None,
                    aug_dir=None, online_aug=False, repeat_pad=True, is_train=True):
        super(Dataset_for, self).__init__(args, list_IDs, labels, base_dir, algo, vocoders, 
                 augmentation_methods, eval_augment, num_additional_real, num_additional_spoof, 
                 trim_length, wav_samp_rate, noise_path, rir_path, 
                 aug_dir, online_aug, repeat_pad, is_train)
    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        filepath = os.path.join(self.base_dir, utt_id)
        X = load_audio(filepath, self.sample_rate)
        
        # apply augmentation
        # randomly choose an augmentation method
        if self.is_train:
            augmethod_index = random.choice(range(len(self.augmentation_methods)))
            X = globals()[self.augmentation_methods[augmethod_index]](X, self.args, self.sample_rate, 
                                                                        audio_path = filepath)

        X_pad= pad(X,padding_type="repeat" if self.repeat_pad else "zero", max_len=self.trim_length, random_start=True)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        return idx, x_inp, target
class Dataset_for_dev(Dataset_base):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[],
                    augmentation_methods=[], eval_augment=None, num_additional_real=2, num_additional_spoof=2,
                    trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None,
                    aug_dir=None, online_aug=False, repeat_pad=True, is_train=True):
        super(Dataset_for_dev, self).__init__(args, list_IDs, labels, base_dir, algo, vocoders, 
                 augmentation_methods, eval_augment, num_additional_real, num_additional_spoof, 
                 trim_length, wav_samp_rate, noise_path, rir_path, 
                 aug_dir, online_aug, repeat_pad, is_train)
        if repeat_pad:
            self.padding_type = "repeat"
        else:
            self.padding_type = "zero"
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        X_pad = pad(X,self.padding_type,self.trim_length, random_start=True)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return index, x_inp, target

class Dataset_for_eval(Dataset_base):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[],
                    augmentation_methods=[], eval_augment=None, num_additional_real=2, num_additional_spoof=2,
                    trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None,
                    aug_dir=None, online_aug=False, repeat_pad=True, is_train=True, enable_chunking=False
                    ):
        super(Dataset_for_eval, self).__init__(args, list_IDs, labels, base_dir, algo, vocoders, 
                 augmentation_methods, eval_augment, num_additional_real, num_additional_spoof, 
                 trim_length, wav_samp_rate, noise_path, rir_path, 
                 aug_dir, online_aug, repeat_pad, is_train)
        self.enable_chunking = enable_chunking
        if repeat_pad:
            self.padding_type = "repeat"
        else:
            self.padding_type = "zero"
    
    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        filepath = os.path.join(self.base_dir, utt_id)
        X, _ = librosa.load(filepath, sr=16000)
        # apply augmentation at inference time
        if self.eval_augment is not None:
            # print("eval_augment:", self.eval_augment)
            X = globals()[self.eval_augment](X, self.args, self.sample_rate, audio_path = filepath)
        if not self.enable_chunking:
            X= pad(X,padding_type=self.padding_type,max_len=self.trim_length, random_start=True)
        x_inp = Tensor(X)
        return x_inp, utt_id