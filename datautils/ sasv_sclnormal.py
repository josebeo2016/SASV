import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import librosa
import pickle
from core_scripts.data_io import wav_augmentation as nii_wav_aug
from core_scripts.data_io import wav_tools as nii_wav_tools
from datautils.RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import logging
import random
# base loader
from datautils.baseloader import Dataset_base
from datautils.dataio import pad, load_audio
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
    # target: 0, nontarget: 1
    d_meta = {}
    file_list=[]
    # get dir of metafile only
    dir_meta = os.path.dirname(dir_meta)
    if is_train:
        metafile = os.path.join(dir_meta, 'ASVspoof2019.LA.cm.train.trn.txt')
    elif is_dev:
        metafile = os.path.join(dir_meta, 'ASVspoof2019.LA.asv.dev.gi.trl.txt')
    elif is_eval:
        metafile = os.path.join(dir_meta, 'ASVspoof2019.LA.asv.eval.gi.trl.txt')
        
    with open(metafile, 'r') as f:
        l_meta = f.readlines()
    
    if (is_train):
        for line in l_meta:
            utt = line.strip().split()
            file_list.append(utt[0])
            d_meta[utt[0]] = 1
        return d_meta,file_list
    
    if (is_dev):
        for line in l_meta:
            utt = line.strip().split()
            file_list.append(utt[0])
            d_meta[utt[0]] = 1
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            utt = line.strip().split()
            file_list.append(utt[0])

        return d_meta,file_list

class Dataset_Train(Dataset_base):
    def __init__(self, **kwargs):
        super(Dataset_Train, self).__init__(**kwargs)
        # get the bonafide dictionary
        self.speaker_bona = pickle.load(open(os.path.join(self.datadir, 'speaker_bona_train.pkl'), 'rb'))
        # get the spoof dictionary
        self.speaker_spoof = pickle.load(open(os.path.join(self.datadir, 'speaker_spoof_train.pkl'), 'rb'))
        
        # get all the bonafide files
        self.list_IDs = []
        with open(os.path.join(self.datadir, 'bona_train.txt'), 'r') as f:
            self.list_IDs = f.readlines()
        self.list_IDs = [x.strip() for x in self.list_IDs]
        self.n_class = getattr(self, 'n_class', 2)

        


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # Anchor real audio sample
        spid, utt, _, _, _ = self.list_IDs[idx].strip().split()
        real_audio_file = os.path.join(self.base_dir, self.list_IDs[utt]+'.wav')
        anchor = load_audio(real_audio_file)
        anchor = np.expand_dims(anchor, axis=1)
        if self.n_class == 2:
            # randomly choose the label
            label = random.choice([0, 1]) # randomly choose the label following the Normal distribution
            nontarget_type = random.choice([0, 1, 2])
            speakers = list(self.speaker_bona.keys())
            # nontarget_type: 0: bona different speaker, 1: spoof same speaker, 2: spoof different speaker
            if label == 0: # target
                # choose another bonafide file from the same speaker
                bonafide_files = self.speaker_bona[spid]
                bonafide_files.remove(utt)
                bonafide_file = random.choice(bonafide_files)
                bonafide_file = os.path.join(self.base_dir, bonafide_file+'.wav')
                tst_trl = load_audio(bonafide_file)
            else: # nontarget
                if nontarget_type == 0:
                    # choose a bonafide file from a different speaker
                    speakers.remove(spid)
                    spid = random.choice(speakers)
                    bonafide_files = self.speaker_bona[spid]
                    bonafide_file = random.choice(bonafide_files)
                    bonafide_file = os.path.join(self.base_dir, bonafide_file+'.wav')
                    tst_trl = load_audio(bonafide_file)
                elif nontarget_type == 1:
                    # choose a spoof file from the same speaker
                    spoof_files = self.speaker_spoof[spid]
                    spoof_file = random.choice(spoof_files)
                    spoof_file = os.path.join(self.base_dir, spoof_file+'.wav')
                    tst_trl = load_audio(spoof_file)
                else:
                    # choose a spoof file from a different speaker
                    speakers.remove(spid)
                    spid = random.choice(speakers)
                    spoof_files = self.speaker_spoof[spid]
                    spoof_file = random.choice(spoof_files)
                    spoof_file = os.path.join(self.base_dir, spoof_file+'.wav')
                    tst_trl = load_audio(spoof_file)
            tst_trl = [np.expand_dims(tst_trl, axis=1)]
        elif self.n_class == 4:
            # Now the batch is always contain 5 samples, form 4 pairs 
            # 1 target, 3 nontarget
            label = [0, 1, 1, 1]
            # choose another bonafide file from the same speaker
            bonafide_files = self.speaker_bona[spid]
            bonafide_files.remove(utt)
            bonafide_file = random.choice(bonafide_files)
            bonafide_file = os.path.join(self.base_dir, bonafide_file+'.wav')
            bona_same_speaker = load_audio(bonafide_file)
            
            # choose a bonafide file from a different speaker
            speakers = list(self.speaker_bona.keys())
            speakers.remove(spid)
            spid = random.choice(speakers)
            bonafide_files = self.speaker_bona[spid]
            bonafide_file = random.choice(bonafide_files)
            bonafide_file = os.path.join(self.base_dir, bonafide_file+'.wav')
            bona_diff_speaker = load_audio(bonafide_file)
            
            # choose a spoof file from the same speaker
            spoof_files = self.speaker_spoof[spid]
            spoof_file = random.choice(spoof_files)
            spoof_file = os.path.join(self.base_dir, spoof_file+'.wav')
            spoof_same_speaker = load_audio(spoof_file)
            
            # choose a spoof file from a different speaker
            speakers = list(self.speaker_spoof.keys())
            speakers.remove(spid)
            spid = random.choice(speakers)
            spoof_files = self.speaker_spoof[spid]
            spoof_file = random.choice(spoof_files)
            spoof_file = os.path.join(self.base_dir, spoof_file+'.wav')
            spoof_diff_speaker = load_audio(spoof_file)
            tst_trl = [np.expand_dims(bona_same_speaker, axis=1)] + [np.expand_dims(bona_diff_speaker, axis=1)] + [np.expand_dims(spoof_same_speaker, axis=1)] + [np.expand_dims(spoof_diff_speaker, axis=1)]
        
        tst_trl = nii_wav_aug.batch_pad_for_multiview(
                tst_trl, self.sample_rate, self.trim_length, random_trim_nosil=True, repeat_pad=self.repeat_pad)
        tst_trl = np.concatenate(tst_trl, axis=1)
        anchor = Tensor(anchor)
        tst_trl = Tensor(tst_trl)
        # return the batch data in the format of anchor, others in tst_trl, and label
        return anchor, tst_trl, Tensor(label)
