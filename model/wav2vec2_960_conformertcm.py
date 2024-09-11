import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
try:
    from model.loss_metrics import supcon_loss
    from model.wav2vec2_960 import SSLModel
    from model.conformer_tcm.model import MyConformer
except:
    from .loss_metrics import supcon_loss
    from .conformer_tcm.model import MyConformer
    from .wav2vec2_960 import SSLModel


___author__ = "Phucdt"
__email__ = "phucdt@soongsil.ac.kr"

class DropoutForMC(nn.Module):
    """Dropout layer for Bayesian model
    THe difference is that we do dropout even in eval stage
    """
    def __init__(self, p, dropout_flag=True):
        super(DropoutForMC, self).__init__()
        self.p = p
        self.flag = dropout_flag
        return
        
    def forward(self, x):
        return torch.nn.functional.dropout(x, self.p, training=self.flag)


class Model(nn.Module):
    def __init__(self, args, device, is_train = True):
        super().__init__()
        self.device = device
        self.is_train = is_train
        self.flag_fix_ssl = args['flag_fix_ssl']
        self.contra_mode = args['contra_mode']
        self.loss_type = args['loss_type']
        ####
        # create network wav2vec 2.0
        ####
        # self.ssl_model = SSLModel(self.device)
        ssl_kwargs = args.get('xlsr', {})
        self.ssl_model = SSLModel(self.device, **ssl_kwargs)
        
        self.LL = nn.Linear(self.ssl_model.out_dim, args['conformer']['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.selu = nn.SELU(inplace=True)
        
        self.loss_CE = nn.CrossEntropyLoss()
        
        self.backend=MyConformer(**args['conformer'])

        
        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
        # Post-processing
        
    def _forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        if self.flag_fix_ssl:
            self.ssl_model.is_train = False
            with torch.no_grad():
                x_ssl_feat = self.ssl_model(x.squeeze(-1))
        else:
            self.ssl_model.is_train = True
            x_ssl_feat = self.ssl_model(x.squeeze(-1)) #(bs,frame_number,feat_dim)
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        feats = x
        x = nn.GELU()(x)
        
        # output [batch, 2]
        # emb [batch, 64]
        output, emb = self.backend(x)
        output = F.log_softmax(output, dim=1)
        if (self.is_train):
            return output, feats, emb
        return output
    
    def forward(self, x_big):
        # make labels to be a tensor of [bz]
        # labels = labels.squeeze(0)
        if (x_big.dim() == 3):
            x_big = x_big.transpose(0,1)
            batch, length, sample_per_batch = x_big.shape
            # x_big is a tensor of [length, batch, sample per batch]
            # transform to [length, batch*sample per batch] by concat last dim
            x_big = x_big.transpose(1,2)
            x_big = x_big.reshape(batch * sample_per_batch, length)
        if (self.is_train):
            # x_big is a tensor of [1, length, bz]
            # convert to [bz, length]
            # x_big = x_big.squeeze(0).transpose(0,1)
            output, feats, emb = self._forward(x_big)
            # calculate the loss
            return output, feats, emb
        else:
            # in inference mode, we don't need the emb
            # the x_big now is a tensor of [bz, length]
            print("Inference mode")
            
            return self._forward(x_big)
        
    
    def loss(self, output, feats, emb, labels, config, info=None):
        real_bzs = output.shape[0]
        # print("real_bzs", real_bzs)
        # print("labels", labels)
        L_CE = 1/real_bzs *self.loss_CE(output, labels)
        
        # reshape the feats to match the supcon loss format
        feats = feats.unsqueeze(1)
        # print("feats.shape", feats.shape)
        L_CF1 = 1/real_bzs *supcon_loss(feats, labels=labels, contra_mode=self.contra_mode, sim_metric=self.sim_metric_seq)
        # reshape the emb to match the supcon loss format
        emb = emb.unsqueeze(1)
        emb = emb.unsqueeze(-1)
        # print("emb.shape", emb.shape)
        L_CF2 = 1/real_bzs *supcon_loss(emb, labels=labels, contra_mode=self.contra_mode, sim_metric=self.sim_metric_seq)
        if self.loss_type == 1:
            return {'L_CE':L_CE, 'L_CF1':L_CF1, 'L_CF2':L_CF2}
        elif self.loss_type == 2:
            return {'L_CE':L_CE, 'L_CF1':L_CF1}
        elif self.loss_type == 3:
            return {'L_CE':L_CE, 'L_CF2':L_CF2}
        # ablation study
        elif self.loss_type == 4:
            return {'L_CE':L_CE}
        elif self.loss_type == 5:
            return {'L_CF1':L_CF1, 'L_CF2':L_CF2}
        
