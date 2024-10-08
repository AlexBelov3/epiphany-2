from utils import *

import pandas as pd
import numpy as np
import torch.utils.data 
import torch
import pickle
import h5py as h5
import os
from torch.autograd import Variable
import h5py 
import time
import sys
#wandb.init()

class Chip2HiCDataset(torch.utils.data.Dataset):
    def __init__(self, seq_length=200, window_size=14000, chroms=['chr22'], mode='train', save_dir='./Epiphany_dataset', zero_pad=True, num_Vs = 1):

        save_path_X = os.path.join(save_dir, 'GM12878_X.h5')
        # save_path_y = os.path.join(save_dir, 'new_GM12878_y.pickle')
        save_path_y = os.path.join(save_dir, 'GM12878_y.pickle')

        self.seq_length = seq_length
        self.num_Vs = num_Vs
        self.chroms = chroms
        self.buf = 200 #100
        self.window_size = window_size
        self.num_channels = 5
        self.inputs = {}
        self.labels = {}
        self.sizes = []
        self.zero_pad = zero_pad
        self.co_signals = {}

        print("Loading input:")
        self.inputs = h5.File(save_path_X, 'r')
        print("Loading labels:")
        with open(save_path_y, 'rb') as handle:
            self.labels = pickle.load(handle)
        print(self.labels.keys())
        self.labels = {chrom: self.labels[chrom] for chrom in chroms if chrom in self.labels}

        for chr in self.chroms:
            diag_log_list = self.labels[chr]
            print(len(diag_log_list[0]))
            self.sizes.append((len(diag_log_list[0]) - 2*self.buf)//self.num_Vs + 1)
            # self.sizes.append((len(diag_log_list[0]) - 2 * self.buf) + 1)

        print(self.sizes)

        return

    def __len__(self):

        return int(np.sum(self.sizes))


    def __getitem__(self, index):

        arr = np.array(np.cumsum(self.sizes).tolist())
        arr[arr <= index*self.num_Vs] = 100000
        chrom_idx = np.argmin(arr)
        chr = self.chroms[chrom_idx]
        idx = int(index*self.num_Vs - ([0] + np.cumsum(self.sizes).tolist())[chrom_idx])
        start = idx*self.num_Vs + self.buf
        # start = idx + self.buf
        end = np.minimum(idx*self.num_Vs + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf)
        # end = np.minimum(idx + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf)
        contact_data = []
        # for t in range(idx + self.buf, np.minimum(idx + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf),1):
        for t in range(idx + self.buf, np.minimum(idx + self.seq_length + self.num_Vs-1 + self.buf, len(self.labels[chr][0]) - self.buf), 1):
            contact_vec = data_preparation(t,self.labels[chr],self.inputs[chr], distance=100)
            contact_data.append(contact_vec)

        X_chr = self.inputs[chr][:self.num_channels, 100*start-(self.window_size//2):100*end+(self.window_size//2)].astype('float32')
        # X_chr = self.inputs[chr][:self.num_channels,
        #         (100 + self.num_Vs) * start - (self.window_size // 2):(100 + self.num_Vs) * end + (
        #                     self.window_size // 2)].astype('float32')
        y_chr = np.array(contact_data)
        if self.zero_pad and y_chr.shape[0] < self.seq_length:
            try:
                pad_y = np.zeros((self.seq_length - y_chr.shape[0], y_chr.shape[1]))
                y_chr = np.concatenate((y_chr, pad_y), axis=0)
            except:
                y_chr = np.zeros((self.seq_length,100)) #100

            pad_X = np.zeros((X_chr.shape[0],self.seq_length*100+self.window_size - X_chr.shape[1])) #100
            X_chr = np.concatenate((X_chr, pad_X), axis=1)

        co_signal = []
        return X_chr.astype('float32'), y_chr.astype('float32'), co_signal

