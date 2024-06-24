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
    def __init__(self, seq_length=200, window_size=14000, chroms=['chr22'], mode='train', save_dir='/data/leslie/belova1//Epiphany_dataset', zero_pad=True):

        save_path_X = os.path.join(save_dir, 'GM12878_X.h5')
        save_path_y = os.path.join(save_dir, 'new_GM12878_y.pickle')

        self.seq_length = seq_length
        self.chroms = chroms
        self.buf = 100
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
            # self.sizes.append((len(diag_log_list[0]) - 2*self.buf)//self.seq_length + 1)
            self.sizes.append((len(diag_log_list[0]) - 2 * self.buf) + 1)

        print(self.sizes)

        # for chr in self.chroms:
        #     for signal in self.inputs[chr]:
        #         for i in range ()
        #             print(f"inputs[chr] shape: {np.shape(self.inputs[chr])}")
        #             print(f"number of signals: {len(self.inputs[chr])}")
        #             print(f"signal type: {type(signal)}")
        #             print(f"signal shape: {np.shape(signal)}")
        #             self.co_signals.append(np.outer(signal.astype('float16'), signal.astype('float16')))

        return

    def __len__(self):

        return int(np.sum(self.sizes))


    def __getitem__(self, index):

        arr = np.array(np.cumsum(self.sizes).tolist())
        arr[arr <= index] = 100000
        chrom_idx = np.argmin(arr)
        chr = self.chroms[chrom_idx]
        idx = int(index - ([0] + np.cumsum(self.sizes).tolist())[chrom_idx])
        # start = idx*self.seq_length + self.buf
        start = idx + self.buf
        # end = np.minimum(idx*self.seq_length + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf)
        end = np.minimum(idx + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf)
        contact_data = []

        for t in range(idx + self.buf, np.minimum(idx + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf),1):
            contact_vec = data_preparation(t,self.labels[chr],self.inputs[chr], distance=100)
            contact_data.append(contact_vec)

        X_chr = self.inputs[chr][:self.num_channels, 100*start-(self.window_size//2):100*end+(self.window_size//2)].astype('float32') #100

        y_chr = np.array(contact_data)

        if self.zero_pad and y_chr.shape[0] < self.seq_length:

            try:
                pad_y = np.zeros((self.seq_length - y_chr.shape[0], y_chr.shape[1]))
                y_chr = np.concatenate((y_chr, pad_y), axis=0)
            except:
                y_chr = np.zeros((self.seq_length,100)) #100

            pad_X = np.zeros((X_chr.shape[0],self.seq_length*100+self.window_size - X_chr.shape[1])) #100
            X_chr = np.concatenate((X_chr, pad_X), axis=1)
        # if chr not in self.co_signals.keys():
        #     self.co_signals[chr] = []
        # if len(self.co_signals[chr]) == 0:
        #     t0 = time.time()
        #     MAX_LEN = np.shape(self.inputs[chr])[1]  # Maximum length to extend
        #     n = len(X_chr)
        #     print("Pre-compute co-signal matrix:")
        #     # X_chr = np.arange(1, 34000)  # Original array
        #     # Convert numpy arrays to PyTorch tensors and move them to the GPU
        #     X_chr_tensor = torch.Tensor(X_chr[0])#.cuda()
        #     # Perform outer product on GPU
        #     co_signals_tensor = torch.outer(X_chr_tensor, X_chr_tensor)
        #     # Move result back to CPU for further processing
        #     co_signals = co_signals_tensor.cpu().numpy()
        #     L = MAX_LEN + co_signals.shape[1]
        #     self.co_signals[chr] = np.zeros((2 * n - 1, L))
        #     for i in range(-n, n):
        #         diagonal = np.diagonal(co_signals, offset=i)
        #         self.co_signals[chr][n - 1 + i][abs(i):len(diagonal) + abs(i)] = diagonal
        #     for index in range(i,MAX_LEN+1):
        #         new_X_chr = np.arange(index + 1, index + len(X_chr) + 1)
        #         new_X_chr_tensor = torch.tensor(new_X_chr, dtype=torch.float32)#.cuda()
        #         # Perform element-wise multiplication on GPU
        #         new_prod_tensor = new_X_chr_tensor[-1] * new_X_chr_tensor
        #         # Move result back to CPU
        #         new_prod = new_prod_tensor.cpu().numpy()
        #         # Update self.co_signals with the new products
        #         self.co_signals[chr][:, len(X_chr) + index - 1][:len(new_prod)] = new_prod
        #         self.co_signals[chr][:, len(X_chr) + index - 1][len(new_prod) - 1:] = new_prod[::-1]  # (reversed)
        #     print(time.time() - t0)
        binned_signals = []
        for i in range(np.shape(X_chr)[0]):
            binned_signals.append(bin_and_sum(X_chr[i], 100))
        co_signal = np.outer(binned_signals, binned_signals)
        # co_signal = np.zeros_like(co_signal)
        # print(f"co_signal shape: {np.shape(co_signal)}")
        return X_chr.astype('float32'), y_chr.astype('float32'), co_signal.astype('float32')#, self.co_signals[chr][:, index:len(X_chr) + index] .astype('float32')

