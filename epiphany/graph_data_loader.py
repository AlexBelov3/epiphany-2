import pandas as pd
import numpy as np
import torch.utils.data
import torch
import pickle
import h5py as h5
import os
from torch_geometric.data import Data

def contact_extraction(m, diag_log_list, distance=100):
    '''
    Args:
        m: position along the diagonal
        diag_log_list: list of diagonal bands from HiC map
        distance: how long do we want the stripe to be, default is 100,
                which covers the interaction within 100*10kb = 1Mb genomic distance

    Returns:
        return
    '''

    a_element = []
    for k in range(distance):
        a1 = k // 2
        a2 = k % 2
        if m - a1 - a2 >= 0 and m - a1 - a2 < len(diag_log_list[k]):
            value = diag_log_list[k][m - a1 - a2]
        else:
            value = 0  # Assign a default value for out-of-bounds indices
        a_element.append(value)
    return a_element

def data_preparation(m, diag_log_list, chip_list, distance=100, chip_res=100, hic_res=10000):
    '''
    m: position along the diagonal
    chip_list: the bw_list we generated above
    chip_res: the resolution of ChIP-seq data (100bp)
    hic_res: the resolution of Hi-C data (10kb)
    distance: distance from diagonal
    '''
    res_ratio = int(hic_res / chip_res)
    contacts = contact_extraction(m, diag_log_list, distance)
    return contacts

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, window_size=14000, chroms=['chr22'], save_dir='/data/leslie/belova1/Epiphany_dataset'):
        save_path_X = os.path.join(save_dir, 'GM12878_X.h5')
        save_path_y = os.path.join(save_dir, 'new_GM12878_y.pickle')

        self.window_size = window_size
        self.num_channels = 5
        self.inputs = {}
        self.labels = {}
        self.chroms = chroms

        print("Loading input:")
        self.inputs = h5.File(save_path_X, 'r')
        print("Loading labels:")
        with open(save_path_y, 'rb') as handle:
            self.labels = pickle.load(handle)

        print(self.labels.keys())
        self.labels = {chrom: self.labels[chrom] for chrom in chroms if chrom in self.labels}

        # Initialize the entire Hi-C contact map for each chromosome with buffering
        self.contact_maps = {}
        for chr in self.chroms:
            contact_map = []
            diag_log_list = self.labels[chr]

            # Check if all diagonals have the same length
            max_length = max(len(diag) for diag in diag_log_list)
            diag_log_list_padded = [np.pad(diag, (0, max_length - len(diag)), mode='constant') for diag in diag_log_list]
            hic_matrix = np.array(diag_log_list_padded).T  # Transpose to get correct shape

            # Buffer the Hi-C map by 100 on both sides
            padded_hic = np.pad(hic_matrix, ((100, 100), (0, 0)), mode='constant')
            for t in range(100, len(padded_hic) - 100):
                contact_vec = data_preparation(t, diag_log_list, self.inputs[chr], distance=100)
                contact_map.append(contact_vec)
                # if t in range(100, 105):
                #     print(contact_vec)
            self.contact_maps[chr] = np.array(contact_map)

    def __len__(self):
        return len(self.chroms)

    def __getitem__(self, index):
        chr = self.chroms[index]

        # Fetch the entire Hi-C contact map for the chromosome
        hic_matrix = self.contact_maps[chr]

        # Create nodes (epigenetic data) and positional encoding
        nodes, pos_enc = self.create_nodes(self.inputs[chr], self.window_size, step_size=100)

        # Create edge_index and edge_attr
        num_nodes = nodes.size(0)
        edge_index, edge_attr = self.create_graph(num_nodes, hic_matrix)

        x = torch.cat([nodes, pos_enc], dim=1)  # Concatenate positional encoding

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def create_nodes(self, chromosome, window_size, step_size):
        """
        Create nodes from the chromosome using a sliding window approach.
        Each node represents a window of size 'window_size', spaced 'step_size' apart.
        """
        num_tracks = self.num_channels
        track_length = chromosome.shape[1]
        # Adjust the number of nodes to account for the padding
        num_nodes = (track_length - window_size) // step_size + 1 + 2  # Adding 2 for the padded ends
        nodes = []
        pos_enc = []

        # Adjust the starting point to account for the padding
        for i in range(-100, (num_nodes - 2) * step_size, step_size):
            window = chromosome[:, max(0, i):min(track_length, i + window_size)]
            if i < 0:
                window = np.pad(window, ((0, 0), (-i, 0)), mode='constant')
            elif i + window_size > track_length:
                window = np.pad(window, ((0, 0), (0, i + window_size - track_length)), mode='constant')
            window = window.flatten()
            nodes.append(window)
            pos_enc.append(i + 100)  # Adjusting positional encoding

        nodes = np.array(nodes)
        pos_enc = np.array(pos_enc).reshape(-1, 1)
        return torch.tensor(nodes, dtype=torch.float32), torch.tensor(pos_enc, dtype=torch.float32)

    def create_graph(self, num_nodes, hic_matrix):
        """
        Create a graph where each node is connected to 99 nodes to its left, 99 nodes to its right, and itself.
        The weights are based on the Hi-C matrix.
        """
        edge_index = []
        edge_attr = []

        for i in range(num_nodes):
            left_neighbors = list(range(max(0, i - 99), i))
            right_neighbors = list(range(i + 1, min(num_nodes, i + 100)))
            neighbors = left_neighbors + [i] + right_neighbors

            for j, neighbor in enumerate(neighbors):
                edge_index.append([i, neighbor])
                if neighbor < i:  # left neighbors
                    edge_attr.append(hic_matrix[i][i - neighbor])
                elif neighbor > i:  # right neighbors
                    edge_attr.append(hic_matrix[i][neighbor - i])
                else:  # self
                    edge_attr.append(hic_matrix[i][0])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float32)
        return edge_index, edge_attr