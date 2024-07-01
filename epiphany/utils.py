import os
import pyBigWig
import numpy as np
import pickle
import h5py as h5
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

######## Data Preprocessing Utils ############
def load_chipseq(dir, chrom="chr1", resolution=100):

    """
    Args:
        dir: path to ChipSeq data
        chrom: chromosome to extract (chr1, chr2, ...)
        resolution: resolution in bp

    Returns:
       list of lists containing the 1D sequences of each bigWig file in specified directory
    """

    files = [i for i in os.listdir(dir) if "bigWig" in i]
    print(files)
    idx = [[i for i, s in enumerate(files) if chip in s][0] for chip in ['DNaseI','H3K27ac','H3K4me3','H3K27me3','CTCF']] #, 'SMC3']]
    files = [files[i] for i in idx]
    bw_list = []
    for file in files:

        bwfile = os.path.join(dir,file)
        print(bwfile)
        bw = pyBigWig.open(bwfile)

        value_list = []
        for i in list(range(0,bw.chroms()[chrom]-resolution,resolution)):
            value_list.append(bw.stats(chrom, i, i+resolution)[0])

        value_list = [0 if v is None else v for v in value_list]
        bw_list.append(value_list)

    return bw_list


### DIAGONAL PREDICTION STUFF
# 1. extract orthogonal stripe from the diagonal vector lists
def contact_extraction(m, diag_log_list, distance=100): #100
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
        value = diag_log_list[k][m-a1-a2]
        a_element.append(value)
    return a_element

# def contact_extraction(m, diag_log_list, distance = 100):
#     # horizontal_element = [diag_log_list[k][m] for k in range(distance)]
#     vertical_element = [diag_log_list[k][m-k] for k in range(distance)]
#     return vertical_element


#2. Match ChIP-seq vector with stripe (at same location m)
def data_preparation(m, diag_log_list, chip_list, distance, chip_res = 100, hic_res = 10000):
    '''
    m: position along the diagonal
    chip_list: the bw_list we generated above
    chip_res: the resolution of ChIP-seq data (100bp)
    hic_res: the resolution of Hi-C data (10kb)
    distance: distance from diagonal
    '''
    res_ratio = int(hic_res / chip_res)
    contacts = contact_extraction(m,diag_log_list,distance)
    return contacts #, chip_list[:, (m*res_ratio - int(distance/2)*res_ratio-1000):(m*res_ratio + int(distance/2)*res_ratio)+1000].T

def get_y_vstripe(start, hic, input_size, resolution=1e04):
    hic_start_row = start
    hic_start_col = start + 200 * np.int64(resolution)
    ind_row = np.int64(hic_start_row // resolution)
    ind_col = np.int64(hic_start_col // resolution)
    #vert = hic[ind_row : ind_row + 200, ind_col].toarray()
    vert = hic[ind_row : ind_row + 200][ind_col].toarray()

    hic_start_row = start + 200 * np.int64(resolution)
    hic_start_col = start + 201 * np.int64(resolution)
    ind_row = np.int64(hic_start_row // resolution)
    ind_col = np.int64(hic_start_col // resolution)

    #hori = hic[ind_row, ind_col : ind_col + 200].toarray()
    hori = hic[ind_row][ind_col: ind_col + 200].toarray()

    y = np.append(vert, hori).astype(np.float32)
    y = y.clip(-16, 16)
    return y


def get_y_vstripe_eval(start, hic, input_size, resolution=1e04):
    hic_start_row = start
    hic_start_col = start + 200 * np.int64(resolution)
    ind_row = np.int64(hic_start_row // resolution)
    ind_col = np.int64(hic_start_col // resolution)
    vert = hic[max(0, ind_row) : ind_row + 200, ind_col].toarray()

    desired_length = 200
    num_zeros = desired_length - vert.shape[0]
    vert = np.pad(vert, ((num_zeros, 0), (0, 0)), "constant")

    hic_start_row = start + 200 * np.int64(resolution)
    hic_start_col = start + 201 * np.int64(resolution)
    ind_row = np.int64(hic_start_row // resolution)
    ind_col = np.int64(hic_start_col // resolution)

    hori = hic[ind_row, ind_col : ind_col + 200].toarray()
    num_zeros = desired_length - hori.shape[1]
    hori = np.pad(hori, ((0, 0), (0, num_zeros)), "constant")

    y = np.append(vert, hori).astype(np.float32)
    return y


######## PyTorch Utils ############
def save(net, file_name, num_to_keep=1):
    """Saves the net to file, creating folder paths if necessary.

    Args:
        net(torch.nn.module): The network to save
        file_name(str): the path to save the file.
        num_to_keep(int): Specifies how many previous saved states to keep once this one has been saved.
            Defaults to 1. Specifying < 0 will not remove any previous saves.
    """

    folder = os.path.dirname(file_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), file_name)
    extension = os.path.splitext(file_name)[1]
    checkpoints = sorted(glob.glob(folder + '/*' + extension), key=os.path.getmtime)
    print('Saved %s\n' % file_name)
    if num_to_keep > 0:
        for ff in checkpoints[:-num_to_keep]:
            os.remove(ff)


def restore(net, save_file):
    """Restores the weights from a saved file

    This does more than the simple Pytorch restore. It checks that the names
    of variables match, and if they don't doesn't throw a fit. It is similar
    to how Caffe acts. This is especially useful if you decide to change your
    network architecture but don't want to retrain from scratch.

    Args:
        net(torch.nn.Module): The net to restore
        save_file(str): The file path
    """

    net_state_dict = net.state_dict()
    restore_state_dict = torch.load(save_file)

    restored_var_names = set()

    print('Restoring:')
    for var_name in restore_state_dict.keys():
        if var_name in net_state_dict:
            var_size = net_state_dict[var_name].size()
            restore_size = restore_state_dict[var_name].size()
            if var_size != restore_size:
                print('Shape mismatch for var', var_name, 'expected', var_size, 'got', restore_size)
            else:
                if isinstance(net_state_dict[var_name], torch.nn.Parameter):
                    net_state_dict[var_name] = restore_state_dict[var_name].data
                try:
                    net_state_dict[var_name].copy_(restore_state_dict[var_name])
                    print(str(var_name) + ' -> \t' + str(var_size) + ' = ' + str(int(np.prod(var_size) * 4 / 10**6)) + 'MB')
                    restored_var_names.add(var_name)
                except Exception as ex:
                    print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                              var_name, var_size, restore_size))
                    raise ex

    ignored_var_names = sorted(list(set(restore_state_dict.keys()) - restored_var_names))
    unset_var_names = sorted(list(set(net_state_dict.keys()) - restored_var_names))
    print('')
    if len(ignored_var_names) == 0:
        print('Restored all variables')
    else:
        print('Did not restore:\n\t' + '\n\t'.join(ignored_var_names))
    if len(unset_var_names) == 0:
        print('No new variables')
    else:
        print('Initialized but did not modify:\n\t' + '\n\t'.join(unset_var_names))

    print('Restored %s' % save_file)


def restore_latest(net, folder, ext='.pt'):
    """Restores the most recent weights in a folder

    Args:
        net(torch.nn.module): The net to restore
        folder(str): The folder path
    Returns:
        int: Attempts to parse the epoch from the state and returns it if possible. Otherwise returns 0.
    """

    checkpoints = sorted(glob.glob(folder + '/*' + ext), key=os.path.getmtime)
    start_it = 0
    if len(checkpoints) > 0:
        restore(net, checkpoints[-1])
        try:
            start_it = int(re.findall(r'\d+', checkpoints[-1])[-1])
        except:
            pass
    return start_it


def generate_image(label, pred, path='./', seq_length=1000, bands=200):

    path = os.path.join(path, 'ex.png')
    label = np.squeeze(label.numpy()).T[::-1,:]
    pred = np.squeeze(pred.numpy()).T[::-1,:]
    im1 = np.zeros((seq_length,seq_length))
    for j in range(bands):
        if j > 0:
            np.fill_diagonal(im1[:,j:], pred[bands-1-j,j//2:-j//2])
        else:
            np.fill_diagonal(im1, .5*pred[bands-1-j,:])

    im2 = np.zeros((seq_length,seq_length))
    for j in range(bands-1):
        if j > 0:
            np.fill_diagonal(im2[:,j:], label[bands-1-j,j//2:-j//2])
        else:
            np.fill_diagonal(im2, .5*label[bands-1-j,:])


    plt.imsave(path, im1 + im2.T, cmap='RdYlBu_r', vmin=0) #, vmin=0, vmax=6) #, vmin=-4, vmax=4)
    return plt.imread(path)


# def extract_diagonals(arr):
#     arr = arr.detach().cpu().numpy() if torch.is_tensor(arr) else arr
#     diag_up = np.diagonal(arr, axis1=0, axis2=1).T
#     diag_down = np.fliplr(arr).diagonal(axis1=0, axis2=1).T
#     return diag_up, diag_down

def generate_image_vstripe(label, pred, path='./', seq_length=1000):
    if not isinstance(label, np.ndarray):
        label = label.numpy()
    if not isinstance(pred, np.ndarray):
        pred = pred.numpy()

    path = os.path.join(path, 'ex.png')

    # Extract v-stripe from label and pred
    label_up, label_down = extract_diagonals(label)
    pred_up, pred_down = extract_diagonals(pred)

    # Initialize image
    im = np.zeros((seq_length, seq_length))

    # Fill image with label v-stripe
    for i in range(len(label_up)):
        if i > 0:
            np.fill_diagonal(im[:, i:], label_up[-1 - i, i // 2:-i // 2])
        else:
            np.fill_diagonal(im, .5 * label_up[-1 - i, :])

    for i in range(len(label_down)):
        if i > 0:
            np.fill_diagonal(im[:, i:], label_down[-1 - i, i // 2:-i // 2])
        else:
            np.fill_diagonal(im, .5 * label_down[-1 - i, :])

    # Fill image with pred v-stripe
    for i in range(len(pred_up)):
        if i > 0:
            diag_values = pred_up[-1 - i, i // 2:-i // 2]
            for j in range(len(diag_values)):
                if im[j, j + i] != 0:
                    im[j, j + i] = (im[j, j + i] + diag_values[j]) / 2
                else:
                    im[j, j + i] = diag_values[j]
        else:
            diag_values = .5 * pred_up[-1 - i, :]
            for j in range(len(diag_values)):
                if im[j, j] != 0:
                    im[j, j] = (im[j, j] + diag_values[j]) / 2
                else:
                    im[j, j] = diag_values[j]

    for i in range(len(pred_down)):
        if i > 0:
            diag_values = pred_down[-1 - i, i // 2:-i // 2]
            for j in range(len(diag_values)):
                if im[j, j + i] != 0:
                    im[j, j + i] = (im[j, j + i] + diag_values[j]) / 2
                else:
                    im[j, j + i] = diag_values[j]
        else:
            diag_values = .5 * pred_down[-1 - i, :]
            for j in range(len(diag_values)):
                if im[j, j] != 0:
                    im[j, j] = (im[j, j] + diag_values[j]) / 2
                else:
                    im[j, j] = diag_values[j]

    # Save image
    plt.imsave(path, im, cmap='RdYlBu_r', vmin=0)

    return plt.imread(path)

def safe_tensor_to_numpy(label):
    if isinstance(label, list):
        # Convert each tensor in the list to a numpy array
        label = [item.detach().numpy() if isinstance(item, torch.Tensor) else item for item in label]
        # Stack the list into a single numpy array
        label = np.stack(label)
    elif isinstance(label, torch.Tensor):
        label = label.detach().numpy()
    elif not isinstance(label, np.ndarray):
        label = np.array(label)

    label = np.squeeze(label).T
    return label


def generate_image_test(label, y_up_list, y_down_list, path='./', seq_length=200):
    path = os.path.join(path, 'ex_test.png')
    # label = np.squeeze(label.numpy()).T
    # Ensure label is a numpy array
    label = safe_tensor_to_numpy(label)

    # Extract the diagonals
    # label_up, label_down = extract_diagonals(label.T)
    # label_up = label_up
    # label_down = label_down

    # Initialize the image arrays
    im1 = np.zeros((seq_length, seq_length))
    im2 = np.zeros((seq_length, seq_length))

    # Fill the top diagonal with the reconstructed Hi-C from label diagonals
    for i in range(seq_length):
        diag_values_up = y_up_list[i].cpu()
        diag_values_down = y_down_list[i].cpu()
        for j in range(min(100, i)):
            im1[i, i-j] = diag_values_down[j]
    im1 = im1.T
    for i in range(seq_length):
        for j in range(min(100, seq_length - i)):
            # print(f"j: {j}")
            if im1[i, i+j] == 0:
                # print(f"diag_values_up[j]: {diag_values_up[j]}")
                # print(f"im1[i, i+j]: {im1[i, i+j]}")
                im1[i, i + j] = diag_values_up[j]
            else:
                im1[i, i + j] = np.mean([diag_values_up[j], im1[i, i + j]])

    bands = len(label)
    label = np.flip(label, axis=0)
    for j in range(bands - 1):
        if j > 0:
            np.fill_diagonal(im2[:, j:], label[bands - 1 - j, j // 2:-j // 2])
        else:
            np.fill_diagonal(im2, .5 * label[bands - 1 - j, :])

    # Plot the results
    fig, ax = plt.subplots()
    combined_image = im1 + im2.T
    ax.imshow(combined_image, cmap='RdYlBu_r', vmin=0)
    plt.imsave(path, combined_image, cmap='RdYlBu_r', vmin=0)

    return plt.imread(path)


def test(test_loader, model, device, seq_length):
    '''
    Function for the validation step of the training loop
    '''
    y_hat_list = []
    y_true_list = []
    model.eval()

    with tqdm(test_loader, unit="batch") as tepoch:
        for (X_chr, y_chr, X_chr_rev, y_chr_rev) in tepoch:
            # left interactions
            y_hat, hidden = model(X_chr[0], hidden_state=None,seq_length=seq_length)
            y_hat = y_hat.squeeze()
            y_hat, disregard = extract_diagonals(y_hat)
            # right interactions
            y_hat_rev, hidden = model(X_chr_rev[0], hidden_state=None,seq_length=seq_length)
            y_hat_rev = y_hat_rev.squeeze()
            y_hat_rev, disregard = extract_diagonals(y_hat_rev)
            y_hat = torch.cat((torch.Tensor(y_hat), torch.flip(torch.Tensor(y_hat_rev), [0]))) #dims was set to 1...
            y_hat_list.append(y_hat.detach().cpu())

    return y_hat_list


def test_model(model, test_loader, device, seq_length):
    with torch.no_grad():
        y_hat_list = test(test_loader, model, device, seq_length)
    return y_hat_list

# def extract_diagonals(arr):
#     if isinstance(arr, torch.Tensor):
#         arr = arr.detach().cpu().numpy()
#     elif not isinstance(arr, np.ndarray):
#         arr = np.array(arr)
#     assert arr.shape == (200, 100), "Input array must be 200x100 in size"
#
#     up_diagonal = np.zeros(100)
#     down_diagonal = np.zeros(100)
#
#     for i in range(100):
#         up_diagonal[i] = arr[99 + i//2, i]
#         down_diagonal[i] = arr[99 - i//2, i]
#         # up_diagonal[i] = arr[0, i]
#         # down_diagonal[i] = arr[i, 0]
#
#     return up_diagonal, down_diagonal

def extract_diagonals(tensor):
    assert tensor.shape == (200, 100), "Input tensor must be 200x100 in size"

    device = tensor.device
    dtype = tensor.dtype

    up_diagonal = torch.zeros(100, device=device, dtype=dtype)
    down_diagonal = torch.zeros(100, device=device, dtype=dtype)

    for i in range(100):
        up_diagonal[i] = tensor[99 + i // 2, i]
        down_diagonal[i] = tensor[99 - i // 2, i]

    return up_diagonal, down_diagonal


def cpu_jaccard_vstripe(x):
    # calculate jaccard similarity of rows
    scatac_res = 500
    size = x.shape[1]
    eps = 1e-8
    i = np.int16(1000 / scatac_res)

    x = torch.where(x > 0.0, torch.tensor([1.0]), torch.tensor([0.0]))
    num = torch.mm(
        x[2000 * i : 2010 * i, :],
        x[np.r_[:, 0 : 2000 * i, 2010 * i : 4010 * i]].transpose(0, 1),
    )

    x = torch.where(x == 0.0, torch.tensor([1.0]), torch.tensor([0.0]))
    denom = torch.mm(
        x[2000 * i : 2010 * i, :],
        x[np.r_[:, 0 : 2000 * i, 2010 * i : 4010 * i]].transpose(0, 1),
    )
    denom = size - denom

    num = torch.div(num, torch.max(denom, eps * torch.ones_like(denom)))

    return num


def cpu_batch_corcoeff_vstripe(x):
    c = cpu_jaccard_vstripe(x.permute(1, 0))
    c[c != c] = 0
    return c


def bin_and_sum(arr, bin_size=100):
    # Ensure the array length is a multiple of bin_size
    n = len(arr)
    remainder = n % bin_size
    if remainder != 0:
        arr = arr[:n - remainder]

    # Reshape the array to have bin_size columns
    reshaped_arr = arr.reshape(-1, bin_size)
    # Calculate the mean across the columns
    binned_sum = reshaped_arr.sum(axis=1)

    return binned_sum