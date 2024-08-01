import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import argparse
from utils import *
import time
from data_loader_10kb import *
from model_10kb import *
# from model_10kb_Vs import *
from tqdm import tqdm
import subprocess
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



print(torch.__version__)


def main():
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="CUDA ID", default="0")
    parser.add_argument("--b", help="batch size", default="1")
    parser.add_argument("--e", help="number of epochs", default="55")
    parser.add_argument("--lr", help="initial learning rate", default="1e-4") #1e-6
    parser.add_argument("--v", help="experiment version", default="0.1")
    parser.add_argument("--lam", help="tradeoff between l2 and adversarial loss", default="0.95")
    parser.add_argument("--window_size", help="Context (in terms of 100kb) for each orthogonal vector", default="24000") #14000
    parser.add_argument("--m", help="additional comments", default="")
    parser.add_argument("--high_res", action='store_true', help="Use if predicting 5kb resolution Hi-C (10kb is used by default)")
    parser.add_argument('--wandb', action='store_true', help='Toggle wandb')
    parser.add_argument('model', choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

    args = parser.parse_args()


    '''
    if args.high_res:
        from data_loader_5kb import *
        from model_5kb import *
    else:
        from data_loader_10kb import *
        from model_10kb import *
    '''

    print("Run: " + args.m)

    LEARNING_RATE = float(args.lr)
    EXPERIMENT_VERSION = args.v
    LOG_PATH = './logs/' + EXPERIMENT_VERSION + '/'
    LAMBDA = float(args.lam)
    TRAIN_SEQ_LENGTH = 200
    TEST_SEQ_LENGTH = 200
    NUM_Vs = 1

    torch.cuda.set_device(int(args.gpu))
    torch.manual_seed(0)

    if args.model == 'a':
        # chromafold right arm with only conv1d
        model_name = "branch_cov_" + str(NUM_Vs)
        model = branch_cov(num_Vs=NUM_Vs).cuda()
        # model = branch_cov().cuda()
    elif args.model == 'b':
        # chromafold right arm with only conv1d
        model_name = "branch_big_cov"
        model = branch_big_cov().cuda()
    elif args.model == 'c':
        # branch_pbulk with outer product instead of symmetrize_bulk
        model_name = "branch_pbulk_prod"  # _" + str(NUM_Vs)
        model = branch_pbulk_prod().cuda()
    elif args.model == 'd':
        # conv1d right arm ad outer_prod_small right arm
        model_name = "epiphany_2"  # _" + str(NUM_Vs)
        model = trunk(branch_outer_prod_small().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'e':
        # conv1d right arm ad outer_prod_big right arm
        model_name = "outer_prod_big"  # _" + str(NUM_Vs)
        model = trunk(branch_outer_prod_big().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'f':
        # branch_pbulk
        model_name = "branch_pbulk"  # _" + str(NUM_Vs)
        model = branch_pbulk().cuda()
    elif args.model == 'g':
        model_name = "high_res_prod"  # _" + str(NUM_Vs)
        model = trunk(branch_outer_prod_high_res().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'h':
        model_name = "big_cov_plus_transformer"
        model = trunk(branch_transformer().cuda(), branch_big_cov().cuda()).cuda()
    elif args.model == 'i':
        model_name = "branch_small_pbulk"
        model = branch_small_pbulk().cuda()
    elif args.model == 'j':
        model_name = "branch_transformer"  # _" + str(NUM_Vs)
        model = trunk(branch_transformer().cuda(), branch_cov().cuda()).cuda()
    else:
        model_name = "DEFAULT"
        model = Net(1, 5, int(args.window_size)).cuda()
    model_name = model_name
    print(f"Beginning testing {model_name}")
    if args.wandb:
        import wandb
        wandb.init(project=(model_name+"_EVAL"),)

    if args.wandb:
        wandb.watch(model, log='all')

    if os.path.exists(LOG_PATH):
        restore_latest(model, LOG_PATH, ext='.pt_' + model_name)
    else:
        os.makedirs(LOG_PATH)

    eval_length = 800
    # GM12878 Standard
    test_chroms = ['chr3', 'chr11', 'chr17', 'chr2', 'chr22']

    for chr in test_chroms:
        y_list = []
        y_up_list = []
        y_down_list = []
        labels = []
        co_signal = []
        test_set = Chip2HiCDataset(seq_length=TEST_SEQ_LENGTH, window_size=int(args.window_size), chroms=[chr],
                                   mode='test')
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
        for i, (test_data, test_label, co_s) in enumerate(test_loader):
            # if i >= eval_length:
            #     break
            # y_up_list.append(y)
            # y_down_list.append(y_rev)

            # labels.append(test_label[100])
            # if i >= eval_length // NUM_Vs:
            #     break
            test_label = test_label.squeeze()
            y, y_rev = extract_n_diagonals(test_label, NUM_Vs)
            # y, y_rev = torch.cat(y, dim=0), torch.cat(y_rev, dim=0)
            for j in range(NUM_Vs):
                y_up_list.append(y[j])
                y_down_list.append(y_rev[j])
                y_list.append(np.concatenate((y[j], y_rev[j]), axis=0))
            # labels.append([test_label[100], test_label[101]])
            # print(f"label shape: {np.shape(test_label[100 - (NUM_Vs-1)//2 :100 + NUM_Vs//2 + 1])}")
            # labels.append(test_label[100 - (NUM_Vs-1)//2 :100 + NUM_Vs//2 + 1])
            labels.append(test_label[100: 100 + NUM_Vs])


        y_hat_L_list = []
        y_hat_R_list = []
        y_hat_list = []
        model.eval()
        # i = 0
        for (test_data, test_label, co_s) in tqdm(test_loader):
            # if i < eval_length:
                #     if np.linalg.norm(test_label) < 1e-8:
                #         continue
                #     test_data, test_label = torch.Tensor(test_data).cuda(), torch.Tensor(test_label).cuda()  # NEW!!!!
                #     with torch.no_grad():
                #         y_hat = model(test_data)
                #         y_hat_list.append(np.array(y_hat.cpu().squeeze()))
                #         y_hat_L_list.append(torch.tensor(np.array(y_hat.cpu())[0][:100]))
                #         y_hat_R_list.append(torch.tensor(np.array(y_hat.cpu())[0][100:]))
                # if i < eval_length//NUM_Vs:
            if np.linalg.norm(test_label) < 1e-8:
                continue
            test_data, test_label = torch.Tensor(test_data).cuda(), torch.Tensor(test_label).cuda()
            with torch.no_grad():
                y_hat = model(test_data)#.squeeze()
                if NUM_Vs != 1:
                    y_hat = y_hat[0]
                for j in range(NUM_Vs):
                    y_hat_list.append(np.array(y_hat.cpu().squeeze()))
                    y_hat_L_list.append(torch.tensor(np.array(y_hat.cpu())[j][:100]))
                    y_hat_R_list.append(torch.tensor(np.array(y_hat.cpu())[j][100:]))
            # else:
            #     break
            # i += 1
        if args.wandb:
            im = wandb.Image(generate_image_test(labels, y_hat_L_list, y_hat_R_list, path=LOG_PATH, seq_length=len(labels)-1))
            wandb.log({chr + " Evaluation Examples": im})

            # for i in range(len(co_signal)):
            #     im = wandb.Image(
            #         plot_cosignal_matrix(co_signal[i]))
            #     wandb.log({f"{i} " + chr + " Co-Signal": im})
        np.savetxt("hic_real.tsv", generate_hic_true(labels, path=LOG_PATH, seq_length=len(labels)-1), delimiter="\t", fmt="%.6f")
        np.savetxt("hic_pred.tsv", generate_hic_hat(y_hat_L_list, y_hat_R_list, path=LOG_PATH, seq_length=len(labels)-1),
                   delimiter="\t", fmt="%.6f")
        cwd = os.getcwd()
        # Define paths
        r_script_name = "insulation.R"
        r_script_path = os.path.join(cwd, r_script_name)
        real_hic_matrix_path = os.path.join(cwd, "hic_real.tsv")
        pred_hic_matrix_path = os.path.join(cwd, "hic_pred.tsv")
        real_output_data_path = os.path.join(cwd, "real_output_data")
        pred_output_data_path = os.path.join(cwd, "pred_output_data")
        # Full path to Rscript executable
        rscript_executable = "./Rscript"
        rscript_executable = os.path.join(cwd, rscript_executable)
        try:
            subprocess.run([rscript_executable, r_script_path, real_hic_matrix_path, real_output_data_path],
                                    capture_output=True, text=True)
            subprocess.run([rscript_executable, r_script_path, pred_hic_matrix_path, pred_output_data_path],
                                    capture_output=True, text=True)
        except Exception as e:
            print(f"An error occurred while running the R script: {e}")

        # Calculate insulation scores using the output data from the R script
        bins_real_path = f"{real_output_data_path}_bins.tsv"
        counts_real_path = f"{real_output_data_path}_counts.tsv"
        bins_pred_path = f"{pred_output_data_path}_bins.tsv"
        counts_pred_path = f"{pred_output_data_path}_counts.tsv"

        # Load the bins and counts data
        bins = pd.read_csv(bins_real_path, sep="\t")
        counts = np.loadtxt(counts_real_path, delimiter="\t")
        # Calculate insulation scores
        insulation_scores = calculate_insulation_scores(bins, counts)
        real_insulation_scores = np.log2(insulation_scores + 1e-10)
        # Plot log2 insulation scores

        bins = pd.read_csv(bins_pred_path, sep="\t")
        counts = np.loadtxt(counts_pred_path, delimiter="\t")
        # Calculate insulation scores
        insulation_scores = calculate_insulation_scores(bins, counts)
        pred_insulation_scores = np.log2(insulation_scores + 1e-10)

        if args.wandb:
            wandb.log({chr + " Insulation Score": wandb.Image(plot_two_insulation_scores(real_insulation_scores, pred_insulation_scores))})

        correlation_list = []
        for i in range(len(y_list)):
            corr_matrix = np.corrcoef(y_hat_list[i], y_list[i], rowvar=True)
            correlation = corr_matrix[0, 1]
            correlation_list.append(correlation)
        corr = np.corrcoef(np.ravel(y_hat_list), np.ravel(y_list))[0, 1]
        if args.wandb:
            wandb.log({chr + " Correlation Across Vs": wandb.Image(plot_correlation(correlation_list, corr))})

        y_list_by_distance = []
        y_hat_list_by_distance = []
        for j in range(100):
            y_list_distance = []
            y_hat_list_distance = []
            for i in range(len(y_list)):
                y_list_distance.append(y_up_list[i][j])
                y_list_distance.append(y_down_list[i][j])
                y_hat_list_distance.append(y_hat_L_list[i][j])
                y_hat_list_distance.append(y_hat_R_list[i][j])
            y_list_by_distance.append(y_list_distance)
            y_hat_list_by_distance.append(y_hat_list_distance)

        correlation_list = []
        for i in range(len(y_list_by_distance)):
            corr_matrix = np.corrcoef(y_list_by_distance[i], y_hat_list_by_distance[i], rowvar=True)
            correlation = corr_matrix[0, 1]
            correlation_list.append(correlation)
        corr = np.corrcoef(np.ravel(y_hat_list), np.ravel(y_list))[0, 1]
        if args.wandb:
            wandb.log({chr + " Correlation Across Distance from Diag": wandb.Image(plot_correlation(correlation_list))})
if __name__ == '__main__':
    main()
