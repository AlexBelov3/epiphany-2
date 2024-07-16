import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import argparse
from utils import *
import time
from data_loader_10kb import *
from model_10kb import *
from tqdm import tqdm

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
    parser.add_argument('model', choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])

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

    torch.cuda.set_device(int(args.gpu))
    torch.manual_seed(0)

    if args.model == 'a':
        # chromafold right arm with only conv1d
        model_name = "branch_cov"
        model = branch_cov().cuda()
    elif args.model == 'b':
        # chromafold conv1d all --> conv2d
        model_name = "branch_cov_2d"
        model = branch_cov_2d().cuda()
    elif args.model == 'c':
        # modified epiphany (without .as_strided())
        model_name = "epiphany1.1"
        model = Net2(1, 5, int(args.window_size)).cuda()
    elif args.model == 'd':
        # conv1d right arm ad outer_prod_small right arm
        model_name = "epiphany_2"
        model = trunk(branch_outer_prod_small().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'e':
        # conv1d right arm ad outer_prod_big right arm
        model_name = "outer_prod_big"
        model = trunk(branch_outer_prod_big().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'f':
        # use a trunk that adds the two predicitons together? idk
        model_name = "add_trunk"
        model = add_trunk(branch_outer_prod_small().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'g':
        model_name = "high_res_prod"
        model = trunk(branch_outer_prod_high_res().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'h':
        model_name = "trunk_new_loss"
        model = trunk_new_loss(branch_outer_prod_small().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'i':
        model_name = "branch_outer_prod_learned"
        model = trunk_new_loss(branch_outer_prod_learned().cuda(), branch_cov().cuda()).cuda()
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
    # match test chroms with chromafold!!

    for chr in test_chroms:
        y_up_list = []
        y_down_list = []
        labels = []
        test_set = Chip2HiCDataset(seq_length=TEST_SEQ_LENGTH, window_size=int(args.window_size), chroms=[chr],
                                   mode='test')
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
        for i, (test_data, test_label, co_signal) in enumerate(test_loader):
            if i >= eval_length:
                break
            test_label = test_label.squeeze()
            y, y_rev = extract_diagonals(test_label)
            y_up_list.append(y)
            y_down_list.append(y_rev)
            labels.append(test_label[100])


        # if args.wandb:
        #     im = wandb.Image(generate_image_test(labels, y_up_list, y_down_list, path=LOG_PATH, seq_length=eval_length))
        #     wandb.log({chr + " Evaluation Examples": im})

        im = []
        y_hat_L_list = []
        y_hat_R_list = []
        model.eval()
        i = 0
        for (test_data, test_label, co_signal) in tqdm(test_loader):
            if i < eval_length:
                if np.linalg.norm(test_label) < 1e-8:
                    continue
                test_data, test_label = torch.Tensor(test_data).cuda(), torch.Tensor(test_label).cuda()  # NEW!!!!
                with torch.no_grad():
                    y_hat = model(test_data)
                    y_hat_L_list.append(torch.tensor(np.array(y_hat.cpu())[0][:100]))
                    y_hat_R_list.append(torch.tensor(np.array(y_hat.cpu())[0][100:]))
            else:
                break
            i += 1

        if args.wandb:
            # im.append(
            #     wandb.Image(generate_image_test(labels, y_hat_L_list, y_hat_R_list, path=LOG_PATH,
            #                                     seq_length=eval_length)))
            im.append(
                wandb.Image(generate_hic_hat(y_hat_L_list, y_hat_R_list, path=LOG_PATH, seq_length=eval_length)))
            wandb.log({chr + " Evaluation Examples": im})


if __name__ == '__main__':
    main()
