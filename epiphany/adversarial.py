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
from model_10kb_Vs import *
# from model_10kb import *
from tqdm import tqdm
import subprocess

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
    NUM_Vs = 2

    torch.cuda.set_device(int(args.gpu))
    torch.manual_seed(0)

    if args.model == 'a':
        # chromafold right arm with only conv1d
        model_name = "branch_cov"
        model = branch_cov(num_Vs=NUM_Vs).cuda()
        # model = branch_cov().cuda()
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
        model_name = "branch_LSTM"
        model = trunk(branch_BiLSTM().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'i':
        model_name = "branch_outer_prod_learned"
        model = trunk_new_loss(branch_outer_prod_learned().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'j':
        model_name = "branch_transformer"
        model = trunk(branch_transformer().cuda(), branch_cov().cuda()).cuda()
    else:
        model_name = "DEFAULT"
        model = Net(1, 5, int(args.window_size)).cuda()
    print(f"Beginning training {model_name}")
    if args.wandb:
        import wandb
        wandb.init(project=model_name,)


    disc = Disc()#.cuda()
    if args.wandb:
        wandb.watch(model, log='all')

    if os.path.exists(LOG_PATH):
        restore_latest(model, LOG_PATH, ext='.pt_' + model_name)
    else:
        os.makedirs(LOG_PATH)

    with open(os.path.join(LOG_PATH, 'setup.txt'), 'a+') as f:
        f.write("\nVersion: " + args.v)
        f.write("\nBatch Size: " + args.b)
        f.write("\nInitial Learning Rate: " + args.lr)
        f.write("\nComments: " + args.m)


    # GM12878 Standard
    test_chroms = ['chr3', 'chr11', 'chr17']
    # match test chroms with chromafold!!
    # test_chroms = ['chr17']
    train_chroms = ['chr1', 'chr2', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']
    # train_chroms = ['chr19', 'chr20', 'chr21', 'chr22']
    # train_chroms = ['chr22']

    train_set = Chip2HiCDataset(seq_length=TRAIN_SEQ_LENGTH, window_size=int(args.window_size), chroms=train_chroms, mode='train', num_Vs=NUM_Vs)
    test_set = Chip2HiCDataset(seq_length=TEST_SEQ_LENGTH, window_size=int(args.window_size), chroms=test_chroms, mode='test', num_Vs=NUM_Vs)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    train_log = os.path.join(LOG_PATH, 'train_log.txt')
    test_log = os.path.join(LOG_PATH, 'test_log.txt')

    hidden = None
    log_interval = 50
    parameters = list(model.parameters())
    optimizer = optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=0.0005)
    disc_optimizer = optim.Adam(disc.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    min_loss = -10

    t0 = time.time()

    y_up_list = []
    y_down_list = []
    labels = []
    for i, (test_data, test_label, co_signal) in enumerate(test_loader):
        test_label = test_label.squeeze()
        y, y_rev = extract_n_diagonals(test_label, NUM_Vs)
        # y, y_rev = extract_diagonals(test_label)
        y_up_list.append(y[0])
        y_down_list.append(y_rev[0])
        # y_up_list.append(y)
        # y_down_list.append(y_rev)
        labels.append(test_label[100])
        if i > 400:
            break
    #
    if args.wandb:
        im = wandb.Image(generate_image_test(labels, y_up_list, y_down_list, path=LOG_PATH, seq_length=400))
        wandb.log({"Validation Examples": im})

    #scaler = torch.cuda.amp.GracddScaler()
    for epoch in range(int(args.e)):
        disc_preds_train = []

        lr = np.maximum(LEARNING_RATE * np.power(0.5, (int(epoch / 16))), 1e-6) # learning rate decay
        # optimizer = optim.Adam(parameters, lr=lr, weight_decay=0.0005)
        disc_optimizer = optim.Adam(disc.parameters(), lr=lr, weight_decay=0.0005)

        print("="*10 + "Epoch " + str(epoch) + "="*10)

        im = []
        test_loss = []
        preds = []
        labs = []
        y_hat_L_list = []
        y_hat_R_list = []
        model.eval()

        if epoch % 1 == 0:
            i = 0
            for (test_data, test_label, co_signal) in tqdm(test_loader):
                if i < 400:
                    if np.linalg.norm(test_label) < 1e-8:
                        continue
                    test_data, test_label = torch.Tensor(test_data).cuda(), torch.Tensor(test_label).cuda() #NEW!!!!
                    with torch.no_grad():
                        y_hat = model(test_data)

                        y_hat_L_list.append(torch.tensor(np.array(y_hat.cpu())[0][:100]))
                        y_hat_R_list.append(torch.tensor(np.array(y_hat.cpu())[0][100:]))

                        test_label_L, test_label_R = extract_diagonals(test_label.squeeze()) # ONLY LOOKING AT THE LEFT VECTOR
                        test_label = torch.concat((test_label_L, test_label_R), dim=0)
                        loss = model.loss(y_hat, test_label)
                        test_loss.append(loss)
                else:
                    break
                i += 1 # test

            if args.wandb:
                im.append(
                    wandb.Image(generate_image_test(labels, y_hat_L_list, y_hat_R_list, path=LOG_PATH,
                                                    seq_length=400)))  # TEST_SEQ_LENGTH
        test_loss_cpu = torch.stack(test_loss).cpu().numpy()
        if args.wandb:
            wandb.log({"Validation Examples": im})
            wandb.log({'val_correlation': np.mean(test_loss_cpu)})

        print('Test Loss: ', np.mean(test_loss_cpu), ' Best: ', str(min_loss))

        if np.mean(test_loss_cpu) > min_loss:
            min_loss = np.mean(test_loss_cpu)
        save(model, os.path.join(LOG_PATH, ('%03d.pt_' + model_name) % epoch), num_to_keep=3)

        with open(test_log, 'a+') as f:
            f.write(str(np.mean(test_loss_cpu)) + "\n")

        model.train()
        disc.train()
        for batch_idx, (data, label, co_signal) in enumerate(train_loader):
            if (np.linalg.norm(data)) < 1e-8 or data.shape[2]!=int(args.window_size)+20000:
                continue

            hidden = None
            label = torch.tensor(np.squeeze(label), requires_grad=True).cuda()
            data = data.cuda()
            optimizer.zero_grad()
            output = model(data)

            label_1d_v_up, label_1d_v_down = extract_diagonals(label)
            label = torch.concat((label_1d_v_up, label_1d_v_down), dim=0)

            mse_loss = model.loss(output, label, seq_length=TRAIN_SEQ_LENGTH)

            loss = (LAMBDA)*mse_loss #+ (1 - LAMBDA)*adv_loss

            initial_params = {name: param.clone() for name, param in model.named_parameters()}
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"No gradients for {name}")

            optimizer.step()

            for name, param in model.named_parameters():
                if torch.equal(param, initial_params[name]):
                    print(f"Parameter {name} has NOT been updated.")


            if args.wandb:
                wandb.log({'mse_loss': mse_loss.item()})

            if batch_idx % log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))

    t1 = time.time()
    print(t1 - t0)

if __name__ == '__main__':
    main()
