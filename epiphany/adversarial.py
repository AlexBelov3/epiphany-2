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
    parser.add_argument("--window_size", help="Context (in terms of 100kb) for each orthogonal vector", default="20000") #14000
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


    # import numpy as np
    # import time
    # t0 = time.time()
    # X_chr = np.arange(1, 34000)  # 6
    #
    # MAX_LEN = 1  # 25
    # co_signals = np.outer(X_chr, X_chr)
    # print("-" * 40)
    #
    # # print(co_signals)
    # n = len(X_chr)
    # L = MAX_LEN + co_signals.shape[1]
    # new_matrix = np.zeros((2 * n - 1, L))
    #
    # for i in range(-n, n):
    #     diagonal = np.diagonal(co_signals, offset=i)
    #     new_matrix[n - 1 + i][abs(i):len(diagonal) + abs(i)] = diagonal
    # # print(new_matrix)
    #
    # # ADD NEW PRODUCTS
    # for index in range(1, MAX_LEN + 1):
    #     new_X_chr = np.arange(index + 1, index + len(X_chr) + 1)
    #     new_prod = [new_X_chr[-1]] * new_X_chr
    #     new_matrix[:, len(X_chr) + index - 1][:len(new_prod)] = new_prod
    #     new_matrix[:, len(X_chr) + index - 1][len(new_prod) - 1:] = new_prod[::-1]  # (reversed)
    # # print(new_matrix)
    # print(time.time() - t0)


    print("Run: " + args.m)

    LEARNING_RATE = float(args.lr)
    EXPERIMENT_VERSION = args.v
    LOG_PATH = './logs/' + EXPERIMENT_VERSION + '/'
    LAMBDA = float(args.lam)
    TRAIN_SEQ_LENGTH = 200
    TEST_SEQ_LENGTH = 200

    torch.cuda.set_device(int(args.gpu))
    torch.manual_seed(0)
    # model = Net(1, 5, int(args.window_size)).cuda()
    # TESTING:
    # Define Model
    # mod_branch_pbulk = nn.DataParallel(branch_pbulk(), device_ids=[0])
    # mod_branch_cov = nn.DataParallel(branch_cov(), device_ids=[0])
    # mod_branch_cov = branch_cov().cuda()
    # mod_branch_cov_2d = branch_cov_2d().cuda()
    # mod_branch_cov_2d = Net().cuda()
    # mod_branch_pbulk = branch_pbulk().cuda()
    # new_model = nn.DataParallel(trunk(mod_branch_pbulk, mod_branch_cov), device_ids=[0]).cuda()

    # new_model = trunk(branch_outerprod(), Net()).cuda()

    # Define Model
    # mod_branch_pbulk = nn.DataParallel(branch_outerprod(), device_ids=[0])
    # PATH_branch_pbulk = args.bulk_checkpoint
    # mod_branch_pbulk.load_state_dict(torch.load(PATH_branch_pbulk), strict=True)

    # mod_branch_cov = nn.DataParallel(branch_cov(), device_ids=[0])
    # model = nn.DataParallel(trunk(mod_branch_pbulk, mod_branch_cov), device_ids=[0])
    if args.model == 'a':
        # chromafold left arm with outer product and conv2d
        model_name = "branch_pbulk"
        model = branch_pbulk().cuda()
    elif args.model == 'b':
        # chromafold right arm with only conv1d
        model_name = "branch_cov"
        model = branch_cov().cuda()
    elif args.model == 'c':
        # chromafold conv1d all --> conv2d
        model_name = "branch_cov_2d"
        model = branch_cov_2d().cuda()
    elif args.model == 'd':
        # chromafold architecture
        model_name = "chromafold"
        model = trunk(branch_pbulk().cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'e':
        # epiphany architecture (which is similar to chromafold left arm with the outer product replaced by bi-LSTM)
        model_name = "epiphany"
        model = Net(1, 5, int(args.window_size)).cuda()
    elif args.model == 'f':
        # modified chromafold left arm with branch_cov structure replicated in the 1d layers
        model_name = "branch_modified_pbulk"
        model = branch_modified_pbulk().cuda()
    elif args.model == 'g':
        # left arm: epiphany; right arm: cov1d
        model_name = "alex_model"
        model = trunk(Net(1, 5, int(args.window_size)).cuda(), branch_cov().cuda()).cuda()
    elif args.model == 'h':
        # modified epiphany (without .as_strided())
        model_name = "epiphany2"
        model = Net2(1, 5, int(args.window_size)).cuda()
    elif args.model == 'i':
        # conv1d right arm ad outer_prod right arm
        model_name = "epiphany_2"
        model = trunk(branch_outter_prod().cuda(), branch_cov().cuda()).cuda()
    else:
        model_name = "DEFAULT"
        model = Net(1, 5, int(args.window_size)).cuda()
    print(f"BEGINNING TRAINING: {model_name}")
    if args.wandb:
        import wandb
        wandb.init(project=model_name,)


    disc = Disc()#.cuda()
    if args.wandb:
        wandb.watch(model, log='all')
        # # wandb.watch(new_model, log='all')
        # # wandb.watch(mod_branch_cov, log='all')
        # # wandb.watch(mod_branch_pbulk, log='all')
        # wandb.watch(mod_branch_cov_2d, log='all')



    if os.path.exists(LOG_PATH):
        restore_latest(model, LOG_PATH, ext='.pt_' + model_name)
        # restore_latest(new_model, LOG_PATH, ext='.pt_new_model')
        # restore_latest(mod_branch_cov, LOG_PATH, ext='.pt_mod_branch_cov')
        # restore_latest(mod_branch_pbulk, LOG_PATH, ext='.pt_mod_branch_pbulk')
        # restore_latest(mod_branch_cov_2d, LOG_PATH, ext='.pt_mod_branch_cov_2d')
    else:
        os.makedirs(LOG_PATH)

    with open(os.path.join(LOG_PATH, 'setup.txt'), 'a+') as f:
        f.write("\nVersion: " + args.v)
        f.write("\nBatch Size: " + args.b)
        f.write("\nInitial Learning Rate: " + args.lr)
        f.write("\nComments: " + args.m)


    # GM12878 Standard
    #test_chroms = ['chr3', 'chr11', 'chr17']
    # match test chroms with chromafold!!
    test_chroms = ['chr17']
    #train_chroms = ['chr1', 'chr2', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']
    train_chroms = ['chr19', 'chr20', 'chr21', 'chr22']
    # train_chroms = ['chr22']

    train_set = Chip2HiCDataset(seq_length=TRAIN_SEQ_LENGTH, window_size=int(args.window_size), chroms=train_chroms, mode='train')
    test_set = Chip2HiCDataset(seq_length=TEST_SEQ_LENGTH, window_size=int(args.window_size), chroms=test_chroms, mode='test')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    train_log = os.path.join(LOG_PATH, 'train_log.txt')
    test_log = os.path.join(LOG_PATH, 'test_log.txt')

    hidden = None
    log_interval = 50
    parameters = list(model.parameters())
    # for param in list(new_model.parameters()):
    #     param.requires_grad = True
    # for name, param in new_model.named_parameters():
    #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    # parameters = list(new_model.parameters())
    # parameters = list(mod_branch_cov.parameters())
    # parameters = list(mod_branch_pbulk.parameters())
    # parameters = list(mod_branch_cov_2d.parameters())

    optimizer = optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=0.0005)
    disc_optimizer = optim.Adam(disc.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    min_loss = -10

    t0 = time.time()

    y_up_list = []
    y_down_list = []
    labels = []
    for i, (test_data, test_label, co_signal) in enumerate(test_loader):
        test_label = test_label.squeeze()
        y, y_rev = extract_diagonals(test_label)
        y_up_list.append(y)
        y_down_list.append(y_rev)
        labels.append(test_label[100])
        if i > 400:
            break
    #
    if args.wandb:
        im = wandb.Image(generate_image_test(labels, y_up_list, y_down_list, path=LOG_PATH, seq_length=400))
        wandb.log({"Validation Examples": im})

    # fig, ax = plt.subplots()
    # ax.imshow(im, cmap='RdYlBu_r', vmin=0)
    # plt.savefig('2d_array_visualization_V_test.png')
    # print("Plot saved as '2d_array_visualization_V_test.png'")

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
        # new_model.eval()
        # mod_branch_cov.eval()
        # mod_branch_pbulk.eval()
        # mod_branch_cov_2d.eval()

        if epoch % 1 == 0:
            i = 0
            for (test_data, test_label, co_signal) in tqdm(test_loader):
                if i < 400:
                    if np.linalg.norm(test_label) < 1e-8:
                        continue
                    # test_data, test_label, co_signal = torch.Tensor(test_data[0]).cuda(), torch.Tensor(test_label).cuda(), torch.Tensor(co_signal).cuda()
                    # test_data, test_label = torch.Tensor(test_data[0]), torch.Tensor(test_label)
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
        # save(model, os.path.join(LOG_PATH, '%03d.pt_model' % epoch), num_to_keep=1)
        # save(new_model, os.path.join(LOG_PATH, '%03d.pt_new_model' % epoch), num_to_keep=1)
        # save(mod_branch_cov, os.path.join(LOG_PATH, '%03d.pt_mod_branch_cov' % epoch), num_to_keep=1)
        # save(mod_branch_pbulk, os.path.join(LOG_PATH, '%03d.pt_mod_branch_pbulk' % epoch), num_to_keep=1)
        save(model, os.path.join(LOG_PATH, ('%03d.pt_' + model_name) % epoch), num_to_keep=1)

        with open(test_log, 'a+') as f:
            f.write(str(np.mean(test_loss_cpu)) + "\n")

        losses = []
        model.train()
        # new_model.train()
        # mod_branch_cov.train()
        # mod_branch_pbulk.train()
        # mod_branch_cov_2d.train()
        disc.train()
        for batch_idx, (data, label, co_signal) in enumerate(train_loader):
            if (np.linalg.norm(data)) < 1e-8 or data.shape[2]!=int(args.window_size)+20000:
                continue

            hidden = None
            label = torch.tensor(np.squeeze(label), requires_grad=True).cuda()
            # data = data[0].cuda()
            data = data.cuda()
            optimizer.zero_grad()

            # output, hidden = model(data,seq_length=TRAIN_SEQ_LENGTH)
            # output = new_model(data, torch.tensor(co_signal, requires_grad=True).cuda())
            # output = torch.squeeze(output)
            # output_L, output_R = extract_diagonals(output.squeeze())
            # output = torch.concat((output_L, output_R), dim=0)

            # output = mod_branch_cov(data)
            # output = mod_branch_pbulk(data)
            output = model(data)

            label_1d_v_up, label_1d_v_down = extract_diagonals(label)
            label = torch.concat((label_1d_v_up, label_1d_v_down), dim=0)

            # mse_loss = new_model.loss(output, label, seq_length=TRAIN_SEQ_LENGTH)
            # mse_loss = mod_branch_cov.loss(output, label, seq_length=TRAIN_SEQ_LENGTH)
            # mse_loss = mod_branch_pbulk.loss(output, label, seq_length=TRAIN_SEQ_LENGTH)
            mse_loss = model.loss(output, label, seq_length=TRAIN_SEQ_LENGTH)

            loss = (LAMBDA)*mse_loss #+ (1 - LAMBDA)*adv_loss
            # loss = float(0.5)*mse_loss_up + float(0.5)*mse_loss_down
            # initial_params = {name: param.clone() for name, param in new_model.named_parameters()}
            # initial_params = {name: param.clone() for name, param in mod_branch_cov.named_parameters()}
            # initial_params = {name: param.clone() for name, param in mod_branch_pbulk.named_parameters()}
            initial_params = {name: param.clone() for name, param in model.named_parameters()}
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"No gradients for {name}")

            optimizer.step()

            for name, param in model.named_parameters():
                if torch.equal(param, initial_params[name]):
                    print(f"Parameter {name} has NOT been updated.")

            # # Train discriminator
            # disc_optimizer.zero_grad()
            #
            # true_pred = disc(label.view(1,1,label.shape[0], label.shape[1]))
            # fake_pred = disc(output.detach().view(1,1,output.shape[0], output.shape[1]))
            # disc_preds = torch.cat((true_pred, fake_pred), dim=0)
            # #disc_loss = disc.loss(disc_preds, torch.Tensor([1, 0]).view(2, 1)).cuda())
            # disc_loss = disc.loss(disc_preds, torch.Tensor([1, 0]).view(2,1))
            # disc_loss.backward()
            #
            # disc_preds_train.append(torch.sigmoid(true_pred).item())
            # disc_preds_train.append(torch.sigmoid(fake_pred).item())
            # disc_optimizer.step()


            if args.wandb:
                wandb.log({'mse_loss': mse_loss.item()})
                # wandb.log({'adv_loss': adv_loss.item()})
                wandb.log({'L_G': loss.item()})
                # wandb.log({'L_D': disc_loss.item()})

            if batch_idx % log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))

    t1 = time.time()
    print(t1 - t0)

if __name__ == '__main__':
    main()
