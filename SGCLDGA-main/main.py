import numpy as np
import torch
import pickle
from model import SGCLDGA
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor, mm_auc
from parser import args
from tqdm import tqdm
import torch.utils.data as data
from utils import TrnData



device = 'cuda:' + args.cuda
# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_gene = args.batch
inter_batch = args.inter_batch
epoch_no = args.epoch
lambda_1 = args.lambda1

lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q




def train_folds(fold_id):
    print("lambda_1:",lambda_1,"lambda_2:",lambda_2)
    # load data
    f = open('data/train_mat_fold' + str(fold_id), 'rb')
    train = pickle.load(f)
    train = train.astype(np.float64)
    print(type(train))
    train_csr = (train != 0).astype(np.float32)
    print(type(train_csr))
    f = open('data/test_mat_fold' + str(fold_id), 'rb')
    test = pickle.load(f)
    test = test.astype(np.float64)
    print('Data loaded.')
    print(type(train))
    print('gene_num:', train.shape[0], 'drug_num:', train.shape[1], 'lambda_1:', lambda_1, 'lambda_2:', lambda_2,
          'temp:',
          temp, 'q:', svd_q)

    train_mat = train.todense().A

    train_labels = [[] for i in range(train.shape[0])]
    for i in range(len(train.data)):
        row = train.row[i]
        col = train.col[i]
        train_labels[row].append(col)
    print('Test data processed.')

    epoch_gene = min(train.shape[0], 30000)

    # normalizing the adj matrix
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()

    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)

    # construct data loader
    train = train.tocoo()
    matrix = train.todense().A
    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce().cuda(torch.device(device))
    print('Adj matrix normalized.')

    # perform svd reconstruction
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))
    print('Performing SVD...')
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))
    del s
    print('SVD done.')

    # process test set
    test_labels = [[] for i in range(test.shape[0])]
    for i in range(len(test.data)):
        row = test.row[i]
        col = test.col[i]
        test_labels[row].append(col)
    print('Test data processed.')

    loss_list = []
    loss_r_list = []
    loss_s_list = []

    model = SGCLDGA(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm,
                     l,
                     temp, lambda_1, lambda_2, dropout, batch_gene, device)
    model.cuda(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=lr)
    for epoch in range(epoch_no):
        epoch_loss = 0
        epoch_loss_r = 0
        epoch_loss_s = 0
        train_loader.dataset.neg_sampling()
        for i, batch in enumerate(tqdm(train_loader)):
            geneids, pos, neg = batch
            geneids = geneids.long().cuda(torch.device(device))
            pos = pos.long().cuda(torch.device(device))
            neg = neg.long().cuda(torch.device(device))
            iids = torch.concat([pos, neg], dim=0)

            # feed
            optimizer.zero_grad()
            loss, loss_r, loss_s = model(geneids, iids, pos, neg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item()
            epoch_loss_r += loss_r.cpu().item()
            epoch_loss_s += loss_s.cpu().item()

        batch_no = len(train_loader)
        epoch_loss = epoch_loss / batch_no
        epoch_loss_r = epoch_loss_r / batch_no
        epoch_loss_s = epoch_loss_s / batch_no
        loss_list.append(epoch_loss)
        loss_r_list.append(epoch_loss_r)
        loss_s_list.append(epoch_loss_s)
        print('Epoch:', epoch, 'Loss:', epoch_loss, 'Loss_r:', epoch_loss_r, 'Loss_s:', epoch_loss_s)

for i in range(5):
    train_folds(i)





