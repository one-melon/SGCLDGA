import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F
import numpy as np

class SGCLDGA(nn.Module):
    def __dnit__(self, n_g, n_d, d, g_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout,
                 batch_gser, device):
        super(SGCLDGA, self).__dnit__()

        self.E_g_0 = nn.Parameter(nn.init.xavier_gniform_(torch.empty(n_g,d)))
        self.E_d_0 = nn.Parameter(nn.init.xavier_gniform_(torch.empty(n_d,d)))
        self.mlp1 = nn.Sequential(nn.Linear(639, 1280),
                                  nn.ReLU(),
                                  nn.Linear(1280, 1280),  
                                  nn.ReLU(),
                                  nn.Linear(1280, d))

        self.mlp2 = nn.Sequential(nn.Linear(120, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 256),  
                                  nn.ReLU(),
                                  nn.Linear(256, d))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_g_list = [None] * (l + 1)
        self.E_d_list = [None] * (l + 1)
        self.E_g_list[0] = self.E_g_0
        self.E_d_list[0] = self.E_d_0
        self.Z_g_list = [None] * (l + 1)
        self.Z_d_list = [None] * (l + 1)
        self.G_g_list = [None] * (l + 1)
        self.G_d_list = [None] * (l + 1)
        self.G_g_list[0] = self.E_g_0
        self.G_d_list[0] = self.E_d_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_gser = batch_gser

        self.E_g = None
        self.E_d = None

        self.g_mul_s = g_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        if test == True:  # testing phase
            preds = self.E_g[uids] @ self.E_d.T
            return  preds, self.E_g, self.E_d
        else:  # training phase
            for layer in range(1, self.l + 1):
                # GNN propagation
                self.Z_g_list[layer] = (
                    torch.spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_d_list[layer - 1]))
                self.Z_d_list[layer] = (
                    torch.spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_g_list[layer - 1]))

                # svd_adj propagation
                vt_ei = self.vt @ self.E_d_list[layer - 1]
                self.G_g_list[layer] = (self.g_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_g_list[layer - 1]
                self.G_d_list[layer] = (self.v_mul_s @ ut_eu)

                # aggregate
                self.E_g_list[layer] = self.Z_g_list[layer]
                self.E_d_list[layer] = self.Z_d_list[layer]

            self.G_g = sum(self.G_g_list)
            self.G_d = sum(self.G_d_list)

            # aggregate across layers
            self.E_g = sum(self.E_g_list)
            self.E_d = sum(self.E_d_list)

            # cl loss
            G_g_norm = self.G_g
            E_g_norm = self.E_g
            G_d_norm = self.G_d
            E_d_norm = self.E_d
            neg_score = torch.log(torch.exp(G_g_norm[uids] @ E_g_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_d_norm[iids] @ E_d_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_g_norm[uids] * E_g_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (
                torch.clamp((G_d_norm[iids] * E_d_norm[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss
            g_emb = self.E_g[uids]
            pos_emb = self.E_d[pos]
            neg_emb = self.E_d[neg]
            pos_scores = (g_emb * pos_emb).sum(-1)
            neg_scores = (g_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            # print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s



