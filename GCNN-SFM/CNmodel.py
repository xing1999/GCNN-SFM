import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
import math
import csv
import random
import heapq
import re
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter
import Biodata

class model(nn.Module):
    def __init__(self, label_num, other_feature_dim, K=1, d=2, node_hidden_dim=3, gcn_dim=128, gcn_layer_num=4, cnn_dim=64, cnn_layer_num=3, cnn_kernel_size=8, fc_dim=100, dropout_rate=0.2, pnode_nn=True, fnode_nn=True):
        super(model, self).__init__()
        self.label_num = label_num
        self.pnode_dim = d
        self.pnode_num = 4 ** (2 * K)
        self.fnode_num = 4 ** K
        self.node_hidden_dim = node_hidden_dim
        self.gcn_dim = gcn_dim
        self.gcn_layer_num = gcn_layer_num
        self.cnn_dim = cnn_dim
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim
        self.dropout = dropout_rate
        self.pnode_nn = pnode_nn
        self.fnode_nn = fnode_nn
        self.other_feature_dim = other_feature_dim

        self.pnode_d = nn.Linear(self.pnode_num * self.pnode_dim, self.pnode_num * self.node_hidden_dim)
        self.fnode_d = nn.Linear(self.fnode_num, self.fnode_num * self.node_hidden_dim)
        
        self.gconvs_1 = nn.ModuleList()
        self.gconvs_2 = nn.ModuleList()
        
        if self.pnode_nn:
            pnode_dim_temp = self.node_hidden_dim
        else:
            pnode_dim_temp = self.pnode_dim
        
        if self.fnode_nn:
            fnode_dim_temp = self.node_hidden_dim
        else:
            fnode_dim_temp = 1
        
        for l in range(self.gcn_layer_num):
            if l == 0:
                self.gconvs_1.append(pyg_nn.SAGEConv((fnode_dim_temp, pnode_dim_temp), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, fnode_dim_temp), self.gcn_dim))
            else:                                   
                self.gconvs_1.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
        
        self.lns = nn.ModuleList()
        for l in range(self.gcn_layer_num-1):
            self.lns.append(nn.LayerNorm(self.gcn_dim))

        self.convs = nn.ModuleList()
        for l in range(self.cnn_layer_num):
            if l == 0:
                self.convs.append(nn.Conv1d(in_channels=self.gcn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))
            else:
                self.convs.append(nn.Conv1d(in_channels=self.cnn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))
        
        if self.other_feature_dim:
            self.d1 = nn.Linear((self.pnode_num - (self.cnn_kernel_size - 1) * self.cnn_layer_num) * self.cnn_dim, self.fc_dim)
            self.d2 = nn.Linear(self.fc_dim + self.other_feature_dim, self.fc_dim + self.other_feature_dim)
            self.d3 = nn.Linear(self.fc_dim + self.other_feature_dim, self.label_num)
        else:
            self.d1 = nn.Linear((self.pnode_num - (self.cnn_kernel_size - 1) * self.cnn_layer_num) * self.cnn_dim, self.fc_dim)
            self.d2 = nn.Linear(self.fc_dim, self.label_num)


    def forward(self, data):
        x_f = data.x_src
        x_p = data.x_dst
        edge_index_forward = data.edge_index[:,::2]
        edge_index_backward = data.edge_index[[1, 0], :][:,1::2]

        if self.other_feature_dim:
            other_feature = torch.reshape(data.other_feature, (-1, self.other_feature_dim)) 
        
        # transfer primary nodes
        if self.pnode_nn:
            x_p = torch.reshape(x_p, (-1, self.pnode_num * self.pnode_dim))
            x_p = self.pnode_d(x_p)
            x_p = torch.reshape(x_p, (-1, self.node_hidden_dim))
        else:
            x_p = torch.reshape(x_p, (-1, self.pnode_dim))
        
        # transfer feature nodes
        if self.fnode_nn:
            x_f = torch.reshape(x_f, (-1, self.fnode_num))
            x_f = self.fnode_d(x_f)
            x_f = torch.reshape(x_f, (-1, self.node_hidden_dim))
        else:
            x_f = torch.reshape(x_f, (-1, 1))

        for i in range(self.gcn_layer_num):
            x_p = self.gconvs_1[i]((x_f, x_p), edge_index_forward)
            x_p = F.relu(x_p)
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            x_f = self.gconvs_2[i]((x_p, x_f), edge_index_backward)
            x_f = F.relu(x_f)
            x_f = F.dropout(x_f, p=self.dropout, training=self.training)
            if not i == self.gcn_layer_num - 1:
                x_p = self.lns[i](x_p)
                x_f = self.lns[i](x_f)

        x = torch.reshape(x_p, (-1, self.gcn_dim, self.pnode_num))
        
        for i in range(self.cnn_layer_num):
            x = self.convs[i](x)
            x = F.relu(x)
            if not i == 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.other_feature_dim:
            x = x.flatten(start_dim = 1)
            x = self.d1(x)
            x = F.relu(x)
            x = self.d2(torch.cat([x, other_feature], 1))
            x = F.relu(x)
            x = self.d3(x)
            out = F.softmax(x, dim=1)

        else:
            x = x.flatten(start_dim = 1)
            x = self.d1(x)
            x = F.relu(x)
            x = self.d2(x)
            out = F.softmax(x, dim=1)

        return out

def train(dataset, model, learning_rate=1e-4, batch_size=64, epoch_n=30, random_seed=111, val_split=0.1, weighted_sampling=True, model_name="h-12.pt", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    random.seed(random_seed)
    data_list = list(range(0, len(dataset)))
    test_list = random.sample(data_list, int(len(dataset) * val_split))
    trainset = [dataset[i] for i in data_list if i not in test_list]
    testset = [dataset[i] for i in data_list if i in test_list]
    
    if weighted_sampling:
        label_count = Counter([int(data.y) for data in dataset])
        weights = [100/label_count[int(data.y)] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size,follow_batch=['x_src', 'x_dst'], sampler=sampler)
        print(len(train_loader.dataset))
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, follow_batch=['x_src', 'x_dst'])
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, follow_batch=['x_src', 'x_dst'])
    print(len(test_loader.dataset))

    # build model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train
    old_test_acc = 0
    for epoch in range(epoch_n):
        training_running_loss = 0.0
        train_acc = 0.0

        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            label = batch.y

            # forward + backprop + loss
            pred = model(batch)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()

            # update model params
            optimizer.step()

            training_running_loss += loss.detach().item()
            train_acc += (torch.argmax(pred, 1).flatten() == label).type(torch.float).mean().item()

        # test accuracy
        test_acc = evaluation(test_loader, model, device)
        if test_acc > old_test_acc:
            torch.save(model, model_name)
            old_test_acc = test_acc
        print("Epoch {}| Loss: {:.4f}| Train accuracy: {:.4f}| Validation accuracy: {:.4f}".format(epoch, training_running_loss/(i+1), train_acc/(i+1), test_acc))

    return model

def evaluation(loader, model, device):
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
        correct += pred.eq(label).sum().item()

    total = len(loader.dataset)
    acc = correct / total

    return acc


def test(data1, model_name="DNN-22.pt",val_split=0.6, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    data_list = list(range(0, len(data1)))
    test_list = random.sample(data_list, int(len(data1) * val_split))
    testset = [data1[i] for i in data_list if i in test_list]
    model = torch.load(model_name, map_location=device)
    loader = DataLoader(testset, batch_size=len(data1), shuffle=False, follow_batch=['x_src', 'x_dst'])
    # loader = DataLoader(testset, batch_size=64, shuffle=False, follow_batch=['x_src', 'x_dst'])
    # print(len(loader.dataset))
    model.eval()
    TP, FN, FP, TN = 0, 0, 0, 0
    print(len(loader.dataset))
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
            AUC = Calauc(label, pred)
            # correct += pred.eq(label).sum().item()
            A, B, C, D = eff(label, pred)
            TP += A
            FN += B
            FP += C
            TN += D

        # print(TP, FN, FP, TN)
    SN, SP, ACC, MCC = Judeff(TP, FN, FP, TN)
    #
    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))

    print("SN: {}, SP: {}, ACC: {}, MCC: {}, AUC: {}".format(SN, SP, ACC, MCC, AUC))
    modelname=model_name
    date=[modelname,TP,FN,FP,TN,SN, SP, ACC, MCC, AUC]
    csvfile = open('rundate.csv', 'a', encoding='utf-8',newline='')
    writer = csv.writer(csvfile)
    # writer.writerow(['Model','TP', 'FN', 'FP', 'TN','SN', 'SP','ACC','MCC','AUC'])
    writer.writerow(date)
    csvfile.close()



def eff(labels, preds):

    TP, FN, FP, TN = 0, 0, 0, 0
    for idx,label in enumerate(labels):

        if label == 1:
            if label == preds[idx]:
                TP += 1
            else: FN += 1
        elif label == preds[idx]:
            TN += 1
        else: FP += 1

    return TP, FN, FP, TN


def Judeff(TP, FN, FP, TN):

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    MCC = (TP * TN - FP * FN) / (math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))

    return SN, SP, ACC, MCC

def Calauc(labels, preds):

    labels = labels.clone().detach().cpu().numpy()
    preds = preds.clone().detach().cpu().numpy()

    f = list(zip(preds, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    pos_cnt = np.sum(labels == 1)
    neg_cnt = np.sum(labels == 0)
    AUC = (np.sum(rankList) - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)

    return AUC



