from numpy.lib import load
from model import GClip
import torch 
from dataset import load_data
import torch.nn.functional as F
from utils import accuracy
from config import Config
import os
import argparse
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
import random
import numpy as np

import time
import numpy as np
import scipy.sparse as sp
import torch
#from deeprobust.graph.data import Dataset, PrePtbDataset
from noise import gaussian, gaussian_mimic,\
                  superimpose_gaussian, superimpose_gaussian_class,\
                  superimpose_gaussian_random, zero_idx

import torch.nn as nn

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)


    acc_max1 = 0
    epoch_max1 = 0
    f1_max1 = 0
    i1=0



    features, train_index, test_index, labels, sadj, fadj, dsadj, dfadj = load_data(str(args.dataset), args.labelrate, config.k)
    model = GClip(config.feature_dimension, config.nhid1, config.nhid2, config.num_classes, config.dropout)
    #features = gaussian(features, mean=0.0, std=1.0) attacked feature
    #features = torch.Tensor(features)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    train_index = train_index.to(device)
    test_index = test_index.to(device)
    labels = labels.to(device)
    sadj = sadj.to(device)
    fadj = fadj.to(device) 
    dsadj = dsadj.to(device)
    dfadj = dfadj.to(device)
    model = model.to(device)
    cross_loss1 = torch.nn.CrossEntropyLoss().to(device)
    cross_loss2 = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)



    for epoch in range(config.epoch):
        model.train()
        optimizer.zero_grad()
        out, A_pred1, A_pred2, emb1, emb2, logit_scale = model(features, sadj, fadj)

        acc_train = accuracy(out[train_index], labels[train_index])
        class_loss = F.nll_loss(out[train_index], labels[train_index])
        re_loss = 0.5 * (F.binary_cross_entropy(A_pred2.view(-1), dsadj.view(-1)) +
                    F.binary_cross_entropy(A_pred1.view(-1), dfadj.view(-1)))

        logits1 = logit_scale * emb1 @ emb2.t()
        logits2 = logit_scale * emb2 @ emb1.t()
        ground_truth = torch.arange(len(logits1)).long().to(device)
        cross_loss = 0.5 * (cross_loss1(logits1, ground_truth) + cross_loss2(logits2, ground_truth))
        loss = class_loss + config.alpha* re_loss + config.beta * cross_loss

        loss.backward()
        optimizer.step()

        model.eval()

        out, A_pred1, A_pred2, _, _, _ = model(features, sadj, fadj)
        acc = accuracy(out[test_index], labels[test_index])
        preds1 = out[test_index].max(1)[1].type_as(labels)
        label_1 = labels[test_index].cpu().numpy()
        out_1 = preds1.cpu().numpy()
        acc1 = accuracy_score(label_1, out_1)
        macro_f1 = f1_score(label_1, out_1, average='macro')
        if acc > acc_max1: 
            acc_max1 = acc
            epoch_max1 = epoch
            f1_max1 = macro_f1
            i1=1

        if(epoch % 50 == 0):
            print(epoch,
                'Accuracy_train: {:.4f}'.format(acc_train),
                'Accuracy_test: {:.4f}'.format(acc),
                're_loss: {:.4f}'.format(re_loss),
                'Loss_class: {:.4f}'.format(class_loss),
                'Loss_cross: {:.4f}'.format(cross_loss),
                'true_f1:{:.4f}'.format(macro_f1),
                    )
         
    print( 'Epoch: {}'.format(epoch_max1),
        'Accuracy_Max: {:.4f}'.format(acc_max1),
        'f1_max:{:.4f}'.format(f1_max1),
        )