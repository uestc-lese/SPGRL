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
'''
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

data = Dataset(root='/tmp/', name='citeseer', setting='prognn')
adj1, features1, labels1 = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
print('idx_train',idx_train)
print('idx_val',len(idx_val))
print('idx_test',len(idx_test))
adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
adj1 = normalize(adj1+sp.eye(adj1.shape[0]))
adj1 = adj1.todense()
adj1 = torch.Tensor(adj1)
# Load nettack attacked data
perturbed_data = PrePtbDataset(root='/tmp/', name='citeseer',
                                        attack_method='nettack',
                                        ptb_rate=1.0) # here ptb_rate means number of perturbation per nodes
perturbed_adj = perturbed_data.adj
idx_test = perturbed_data.target_nodes
print('adj,setting:gcn',len(adj1))
#print(perturbed_adj)
perturbed_adj = perturbed_adj + perturbed_adj.T.multiply(perturbed_adj.T > perturbed_adj) - perturbed_adj.multiply(perturbed_adj.T > perturbed_adj)
perturbed_adj = normalize(perturbed_adj+sp.eye(perturbed_adj.shape[0]))
perturbed_adj = perturbed_adj.todense()
perturbed_adj = torch.Tensor(perturbed_adj)
print('perturbed_adj,setting:nettack',len(perturbed_adj))
'''
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
    #sum_t = 0.0
    acc_max2 = 0
    epoch_max2 = 0
    f1_max2 = 0

    acc_max3 = 0
    epoch_max3 = 0
    f1_max3 = 0

    i1=0
    i2=0
    i3=0







    features, train_index, test_index, labels, sadj, fadj, dsadj, dfadj = load_data(str(args.dataset), args.labelrate, config.k)
    #print('sadj', sadj[1])
    #print('features',features)
    #print('labels',labels)
    #print('sadj:', sadj)
    #print(test_index)
    MAX_EVALS = 100
    T1 = time.time()


    for i in range(MAX_EVALS):
        m = 0
        for m in range(3):
            model = GClip(config.feature_dimension, config.nhid1, config.nhid2, config.num_classes, config.dropout)
                #model = GClip(1870, 768, 512, 3, 0.5)
                ## GClip(feature_dimension, hidden_layer_1, hidden_layer2, num_classes, drop_out_rate)
            
            #features = gaussian(features,
            #                    mean=0.0,
            #                    std= 1.0)
            
            features = torch.Tensor(features)
            #print(sadj)
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
                #optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate, weight_decay= weight_decay)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
            #perturbed_adj = perturbed_adj.to(device)




            #model.train()

            for epoch in range(200):
                model.train()
                optimizer.zero_grad()
                #out, A_pred1, A_pred2, emb1, emb2, logit_scale = model(features, sadj, fadj)
                out, A_pred1, A_pred2, emb1, emb2, logit_scale,_ ,_ ,_,_ = model(features, sadj, fadj)

                acc_train = accuracy(out[train_index], labels[train_index])
                class_loss = F.nll_loss(out[train_index], labels[train_index])
                re_loss = 0.5 * (F.binary_cross_entropy(A_pred2.view(-1), dsadj.view(-1)) + F.binary_cross_entropy(A_pred1.view(-1), dfadj.view(-1)))
                #re_loss = F.binary_cross_entropy(A_pred2.view(-1), dsadj.view(-1))
                #re_loss = torch.nn.BCELoss(A_pred.view(-1), dfadj.view(-1))
                #print('logit_scale',(nn.Parameter(torch.ones([]) * np.log(1 / 0.07))).exp())
                logits1 = logit_scale * emb1 @ emb2.t()
                logits2 = logit_scale * emb2 @ emb1.t()
                ground_truth = torch.arange(len(logits1)).long().to(device)
                cross_loss = 0.5 * (cross_loss1(logits1, ground_truth) + cross_loss2(logits2, ground_truth))
                    #loss = alpha *(class_loss +  re_loss) + beta * cross_loss
                loss = class_loss + config.alpha* re_loss + config.beta * cross_loss
                #loss = class_loss + config.alpha* re_loss
                    ###loss参数loss = 0.2 * re_loss + 10 * class_loss + 0.1 * cross_loss
                    #loss = 0.2 * re_loss + 10 * class_loss + 0.1 * cross_loss
                loss.backward()
                optimizer.step()
                    #scheduler.step()

                model.eval()

                out, A_pred1, A_pred2, _, _, _, _, _, _, _ = model(features, sadj, fadj)
                acc = accuracy(out[test_index], labels[test_index])
                    #print(out[test_index],labels[test_index])
                preds1 = out[test_index].max(1)[1].type_as(labels)
                    #print(preds1)
                    #correct1 = preds.eq(labels).double()
                    #print(correct1)
                    #correct1 = correct.sum()
                label_1 = labels[test_index].cpu().numpy()
                out_1 = preds1.cpu().numpy()
                acc1 = accuracy_score(label_1, out_1)
                macro_f1 = f1_score(label_1, out_1, average='macro')
                    #print('true_f1:{:.4f}'.format(f1))
                    #print('true_acc:{:.4f}'.format(acc1))
                    #if macro_f1 > f1_max1:
                        #f1_max1=macro_f1
                if acc > acc_max1:
                    acc_max1 = acc
                    epoch_max1 = epoch
                    f1_max1 = macro_f1
                    i1=1
                if (acc < acc_max1)&(acc > acc_max2) &(i != i1):
                    acc_max2 = acc
                    epoch_max2 = epoch
                    f1_max2 = macro_f1
                    i2=i
                if (acc < acc_max1) & (acc < acc_max2) & (acc > acc_max3) & (i != i1) & (i != i2):
                    acc_max3 = acc
                    epoch_max3 = epoch
                    f1_max3 = macro_f1
                    i3=i
                if(epoch % 50 == 0):
                    print(epoch,
                            'Accuracy_train: {:.4f}'.format(acc_train),
                            'Accuracy_test: {:.4f}'.format(acc),
                            're_loss: {:.4f}'.format(re_loss),
                            'Loss_class: {:.4f}'.format(class_loss),
                            'Loss_cross: {:.4f}'.format(cross_loss),
                            'true_f1:{:.4f}'.format(macro_f1),
                                )
            #time_end = time.time()
            #sum_t = (time_end - time_start) + sum_t
    T2 = time.time()
    print(
        'Epoch: {}'.format(epoch_max1),
        'Accuracy_Max: {:.4f}'.format(acc_max1),
        'f1_max:{:.4f}'.format(f1_max1),
        #'time cost', sum_t/config.epoch, 's',
        'Epoch: {}'.format(epoch_max2),
        'Accuracy_Max: {:.4f}'.format(acc_max2),
        'f1_max:{:.4f}'.format(f1_max2),
        #'time cost', sum_t/config.epoch, 's',
        'Epoch: {}'.format(epoch_max3),
        'Accuracy_Max: {:.4f}'.format(acc_max3),
        'f1_max:{:.4f}'.format(f1_max3),
        #'time cost', sum_t/config.epoch, 's',
        )
