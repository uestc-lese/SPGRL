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
import nni
from nni.utils import merge_parameter
import logging
logger = logging.getLogger('mnist_AutoML')



def main(args):
    config_file = "./config/" + str(args['labelrate']) + str(args['dataset']) + ".ini"
    config = Config(config_file)
    #params = nni.get_next_parameter()

    print(args['dropout'])
    #features, train_index, test_index, labels, sadj, fadj, dsadj, dfadj = load_data(str(args.dataset), args.labelrate, config.k)
    features, train_index, test_index, labels, sadj, fadj, dsadj, dfadj = load_data(str(args['dataset']), args['labelrate'], args['k'])
    n_nodes = len(sadj)


    model = GClip(config.feature_dimension, config.nhid1, config.nhid2, config.num_classes, args['dropout'])
                #model = GClip(1870, 768, 512, 3, 0.5)
                ## GClip(feature_dimension, hidden_layer_1, hidden_layer2, num_classes, drop_out_rate)


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
            #KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    #optimizer = torch.optim.Adam(model.parameters(), lr= params['learning_rate'], weight_decay= params['weight_decay'])
    optimizer = torch.optim.Adam(model.parameters(), lr= config.lr, weight_decay=config.weight_decay)
            #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)


    model.train()
            #for epoch in range(200):

            #print(i,hyperparameters)

            

    for epoch in range(config.epoch):
        optimizer.zero_grad()
        out, A_pred1, A_pred2, emb1, emb2, logit_scale, smu, slogvar ,fmu , flogvar= model(features, sadj, fadj)
        SKLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * slogvar - smu.pow(2) - slogvar.exp().pow(2), 1))
        FKLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * flogvar - fmu.pow(2) - flogvar.exp().pow(2), 1))
                #print(SKLD,FKLD)
        acc_train = accuracy(out[train_index], labels[train_index])
        class_loss = F.nll_loss(out[train_index], labels[train_index])
        re_loss = 0.5 * (F.binary_cross_entropy(A_pred2.view(-1), dsadj.view(-1)) + args['gamma']*SKLD +
                                 F.binary_cross_entropy(A_pred1.view(-1), dfadj.view(-1)) + args['delta']*FKLD)
        #re_loss = 0.5 * (F.binary_cross_entropy(A_pred1.view(-1), dsadj.view(-1))  +
        #                 F.binary_cross_entropy(A_pred2.view(-1), dfadj.view(-1))) 
                    # re_loss = torch.nn.BCELoss(A_pred.view(-1), dfadj.view(-1))
        logits1 = logit_scale * emb1 @ emb2.t()
        logits2 = logit_scale * emb2 @ emb1.t()
        ground_truth = torch.arange(len(logits1)).long().to(device)
        cross_loss = 0.5 * (cross_loss1(logits1, ground_truth) + cross_loss2(logits2, ground_truth))
        #loss = class_loss + params['alpha'] * re_loss + params['beta'] * cross_loss
        loss = class_loss + config.alpha * re_loss + config.beta * cross_loss
                    ###loss参数loss = 0.2 * re_loss + 10 * class_loss + 0.1 * cross_loss
        loss.backward()
        optimizer.step()

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
                #acc = accuracy_score(label_1, out_1)
        macro_f1 = f1_score(label_1, out_1, average='macro')
                    #print('true_f1:{:.4f}'.format(f1))
                    #print('true_acc:{:.4f}'.format(acc1))
        nni.report_intermeidate_result(acc)
        logger.debug('test accuracy %g', test_acc)
        logger.debug('Pipe send intermediate result done.')



    nni.report_final_result(test_acc)
    logger.debug('Final result is %g', test_acc)
    logger.debug('Send final result done.')

def get_params():
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
    parse.add_argument("--weight_decay", help="labeled data for train per class", type = float,default=0.5)
    parse.add_argument('--learning_rate', type=float, default=0.0001)
    parse.add_argument('--hidden_size2', type=int, default=768)
    parse.add_argument('--hidden_size1', type=int, default=512)
    parse.add_argument('--alpha', type=float, default=0.02)
    parse.add_argument('--beta', type=float, default=0.01)
    parse.add_argument('--gamma', type=float, default=1.0)
    parse.add_argument('--delta', type=float, default=1.0)
    parse.add_argument('--dropout', type=float, default=0.5)
    parse.add_argument('--k', type=float, default=13)
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #parse = argparse.ArgumentParser()
        #parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
        #parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
        #args = parse.parse_args()
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
 