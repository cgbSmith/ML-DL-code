from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from models import SFGCN
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config

# AMGCN
###################

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # 创建解释器
    parse = argparse.ArgumentParser()
    # 给解释器加上参数
    parse.add_argument("-d", "--dataset", help="dataset", type=str, default="citeseer")
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=20)
    # 解释器进行解析并且将参数赋给解析出来的实体，方便调用
    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    # 读取参数文件并且配置参数
    config = Config(config_file)

    cuda = not config.no_cuda and torch.cuda.is_available()

    use_seed = not config.no_seed
    # 使用随机数种子，系统每次生成的随机数相同
    # 不使用随机数种子，系统每次会采用当前时间值作为种子，每次生成的随机数不同
    # 需要注意的是，每次生成随机数都需要先设置一次随机数种子，才能使得随机数相同
    #例如，我设置了随机种子，random.seed(参数),那么接下来我的随机数为1，3，123，3123
    #当我给相同的参数作为种子的时候，得到的随机也为1,3,123,3123

    if use_seed:
        # 用于生成指定随机数
        np.random.seed(config.seed)
        # 生成cuda随机数种子，方便下次实验结果复现
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

    #以acm为例子
    # fadj为特征图矩阵（3025，3025），sadj为结构图矩阵（3025，3025）
    sadj, fadj = load_graph(args.labelrate, config)
    # features为特征向量（3025，1870），labels为标签向量（3025，1）
    features, labels, idx_train, idx_test = load_data(config)

    model = SFGCN(nfeat=config.fdim,
                  nhid1=config.nhid1,
                  nhid2=config.nhid2,
                  nclass=config.class_num,
                  n=config.n,
                  dropout=config.dropout)
    if cuda:
        model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


    def train(model, epochs):
        model.train()
        optimizer.zero_grad()
        output, att, emb1, com1, com2, emb2, emb = model(features, sadj, fadj)
        loss_class = F.nll_loss(output[idx_train], labels[idx_train])
        loss_dep = (loss_dependence(emb1, com1, config.n) + loss_dependence(emb2, com2, config.n)) / 2
        loss_com = common_loss(com1, com2)
        loss = loss_class + config.beta * loss_dep + config.theta * loss_com
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, macro_f1, emb_test = main_test(model)
        print('e:{}'.format(epochs),
              'ltr: {:.4f}'.format(loss.item()),
              'atr: {:.4f}'.format(acc.item()),
              'ate: {:.4f}'.format(acc_test.item()),
              'f1te:{:.4f}'.format(macro_f1.item()))
        return loss.item(), acc_test.item(), macro_f1.item(), emb_test


    def main_test(model):
        model.eval()
        output, att, emb1, com1, com2, emb2, emb = model(features, sadj, fadj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb


    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(config.epochs):
        loss, acc_test, macro_f1, emb = train(model, epoch)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))
