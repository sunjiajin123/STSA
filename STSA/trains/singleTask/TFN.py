import os
import time
import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')

class TFN():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam(list(model.parameters()), lr=self.args.learning_rate)                                     #optimizer
        # initilize results
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True:                                                                                                      #早停法，一直循环
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()                                                                                                #进入训练模式
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:                                                                                    #开始按批迭代数据
                    vision = batch_data['vision'].to(self.args.device)                                                   #将数据放到GPU上
                    audio = batch_data['audio'].to(self.args.device)                                                     #
                    text = batch_data['text'].to(self.args.device)                                                       #
                    labels = batch_data['labels']['M'].to(self.args.device)                                              #
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()                                                                                #梯度清零
                    # forward
                    outputs = model(text, audio, vision)['M']                                                            #output
                    # compute loss
                    loss = self.criterion(outputs, labels)                                                               #loss
                    # backward
                    loss.backward()                                                                                      #反向传播
                    # update
                    optimizer.step()                                                                                     #更新模型参数
                    # store results
                    train_loss += loss.item()                                                                            #累加每一批次的loss
                    y_pred.append(outputs.cpu())                                                                         #将每一批次的真实标签和预测标签放到列表中，用于计算训练集上的准确率
                    y_true.append(labels.cpu())
                    
            train_loss = train_loss / len(dataloader['train'])                                                           #计算平均loss
            pred, true = torch.cat(y_pred), torch.cat(y_true)                                                            #把每个批度拼起来
            train_results = self.metrics(pred, true)                                                                     #计算这一轮训练后模型的性能
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f %s" % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, train_loss, dict_to_str(train_results)))        #将该轮训练后模型性能注入日志
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)     #判断该轮有没有使性能提升
            # save best model
            if isBetter:                                                                                                 #如果提升了，更新best_valid, best_epoch，并保存模型
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:                                                              #多轮训练但是没有提示验证集性能，停止循环
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()                                                                                                        #与model.train()相对应
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    outputs = model(text, audio, vision)['M']
                    loss = self.criterion(outputs, labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(eval_results)))
        return eval_results