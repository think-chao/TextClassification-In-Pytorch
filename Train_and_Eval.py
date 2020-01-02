"""
    @Time    : 2019/12/23 21:56
    @Author  : Runa
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

import time
from utils.Utils import get_time_different

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train, test, dev):
    start_time = time.time()
    model.train()
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    total_iter = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    labels = train.get_labels()

    train_iter = DataLoader(dataset=train, batch_size=config.batch_size, shuffle=True)
    test_iter = DataLoader(dataset=test, batch_size=config.batch_size, shuffle=True)
    dev_iter = DataLoader(dataset=dev, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.num_epochs):
        print('Epoch:{:0>2}/{:0>2}'.format(epoch+1, config.num_epochs))
        for i, (text, _, label) in enumerate(train_iter):
            outputs = model(text)
            model.zero_grad()
            loss = F.cross_entropy(outputs, label)
            loss.backward()
            optimizer.step()
            if total_iter % 100 == 0 and total_iter > 0:
                true = label.data.cpu()
                pred = torch.max(outputs.data, 1)[1].cpu()
                current_acc = accuracy_score(true, pred)
                dev_acc, dev_loss = evaluate(model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_iter
                else:
                    improve = ''
                time_dif = get_time_different(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_iter, loss.item(), current_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_iter += 1
            if total_iter - last_improve > config.require_improvement:
                print('No optimization for a long time, auto-stopping...')
                flag = True
                break
        if flag:
            break

    test_result(config, model, test_iter, labels=labels)


def test_result(config, model, test_iter, labels):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_recall, test_precision, test_f1, test_loss, test_report = evaluate(config, model, test_iter, test=True, labels=labels)
    msg = 'Test Acc:{:>4.2%}, Test Recall:{:>4.2%}, Test Precision:{:>4.2%}, Test f1:{:>4.2%}, Test Loss:{:>5.2}'
    print(msg.format(test_acc, test_recall, test_precision, test_f1, test_loss))
    print('------Report------')
    print(test_report)
    time_dif = get_time_different(start_time)
    print('Time usage:{}'.format(time_dif))


def evaluate(model, data_iter, test=False, labels=None):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, (text, _, label) in enumerate(data_iter):
            outputs = model(text)
            loss = F.cross_entropy(outputs, label)
            loss_total += loss
            labels = label.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = accuracy_score(labels_all, predict_all)

    if test:
        recall = recall_score(labels_all, predict_all, average='macro')
        precision = precision_score(labels_all, predict_all, average='macro')
        f1 = f1_score(labels_all, predict_all, average='macro')
        report = classification_report(labels_all, predict_all, target_names=labels)
        return acc, recall, precision, f1, loss_total/len(data_iter), report

    return acc, loss_total / len(data_iter)
