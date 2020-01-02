"""
    @Time    : 2019/12/22 15:17
    @Author  : Runa
"""

import time
import torch
import numpy as np
from Train_and_Eval import train, init_network
from importlib import import_module
from utils.Config import Config
from utils.Prepare_Data import MyData
import argparse
import random
import os

parser = argparse.ArgumentParser(description='Chinese Text Classification Parameters')
parser.add_argument('--model', default='TextCNN', type=str, required=True, help='Choose model: TextCNN, FastText, TextRCNN, TextRNN')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for cut-by-word, False for cut-by-char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'
    embedding = 'random' if args.embedding != 'pre_trained' else 'embedding_SougouNews.npz'
    model_name = args.model

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    config = Config(model_name, embedding)
    train_dataset = os.path.join(dataset, 'ptb.train.txt')
    test_dataset = os.path.join(dataset, 'ptb.test.txt')
    dev_dataset = os.path.join(dataset, 'ptb.valid.txt')

    train_data = MyData(train_dataset)
    test_data = MyData(test_dataset)
    dev_data = MyData(dev_dataset)

    vocab_len = len(train_data.get_vocab())
    labels_len = len(train_data.get_labels())

    x = import_module('Models.'+model_name)
    model = x.Model(config, vocab_len, labels_len, 2).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters())
    train(config, model, train_data, test_data, dev_data)