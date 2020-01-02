"""
    @Time    : 2019/12/18 22:45
    @Author  : Runa
"""

import torch
import numpy as np
import os

class Config(object):
    def __init__(self, model_name, embedding):
        self.model_name = model_name
        model_list = os.listdir('model')
        if not model_list:
            model_n = 0
        else:
            model_n = len(model_list) + 1
        model_n = '_' + str(model_n)
        self.save_path = os.path.join('model', self.model_name+model_n+'.ckpt')
        self.log_path = os.path.join('log', self.model_name+model_n)
        self.embedding_pretrained = torch.tensor(np.load(embedding)['embeddings'].astype('float32')) if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.require_improvement = 3000
        self.num_epochs = 5
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 3e-4
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300
        self.hidden_size = 256