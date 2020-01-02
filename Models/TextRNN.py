"""
    @Time    : 2019/12/29 15:29
    @Author  : Runa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, config, vocab_size, num_class, num_layers):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, config.embed, padding_idx=0)

        self.lstm = nn.LSTM(config.embed, config.hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, num_class)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out