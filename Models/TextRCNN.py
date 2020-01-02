"""
    @Time    : 2019/12/30 0:27
    @Author  : Runa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, config, vocab_size, classes, num_layers):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, config.embed, padding_idx=0)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2, classes)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(x)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out