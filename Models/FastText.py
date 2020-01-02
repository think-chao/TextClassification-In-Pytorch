"""
    @Time    : 2019/12/22 18:21
    @Author  : Runa
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, config, vocab_size, num_classes):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, config.embed, padding_idx=0)
        self.dropout = config.dropout
        self.num_class = num_classes
        self.fc1 = nn.Linear(config.embed*3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, self.num_classes)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat([out_word, out_bigram, out_trigram], -1)
        
        out = torch.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

