"""
    @Time    : 2019/12/20 0:13
    @Author  : Runa
"""
from collections import defaultdict

import jieba
import json
import torch
import os
import pickle as pkl
from tqdm import tqdm
from .Config import Config
from torch.utils.data import Dataset, DataLoader

UNK, PAD = '<UNK>', '<PAD>'

class MyData(Dataset):
    def __init__(self, data, pad_size=32):
        self.data_path = data
        self.vocab_path = os.path.join(os.path.dirname(data), 'vocab.pkl')
        self.label_path = os.path.join(os.path.dirname(data), 'label.pkl')
        self.pad_size = pad_size
        self.max_size = 10000
        self.min_freq = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.build_dataset(self.data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = torch.tensor(self.data[index]['text'], dtype=torch.long).to(self.device)
        len = torch.tensor(self.data[index]['len'], dtype=torch.long).to(self.device)
        label = torch.tensor(self.data[index]['label'], dtype=torch.long).to(self.device)
        return text, len, label

    def get_labels(self):
        return self.labels

    def get_vocab(self):
        if os.path.exists(self.vocab_path):
            vocab = pkl.load(open(self.vocab_path, 'rb'))
        else:
            if 'train' in self.data_path:
                vocab = self.build_vocab()
                pkl.dump(vocab, open(self.vocab_path, 'wb'))
                print(f"Build Vocab Size:{len(vocab)}")
        return vocab

    def build_vocab(self):
        vocab_dic = {}
        with open(self.data_path, 'r', encoding='utf8') as file:
            for line in tqdm(file):
                l = line.strip()
                if not l:
                    continue
                content = l.split('\t')[1]
                for word in jieba.cut(content):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= self.min_freq], key=lambda x: x[1], reverse=True)[:self.max_size]
        vocab_dic = {word_count[0]: idx+2 for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: 1, PAD: 0})

        return vocab_dic

    def build_dataset(self, path):
        vocab = self.get_vocab()

        def load_dataset(path):
            data = defaultdict(dict)
            labels = dict()
            with open(path, 'r', encoding='utf8') as file:
                for line in tqdm(file.readlines()):
                    l = line.strip()
                    if not l:
                        continue
                    label, *content = l.split('\t')
                    if len(content) == 2:
                        content = ' '.join(content)
                    else:
                        content = content[0]
                    if 'train' not in self.data_path:
                        labels = pkl.load(open(self.label_path, 'rb'))
                    else:
                        if label not in labels:
                            labels[label] = len(labels)
                    label_idx = labels[label]
                    tokens = list(jieba.cut(content))
                    seq_len = len(tokens)
                    tokens = [vocab.get(tok, vocab.get(UNK)) for tok in tokens]
                    if seq_len < self.pad_size:
                        tokens.extend([vocab.get(PAD)] * (self.pad_size - seq_len))
                    else:
                        tokens = tokens[:self.pad_size]
                        seq_len = self.pad_size
                    id = len(data)
                    data[id]['text'] = tokens
                    data[id]['len'] = seq_len
                    data[id]['label'] = label_idx
            print('Building {} Done !'.format(os.path.split(self.data_path)[1]))
            return data, labels

        self.data, self.labels = load_dataset(path)

        if 'train' in self.data_path:
            pkl.dump(self.labels, open(self.label_path, 'wb'))

