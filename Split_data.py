"""
    @Time    : 2019/12/25 21:54
    @Author  : Runa
"""

import pandas as pd
import shutil
import os
from sklearn.model_selection import train_test_split

def split_data(data, name):
    data = pd.read_csv(data, sep='\t', names=['label', 'text'])
    train, test = train_test_split(data, test_size=0.2)
    test, dev = train_test_split(test, test_size=0.5)
    os.makedirs(name)
    train.to_csv(os.path.join(name, name+'_train.txt'), sep='\t', index=None, header=None)
    test.to_csv(os.path.join(name, name + '_test.txt'), sep='\t', index=None, header=None)
    dev.to_csv(os.path.join(name, name + '_dev.txt'), sep='\t', index=None, header=None)
    
if __name__ == '__main__':
    split_data('toutiao-text-classfication-dataset/news_data.txt', 'news')
