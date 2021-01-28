# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import path_config as path

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, use_type):
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size, vocab):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                try:
                    content, label = lin.split('\t')
                except ValueError:
                    content = lin
                    label = '-1'
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, float(label), seq_len))
                # print(contents)
        return contents  # [([...], 0), ([...], 1), ...]

    if use_type == 'train':
        train = load_dataset(config.train_path, config.pad_size, vocab)
        val = load_dataset(config.val_path, config.pad_size, vocab)
        test = load_dataset(config.test_path, config.pad_size, vocab)
        return vocab, train, val, test
    else:
        test = load_dataset(config.test_path, config.pad_size, vocab)
        return vocab, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) / batch_size
        self.residue = False  # 记录batch数量是否为整数
        if self.n_batches == 0 or (len(batches) % self.n_batches != 0):
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def generate_embed(emb_dim):
    """根据训练集生成词向量矩阵"""
    embeddings = np.random.rand(len(word_to_id) + 1, emb_dim)
    f = open(path.pretrain_path, 'rb')    # 预先训练好的
    index_dict = pkl.load(f)        # 索引字典，{单词: 索引数字}
    word_vectors = pkl.load(f)      # 词向量, {单词: 词向量(wv_size维长的数组)}
    for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
        if w in word_to_id:
            idx = word_to_id[w]
            embeddings[idx, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
    # print(embeddings)
    np.savez_compressed(path.embed_path, embeddings=embeddings)


if __name__ == "__main__":
    '''提取预训练词向量'''

    if os.path.exists(path.vocab_path):
        word_to_id = pkl.load(open(path.vocab_path, 'rb'))
    else:
        tokenizer = lambda x: x.split(' ')  # 以word为单位构建词表
        word_to_id = build_vocab(path.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(path.vocab_path, 'wb'))

    generate_embed(emb_dim=100)