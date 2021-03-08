# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, test
from utils import build_dataset, build_iterator, get_time_dif
from importlib import import_module
import argparse
import path_config as path

parser = argparse.ArgumentParser(description='Model-based')
parser.add_argument('--type', default='train', type=str, help='train or test')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, TextRCNN, TextRNN_Att, DPCNN')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--file', type=str, help='Upload your text file')
args = parser.parse_args()


if __name__ == '__main__':
    embedding = 'embedding_banner.npz'
    if args.embedding == 'random':  # 随机初始化:random
        embedding = 'random'
    model_name = args.model  # TextCNN, TextRNN, TextRCNN, TextRNN_Att, DPCNN
    x = import_module('models.' + model_name)
    config = x.Config(path.dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    if args.type == 'train':
        start_time = time.time()
        print("Loading data...")
        vocab, train_data, dev_data, test_data = build_dataset(config, args.type)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        # train
        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)
        init_network(model)
        print(model.parameters)
        train(config, model, train_iter, dev_iter, test_iter)
        test(config, model, test_iter, args.type)

    if args.type == 'test':
        from data_pre import test_data_pre
        start_time = time.time()
        config.test_path = path.user_test_pre_path  # Preprocessed data
        banners = test_data_pre(args.file, config.test_path)
        vocab, test_data = build_dataset(config, args.type)
        test_iter = build_iterator(test_data, config)

        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)
        init_network(model)
        print(model.parameters)
        test(config, model, test_iter, args.type, banners)