# coding: utf-8

dataset = 'dataset'  # 数据集目录
train_path = dataset + '/data/train.txt'    # 训练集
val_path = dataset + '/data/valid.txt'      # 验证集
test_path = dataset + '/data/test.txt'      # 测试集
class_path = dataset + '/data/class.txt'    # 类别集合
vocab_path = dataset + "/data/vocab.pkl"    # 训练集对应的词表
embed_path = dataset + "/data/embedding_banner"  # 训练集对应的词向量矩阵


pretrain_path = "pretrained/index_wv.pkl"          # 预训练词向量
tfidf_path = "pretrained/tfidf_vec.pkl"   # tfidf向量

user_test_path = 'test'  # 测试数据目录
user_test_pre_path = user_test_path + '/test-pre.txt'   # 预处理后的测试数据
user_test_result_path = user_test_path + '/DevTag.json'     # 测试结果

