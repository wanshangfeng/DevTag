import sys
import logging
import multiprocessing
import smart_open
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models.word2vec import LineSentence, logger
import pickle
import path_config as path


def word2vector(X_train):
    """词向量"""
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    wv_model = Word2Vec(X_train, size=wv_size, window=6, sg=1, min_count=5, workers=multiprocessing.cpu_count(), iter=10)

    gensim_dict = Dictionary()    # 创建词语词典
    gensim_dict.doc2bow(wv_model.wv.vocab.keys(), allow_update=True)

    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: wv_model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec



if __name__ == '__main__':

    X_train = LineSentence(smart_open.open(path.train_path, encoding='utf-8'))

    wv_size = 100  # 词向量维度

    print('Training...')
    index_dict, word_vectors = word2vector(X_train)  # 索引字典、词向量字典

    output = open(path.pretrain_path, 'wb')
    pickle.dump(index_dict, output)
    pickle.dump(word_vectors, output)
    output.close()