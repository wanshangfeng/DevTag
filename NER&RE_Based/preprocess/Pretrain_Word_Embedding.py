import os
import json
from glove import Glove
from glove import Corpus
from IPython.core.debugger import set_trace
import re
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
config = yaml.load(open("build_data_config.yaml", "r"), Loader=yaml.FullLoader)


# Data


data_home = config["data_out_dir"]

experiment_name = config["exp_name"]
emb_dim = 300

data_dir = os.path.join(data_home, experiment_name)
train_data_path = os.path.join(data_dir, "train_data.json")
valid_data_path = os.path.join(data_dir, "valid_data.json")
test_data_dir = os.path.join(data_dir, "test_data")
test_data_path_dict = {}
for path, folds, files in os.walk(test_data_dir):
    for file_name in files:
        file_path = os.path.join(path, file_name)
        file_name = re.match("(.*?)\.json", file_name).group(1)
        test_data_path_dict[file_name] = file_path

train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
test_data_dict = {}
for file_name, path in test_data_path_dict.items():
    test_data_dict[file_name] = json.load(open(path, "r", encoding="utf-8"))

all_data = train_data + valid_data
for data in list(test_data_dict.values()):
    all_data.extend(data)

corpus = [sample["text"].split(" ") for sample in all_data]
len(corpus)


# Glove


def train_glove_emb(corpus, window=10, emb_dim=100, learning_rate=0.05, epochs=10, thr_workers=0):
    corpus_model = Corpus()
    corpus_model.fit(corpus, window=window)
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)

    glove = Glove(no_components=emb_dim, learning_rate=learning_rate)
    glove.fit(corpus_model.matrix,
              epochs=epochs,
              no_threads=thr_workers,
              verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    return glove


# glove
golve = train_glove_emb(corpus, emb_dim=emb_dim)

# save
save_path = os.path.join("../pretrained_word_emb", "glove_{}_{}.emb".format(emb_dim, experiment_name))
golve.save(save_path)

golve.most_similar('university', number=10)

golve.word_vectors.shape

# Quick Start

# # get similar words
# golve.most_similar('Massachusetts', number = 10)

# # emb matrix shape
# golve.word_vectors.shape

# # get id
# golve.dictionary['Virginia']

# # # 指定词条词向量
# # glove.word_vectors[glove.dictionary['university']]

# # save
# save_path = os.path.join(data_home, "pretrained_word_embeddings", "glove_100.emb")
# glove.save(save_path)

# # load
# glove = Glove()
# glove = glove.load(save_path)
