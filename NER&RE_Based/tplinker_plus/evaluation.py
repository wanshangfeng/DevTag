#!/usr/bin/env python
# coding: utf-8
import sys

sys.path.append("..")

import json
import os
from tqdm import tqdm
import re
from IPython.core.debugger import set_trace
from pprint import pprint
import unicodedata
from transformers import BertModel, BasicTokenizer, BertTokenizerFast
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glob
import time
from common.utils import Preprocessor
from .tplinker_plus import (HandshakingTaggingScheme,
                            DataMaker4Bert,
                            DataMaker4BiLSTM,
                            TPLinkerPlusBert,
                            TPLinkerPlusBiLSTM,
                            MetricsCalculator)
import wandb
from . import config
from glove import Glove
import numpy as np

config = config.eval_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_home = config["data_home"]
experiment_name = config["exp_name"]
test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
batch_size = hyper_parameters["batch_size"]
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])
ent2id_path = os.path.join(data_home, experiment_name, config["ent2id"])
save_res_dir = os.path.join(config["save_res_dir"], experiment_name)
word_embedding_path = config["pretrained_word_embedding_path"]
model_state_dir = config["model_state_dict_dir"]
global max_seq_len
max_seq_len = hyper_parameters["max_seq_len"]
save_res = config["save_res"]
score = config["score"]
use_type = "eval"  # "test"
if use_type == "test":
    global save_path

# for reproductivity
torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_last_k_paths(path_list, k):
    path_list = sorted(path_list, key=lambda x: int(re.search("(\d+)", x.split("\\")[-1]).group(1)))
    #     pprint(path_list)
    return path_list[-k:]


def filter_duplicates(rel_list, ent_list):
    rel_memory_set = set()
    filtered_rel_list = []

    for rel in rel_list:
        rel_memory = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                 rel["subj_tok_span"][1],
                                                                 rel["predicate"],
                                                                 rel["obj_tok_span"][0],
                                                                 rel["obj_tok_span"][1])
        if rel_memory not in rel_memory_set:
            filtered_rel_list.append(rel)
            rel_memory_set.add(rel_memory)

    ent_memory_set = set()
    filtered_ent_list = []
    for ent in ent_list:
        ent_memory = "{}\u2E80{}\u2E80{}".format(ent["tok_span"][0],
                                                 ent["tok_span"][1],
                                                 ent["type"])
        if ent_memory not in ent_memory_set:
            filtered_ent_list.append(ent)
            ent_memory_set.add(ent_memory)

    return filtered_rel_list, filtered_ent_list


def predict(data_maker, rel_extractor, test_data, ori_test_data, split_test_data, handshaking_tagger):
    '''
    test_data: if split, it would be samples with subtext
    ori_test_data: the original data has not been split, used to get original text here
    '''
    indexed_test_data = data_maker.get_indexed_data(test_data, max_seq_len, data_type="test")  # fill up to max_seq_len
    test_dataloader = DataLoader(MyDataset(indexed_test_data),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=False,
                                 collate_fn=lambda data_batch: data_maker.generate_batch(data_batch, data_type="test"),
                                 )

    pred_sample_list = []
    for batch_test_data in tqdm(test_dataloader, desc="Predicting"):
        if config["encoder"] == "BERT":
            sample_list, batch_input_ids, \
            batch_attention_mask, batch_token_type_ids, \
            tok2char_span_list, _ = batch_test_data

            batch_input_ids, \
            batch_attention_mask, \
            batch_token_type_ids = (batch_input_ids.to(device),
                                    batch_attention_mask.to(device),
                                    batch_token_type_ids.to(device),
                                    )

        elif config["encoder"] in {"BiLSTM", }:
            sample_list, batch_input_ids, \
            tok2char_span_list, _ = batch_test_data

            batch_input_ids = batch_input_ids.to(device)

        with torch.no_grad():
            if config["encoder"] == "BERT":
                batch_pred_shaking_tag, _ = rel_extractor(batch_input_ids,
                                                          batch_attention_mask,
                                                          batch_token_type_ids,
                                                          )
            elif config["encoder"] in {"BiLSTM", }:
                batch_pred_shaking_tag, _ = rel_extractor(batch_input_ids)

        batch_pred_shaking_tag = (batch_pred_shaking_tag > 0.).long()

        for ind in range(len(sample_list)):
            gold_sample = sample_list[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            tok2char_span = tok2char_span_list[ind]
            pred_shaking_tag = batch_pred_shaking_tag[ind]
            tok_offset, char_offset = 0, 0
            if split_test_data:
                tok_offset, char_offset = gold_sample["tok_offset"], gold_sample["char_offset"]
            rel_list, ent_list = handshaking_tagger.decode_rel(text,
                                                               pred_shaking_tag,
                                                               tok2char_span,
                                                               tok_offset=tok_offset, char_offset=char_offset)
            pred_sample_list.append({
                "text": text,
                "id": text_id,
                "relation_list": rel_list,
                "entity_list": ent_list,
            })

    # merge
    text_id2pred_res = {}
    for sample in pred_sample_list:
        text_id = sample["id"]
        if text_id not in text_id2pred_res:
            text_id2pred_res[text_id] = {
                "rel_list": sample["relation_list"],
                "ent_list": sample["entity_list"],
            }
        else:
            text_id2pred_res[text_id]["rel_list"].extend(sample["relation_list"])
            text_id2pred_res[text_id]["ent_list"].extend(sample["entity_list"])

    text_id2text = {sample["id"]: sample["text"] for sample in ori_test_data}
    merged_pred_sample_list = []
    for text_id, pred_res in text_id2pred_res.items():
        filtered_rel_list, filtered_ent_list = filter_duplicates(pred_res["rel_list"], pred_res["ent_list"])
        merged_pred_sample_list.append({
            "id": text_id,
            "text": text_id2text[text_id],
            "relation_list": filtered_rel_list,
            "entity_list": filtered_ent_list,
        })

    return merged_pred_sample_list


def get_test_prf(handshaking_tagger, metrics, pred_sample_list, gold_test_data, pattern="whole_text"):
    text_id2gold_n_pred = {}  # text id to gold and pred results

    for sample in gold_test_data:
        text_id = sample["id"]
        if pattern == "event_extraction":
            text_id2gold_n_pred[text_id] = {
                "gold_relation_list": sample["relation_list"],
                "gold_entity_list": sample["entity_list"],
                "gold_event_list": sample["event_list"],
            }
        else:
            text_id2gold_n_pred[text_id] = {
                "gold_relation_list": sample["relation_list"],
                "gold_entity_list": sample["entity_list"],
            }

    for sample in pred_sample_list:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id]["pred_relation_list"] = sample["relation_list"]
        text_id2gold_n_pred[text_id]["pred_entity_list"] = sample["entity_list"]

    correct_num, pred_num, gold_num = 0, 0, 0
    ent_correct_num, ent_pred_num, ent_gold_num = 0, 0, 0
    ee_cpg_dict = {
        "trigger_iden_cpg": [0, 0, 0],
        "trigger_class_cpg": [0, 0, 0],
        "arg_iden_cpg": [0, 0, 0],
        "arg_class_cpg": [0, 0, 0],
    }
    ere_cpg_dict = {
        "rel_cpg": [0, 0, 0],
        "ent_cpg": [0, 0, 0],
    }
    for gold_n_pred in text_id2gold_n_pred.values():
        gold_rel_list = gold_n_pred["gold_relation_list"]
        pred_rel_list = gold_n_pred["pred_relation_list"] if "pred_relation_list" in gold_n_pred else []
        gold_ent_list = gold_n_pred["gold_entity_list"]
        pred_ent_list = gold_n_pred["pred_entity_list"] if "pred_entity_list" in gold_n_pred else []

        if pattern == "event_extraction":
            pred_event_list = handshaking_tagger.trans2ee(pred_rel_list, pred_ent_list)  # transform to event list
            gold_event_list = gold_n_pred["gold_event_list"]  # *
            metrics.cal_event_cpg(pred_event_list, gold_event_list, ee_cpg_dict)
        else:
            metrics.cal_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ere_cpg_dict, pattern)

    if pattern == "event_extraction":
        trigger_iden_prf = metrics.get_prf_scores(ee_cpg_dict["trigger_iden_cpg"][0],
                                                  ee_cpg_dict["trigger_iden_cpg"][1],
                                                  ee_cpg_dict["trigger_iden_cpg"][2])
        trigger_class_prf = metrics.get_prf_scores(ee_cpg_dict["trigger_class_cpg"][0],
                                                   ee_cpg_dict["trigger_class_cpg"][1],
                                                   ee_cpg_dict["trigger_class_cpg"][2])
        arg_iden_prf = metrics.get_prf_scores(ee_cpg_dict["arg_iden_cpg"][0], ee_cpg_dict["arg_iden_cpg"][1],
                                              ee_cpg_dict["arg_iden_cpg"][2])
        arg_class_prf = metrics.get_prf_scores(ee_cpg_dict["arg_class_cpg"][0], ee_cpg_dict["arg_class_cpg"][1],
                                               ee_cpg_dict["arg_class_cpg"][2])
        prf_dict = {
            "trigger_iden_prf": trigger_iden_prf,
            "trigger_class_prf": trigger_class_prf,
            "arg_iden_prf": arg_iden_prf,
            "arg_class_prf": arg_class_prf,
        }
        return prf_dict
    else:
        rel_prf = metrics.get_prf_scores(ere_cpg_dict["rel_cpg"][0], ere_cpg_dict["rel_cpg"][1],
                                         ere_cpg_dict["rel_cpg"][2])
        ent_prf = metrics.get_prf_scores(ere_cpg_dict["ent_cpg"][0], ere_cpg_dict["ent_cpg"][1],
                                         ere_cpg_dict["ent_cpg"][2])
        prf_dict = {
            # "rel_prf": rel_prf,
            "ent_prf": ent_prf,
        }
        return prf_dict


def main():
    # Load Data

    test_data_path_dict = {}
    for file_path in glob.glob(test_data_path):
        file_name = re.search("(.*?)\.json", file_path.split("\\")[-1]).group(1)
        test_data_path_dict[file_name] = file_path

    test_data_dict = {}
    for file_name, path in test_data_path_dict.items():
        test_data_dict[file_name] = json.load(open(path, "r", encoding="utf-8"))

    # Split

    if config["encoder"] == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False,
                                                      do_lower_case=False)
        tokenize = tokenizer.tokenize
        get_tok2char_span_map = lambda text: \
            tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    elif config["encoder"] in {"BiLSTM", }:
        tokenize = lambda text: text.split(" ")

        def get_tok2char_span_map(text):
            tokens = text.split(" ")
            tok2char_span = []
            char_num = 0
            for tok in tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1  # +1: whitespace
            return tok2char_span

    preprocessor = Preprocessor(tokenize_func=tokenize,
                                get_tok2char_span_map_func=get_tok2char_span_map)

    all_data = []
    for data in list(test_data_dict.values()):
        all_data.extend(data)

    max_tok_num = 0
    for sample in tqdm(all_data, desc="Calculate the max token number"):
        tokens = tokenize(sample["text"])
        max_tok_num = max(len(tokens), max_tok_num)

    split_test_data = False

    global max_seq_len
    if max_tok_num > max_seq_len:
        split_test_data = True
        print(
            "max_tok_num: {}, lagger than max_test_seq_len: {}, test data will be split!".format(max_tok_num,
                                                                                                 max_seq_len))
    else:
        print("max_tok_num: {}, less than or equal to max_test_seq_len: {}, no need to split!".format(max_tok_num,
                                                                                                      max_seq_len))
    max_seq_len = min(max_tok_num, max_seq_len)

    if config["hyper_parameters"]["force_split"]:
        split_test_data = True
        print("force to split the test dataset!")

    ori_test_data_dict = copy.deepcopy(test_data_dict)
    if split_test_data:
        test_data_dict = {}
        for file_name, data in ori_test_data_dict.items():
            test_data_dict[file_name] = preprocessor.split_into_short_samples(data,
                                                                              max_seq_len,
                                                                              sliding_len=config["hyper_parameters"][
                                                                                  "sliding_len"],
                                                                              encoder=config["encoder"],
                                                                              data_type="test")

    # Decoder(Tagger)

    rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
    ent2id = json.load(open(ent2id_path, "r", encoding="utf-8"))
    handshaking_tagger = HandshakingTaggingScheme(rel2id, max_seq_len, ent2id)
    tag_size = handshaking_tagger.get_tag_size()

    # Dataset

    if config["encoder"] == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False,
                                                      do_lower_case=False)
        data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)
        get_tok2char_span_map = lambda text: \
            tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]

    elif config["encoder"] in {"BiLSTM", }:
        # token2idx_path = os.path.join(*config["token2idx"])
        token2idx_path = os.path.join(data_home, experiment_name, config["token2idx"])
        token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
        idx2token = {idx: tok for tok, idx in token2idx.items()}

        def text2indices(text, max_seq_len):
            input_ids = []
            tokens = text.split(" ")
            for tok in tokens:
                if tok not in token2idx:
                    input_ids.append(token2idx['<UNK>'])
                else:
                    input_ids.append(token2idx[tok])
            if len(input_ids) < max_seq_len:
                input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))
            input_ids = torch.tensor(input_ids[:max_seq_len])
            return input_ids

        def get_tok2char_span_map(text):
            tokens = text.split(" ")
            tok2char_span = []
            char_num = 0
            for tok in tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1  # +1: whitespace
            return tok2char_span

        data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map, handshaking_tagger)

    # Model

    if config["encoder"] == "BERT":
        encoder = BertModel.from_pretrained(config["bert_path"])
        hidden_size = encoder.config.hidden_size
        rel_extractor = TPLinkerPlusBert(encoder,
                                         tag_size,
                                         config["hyper_parameters"]["shaking_type"],
                                         config["hyper_parameters"]["inner_enc_type"],
                                         )

    elif config["encoder"] in {"BiLSTM", }:
        glove = Glove()
        glove = glove.load(word_embedding_path)

        # prepare embedding matrix
        word_embedding_init_matrix = np.random.normal(-1, 1,
                                                      size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
        count_in = 0

        # 在预训练词向量中的用该预训练向量
        # 不在预训练集里的用随机向量
        for ind, tok in tqdm(idx2token.items(), desc="Embedding matrix initializing..."):
            if tok in glove.dictionary:
                count_in += 1
                word_embedding_init_matrix[ind] = glove.word_vectors[glove.dictionary[tok]]

        print(
            "{:.4f} tokens are in the pretrain word embedding matrix".format(count_in / len(idx2token)))  # 命中预训练词向量的比例
        word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)

        rel_extractor = TPLinkerPlusBiLSTM(word_embedding_init_matrix,
                                           hyper_parameters["emb_dropout"],
                                           hyper_parameters["enc_hidden_size"],
                                           hyper_parameters["dec_hidden_size"],
                                           hyper_parameters["rnn_dropout"],
                                           tag_size,
                                           hyper_parameters["shaking_type"],
                                           hyper_parameters["inner_enc_type"],
                                           )

    rel_extractor = rel_extractor.to(device)

    # Merics

    metrics = MetricsCalculator(handshaking_tagger)

    # Prediction

    # get model state paths
    target_run_ids = set(config["run_ids"])
    run_id2model_state_paths = {}
    for root, dirs, files in os.walk(model_state_dir):
        for file_name in files:
            run_id = root[-8:]
            if re.match(".*model_state.*\.pt", file_name) and run_id in target_run_ids:
                if run_id not in run_id2model_state_paths:
                    run_id2model_state_paths[run_id] = []
                model_state_path = os.path.join(root, file_name)
                run_id2model_state_paths[run_id].append(model_state_path)

    # only last k models
    k = config["last_k_model"]
    for run_id, path_list in run_id2model_state_paths.items():
        run_id2model_state_paths[run_id] = get_last_k_paths(path_list, k)
    print("run_id2model_state_paths", run_id2model_state_paths)
    # predict
    res_dict = {}
    predict_statistics = {}
    for file_name, short_data in test_data_dict.items():
        ori_test_data = ori_test_data_dict[file_name]
        for run_id, model_path_list in run_id2model_state_paths.items():
            save_dir4run = os.path.join(save_res_dir, run_id)
            if config["save_res"] and not os.path.exists(save_dir4run):
                os.makedirs(save_dir4run)

            for model_state_path in model_path_list:
                res_num = re.search("(\d+)", model_state_path.split("\\")[-1]).group(1)
                global save_path
                if use_type == "eval":
                    save_path = os.path.join(save_dir4run, "{}_res_{}.json".format(file_name, res_num))
                if os.path.exists(save_path):
                    pred_sample_list = [json.loads(line) for line in open(save_path, "r", encoding="utf-8")]
                    print("{} already exists, load it directly!".format(save_path))
                else:
                    # load model state
                    rel_extractor.load_state_dict(torch.load(model_state_path))
                    rel_extractor.eval()
                    print("run_id: {}, model state {} loaded".format(run_id, model_state_path.split("\\")[-1]))

                    # predict
                    pred_sample_list = predict(data_maker, rel_extractor, short_data, ori_test_data,
                                               split_test_data, handshaking_tagger)

                res_dict[save_path] = pred_sample_list
                # predict_statistics[save_path] = len([s for s in pred_sample_list if len(s["relation_list"]) > 0])
                predict_statistics[save_path] = len([s for s in pred_sample_list])
    pprint(predict_statistics)

    # check
    for path, res in res_dict.items():
        for sample in tqdm(res, desc="check char span"):
            text = sample["text"]
            for rel in sample["relation_list"]:
                assert rel["subject"] == text[rel["subj_char_span"][0]:rel["subj_char_span"][1]]
                assert rel["object"] == text[rel["obj_char_span"][0]:rel["obj_char_span"][1]]

    # save
    if save_res:
        for path, res in res_dict.items():
            with open(path, "w", encoding="utf-8") as file_out:
                for sample in tqdm(res, desc="Output"):
                    # if len(sample["relation_list"]) == 0:
                    #     continue
                    json_line = json.dumps(sample, ensure_ascii=False)
                    file_out.write("{}\n".format(json_line))
            print("The result is saved to ", path)

    # score
    if score:
        filepath2scores = {}
        for file_path, pred_samples in res_dict.items():
            file_name = re.search("(.*?)_res_\d+\.json", file_path.split("\\")[-1]).group(1)
            gold_test_data = ori_test_data_dict[file_name]
            prf_dict = get_test_prf(handshaking_tagger, metrics,
                                    pred_samples, gold_test_data, pattern=hyper_parameters["match_pattern"])
            filepath2scores[file_path] = prf_dict
        print("---------------- Results -----------------------")
        pprint(filepath2scores)


if __name__ == '__main__':
    main()
