#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import json
import os
import re
from common.utils import Preprocessor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True)
    parser.add_argument('-T', '--tag', type=str, required=True, choices=["dvp", "pd", "vd", "vp", "d", "v", "p"])
    parser.add_argument('-M', '--model', default="BiLSTM", type=str)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    tag = args.tag
    model = args.model
    file = os.path.abspath(args.filename)

    file_name = re.search("(.*?)\.json", file.split("\\")[-1]).group(1)
    dirname = os.getcwd()
    pre_path = dirname + "\\test\\{}-pre.json".format(file_name)
    res_path = dirname + "\\test\\{}-res-{}.json".format(file_name, tag)
    devtag_path = dirname + "\\test\\{}-devtag-{}.json".format(file_name, tag)

    logging.info("flie_name: {}".format(file_name))
    logging.info("Tag: {}, Model: {}".format(tag, model))
    logging.info("pre_path: {}".format(pre_path))
    logging.info("devtag_path: {}".format(devtag_path))

    if not os.path.exists(pre_path):  # Preprocessing is performed only once
        try:
            data = json.load(open(file, "r", encoding="utf-8"))
        except:
            f = open(file, "r", encoding="utf-8")
            data = []
            for line in f.readlines():
                dic = json.loads(line)
                data.append(dic)
            f.close()
        preprocessor = Preprocessor()
        preprocessor.clean4test(data, save_path=pre_path, add_id=True)

    start_time = datetime.utcnow()

    if tag in ["d", "v", "p"]:
        from tplinker_plus import evaluation
        from tplinker_plus import res2tag

        evaluation.test_data_path = pre_path
        evaluation.data_home = os.getcwd() + evaluation.data_home.lstrip("..")
        evaluation.rel2id_path = os.getcwd() + evaluation.rel2id_path.lstrip("..")
        evaluation.ent2id_path = os.getcwd() + evaluation.ent2id_path.lstrip("..")
        evaluation.word_embedding_path = os.getcwd() + evaluation.word_embedding_path.lstrip("..")
        evaluation.experiment_name = "data-{}".format(tag)
        evaluation.model_state_dir = os.getcwd() + "\\tplinker_plus" + evaluation.model_state_dir.lstrip(".")
        evaluation.use_type = "test"
        evaluation.save_path = res_path
        evaluation.save_res = True
        evaluation.score = False

        evaluation.main()
        res2tag.res2tag(res_path, devtag_path)

    if tag in ["dvp", "pd", "vd", "vp"]:
        from tplinker import evaluation
        from tplinker import res2tag

        evaluation.test_data_path = pre_path
        evaluation.data_home = os.getcwd() + evaluation.data_home.lstrip("..")
        evaluation.rel2id_path = os.getcwd() + evaluation.rel2id_path.lstrip("..")
        evaluation.word_embedding_path = os.getcwd() + evaluation.word_embedding_path.lstrip("..")
        evaluation.experiment_name = "data-{}".format(tag)
        evaluation.model_state_dir = os.getcwd() + "\\tplinker" + evaluation.model_state_dir.lstrip(".")
        evaluation.use_type = "test"
        evaluation.save_path = res_path
        evaluation.save_res = True
        evaluation.score = False

        evaluation.main()
        res2tag.res2tag(res_path, devtag_path)

    end_time = datetime.utcnow()

    duration = end_time - start_time
    logging.info("the running time is %s" % duration.total_seconds())


if __name__ == "__main__":
    main()
