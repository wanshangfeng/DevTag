#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True)
    parser.add_argument('-T', '--tag', type=str, required=True, choices=["dvp", "pd", "vd", "vp", "d", "v", "p"])
    parser.add_argument('-M', '--model', default="BiLSTM", type=str)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    tag = args.tag
    model = args.model

    logging.info("Tag: {}, Model: {}.".format(tag, model))

    dirname = os.getcwd()
    res_path = dirname + "\\test\\res-{}.json".format(tag)
    devtag_path = dirname + "\\test\\devtag-{}.json".format(tag)

    start_time = datetime.utcnow()

    if tag in ["d", "v", "p"]:
        from tplinker_plus import evaluation
        from tplinker_plus import res2tag

        evaluation.test_data_path = os.path.abspath(args.filename)
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

        evaluation.test_data_path = os.path.abspath(args.filename)
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
