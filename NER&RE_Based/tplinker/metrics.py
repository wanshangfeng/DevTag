#!/usr/bin/env python
# coding: utf-8
import os
import sys
import config
import pandas as pd
import json
from sklearn import metrics

config = config.eval_config
experiment_name = config["exp_name"]
save_res_dir = os.path.join(config["save_res_dir"], experiment_name)
run_ids = config["run_ids"]
data_home = config["data_home"]
report_dir = "../reports/{}.csv".format(experiment_name)


for run_id in run_ids:
    save_dir4run = os.path.join(save_res_dir, run_id)
    if not os.path.exists(save_dir4run):
        print("The result does not exist.")
        sys.exit(-1)
    filepath_list = os.listdir(save_dir4run)
    for file in filepath_list:
        if "out" in file:
            out_file = os.path.join(save_dir4run, file)
        if "tag" in file:
            tag_file = os.path.join(save_dir4run, file)


def load_tag(file):
    tags = []
    with open(file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            dic = json.loads(line)
            tags.append(dic['device_type'].lower() + '/' + dic['vendor'].lower() + '/' + dic['product'].lower())
    return tags


labels_all = load_tag(tag_file)
predict_all = load_tag(out_file)
acc = metrics.accuracy_score(labels_all, predict_all)
report_display = metrics.classification_report(labels_all, predict_all, digits=4, output_dict=True)
print(report_display)

df = pd.DataFrame(report_display).transpose()
df.to_csv(report_dir, index=True)

