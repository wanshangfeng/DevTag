#!/usr/bin/env python
# coding: utf-8
import os
import sys
import re
import json
from . import config


config = config.eval_config
experiment_name = config["exp_name"]
save_res_dir = os.path.join(config["save_res_dir"], experiment_name)
run_ids = config["run_ids"]
data_home = config["data_home"]


def res2tag(save_path, out_path):
    text_id_list = []
    with open(save_path, "r", encoding="utf-8") as f, \
            open(out_path, "w", encoding="utf-8") as file_out:
        for line in f.readlines():
            pred_sample = json.loads(line)
            ent_list = pred_sample["entity_list"]
            # print(ent_list)
            device_type, brand, product = "", "", ""

            for ent in ent_list:
                if ent["type"] == "device_type":
                    device_type = ent["text"]
                elif ent["type"] == "brand":
                    brand = ent["text"]
                elif ent["type"] == "product":
                    product = ent["text"]

            devtag = {
                'vendor': brand,
                'product': product,
                'device_type': device_type,
                'banner': pred_sample["text"]
            }
            text_id_list.append(int(re.search("(\d+)", pred_sample["id"]).group(1)))
            json.dump(devtag, file_out)
            file_out.write('\n')
    return text_id_list


if __name__ == '__main__':

    for run_id in run_ids:
        save_dir4run = os.path.join(save_res_dir, run_id)
        if not os.path.exists(save_dir4run):
            print("The result does not exist.")
            sys.exit(-1)
        filepath_list = os.listdir(save_dir4run)
        for file in filepath_list:
            if "res" not in file:
                continue
            print('正在处理', file)
            res_num = re.search("(\d+)", file).group(1)
            file_name = file.split("_res_")[0]
            # print(res_num, file_name)
            save_path = os.path.join(save_dir4run, file)
            out_path = os.path.join(save_dir4run, "{}_out_{}.json".format(file_name, res_num))
            # print(save_path, out_path)
            if os.path.exists(save_path) and not os.path.exists(out_path):
                # if os.path.exists(save_path):
                text_id_list = res2tag(save_path, out_path)


            test_out_path = os.path.join(save_dir4run, "{}_tag.json".format(file_name))
            test_data_path = os.path.join(data_home, experiment_name, file_name + ".json")
            if not os.path.exists(test_out_path):
                # print(test_data_path, test_out_path)
                test_sample = json.load(open(test_data_path, "r", encoding="utf-8"))
                file_out = open(test_out_path, "w", encoding="utf-8")
                for text_id in text_id_list:
                    pred_sample = test_sample[text_id]
                    ent_list = pred_sample["entity_list"]
                    # print(ent_list)
                    device_type, brand, product = "", "", ""

                    for ent in ent_list:
                        if ent["type"] == "DEFAULT":
                            device_type = ent["text"]
                        elif ent["type"] == "brand":
                            brand = ent["text"]
                        elif ent["type"] == "product":
                            product = ent["text"]
                    devtag = {
                        'vendor': brand,
                        'product': product,
                        'device_type': device_type,
                        'banner': pred_sample["text"]
                    }
                    json.dump(devtag, file_out)
                    file_out.write('\n')
                file_out.close()
