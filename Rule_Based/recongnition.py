### 识别banner
import re
from lables.extra_rules import FTP,HTTP,RTSP,SMTP,TELNET

from preprocess import clean_all


def rules(protocol,device_type,vendor):
    rule = None

    if protocol.lower() == "ftp":
        rule = FTP()
    if protocol.lower() == "http":
        rule = HTTP()
    if protocol.lower() == "rtsp":
        rule = RTSP()
    if protocol.lower() == "smtp":
        rule = SMTP()
    if protocol.lower() == "telnet":
        rule = TELNET()

    rule.generate_rules(device_type,vendor)
    return rule.rule

# number = ["1","2","3","4","5"]
regu_col = ['$1','$2','$3','$4','$5']

def tag_all(banner,all_rules,tag_list):
    if len(all_rules) < 1:
        return None
    

    for item in all_rules:

        filterword = item["filterword"]
        tag = item["tag"]
        product = tag["product"]
        contain_all = True

        if type(filterword) == list:
            for fil in filterword:
                if banner.find(fil) < 0:
                    contain_all = False
                    break
        else:
            words = filterword.strip("####").split("####")
            ##如何product不是正则
            is_regual = False
            pro_index = 0
            replace_index = 0
            for item in regu_col:
                if product.find(item) >=0:
                    replace_index = item
                    item = item.replace('$','')
                    pro_index = int(item.strip())
                    is_regual = True
                    break

            if is_regual == False:
                for i in words:
                    i = i.strip()
                    i_re = re.compile(i,re.IGNORECASE)
                    if i_re.search(banner) is None:
                        contain_all = False
                        break
            else:
                if_find = False
                for i in words:
                    i = i.strip()
                    i_re = re.compile(i,re.IGNORECASE)
                    if i_re.search(banner) is None:
                        contain_all = False
                        break
                    num = i.count("(")

                    if pro_index <= num and if_find == False:
                        produ = i_re.search(banner).group(pro_index)
                        product = product.replace(replace_index,produ)
                        tag["product"] = product
                        if_find = True

                    if pro_index > num and num > 0:
                        pro_index = pro_index - num             


        if contain_all:
            tag_list.append(tag) 

    if len(tag_list) > 0:
        return tag_list
    else:
        return None


###识别正则
def tag_rules(protocol,banner,device_type,vendor):

    tag_list = list()
    
    banner = clean_all(protocol,banner)
    all_rule = rules(protocol,device_type,vendor)
    tag_list = tag_all(banner,all_rule,tag_list)


    if tag_list is not None:
        return tag_list

    return None


if __name__ == "__main__":
    banner = "postfix"
    tag = tag_rules("SMTP",banner,device_type)
    print(tag)