import re
from rules import FTP, HTTP, RTSP, SMTP, TELNET

from banner_preprocess import clean_banner


def extra_rules(protocol, device_type, vendor):
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

    rule.generate_rules(device_type, vendor)
    return rule.rule

regex_note = ['$1','$2','$3','$4','$5']

def recognition_banner(banner, all_rules, tag_list):

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
            is_regex = False
            pro_index = 0
            replace_index = 0
            for item in regex_note:
                if product.find(item) >=0:
                    replace_index = item
                    item = item.replace('$', '')
                    pro_index = int(item.strip())
                    is_regex = True
                    break

            if is_regex == False:
                for i in words:
                    i = i.strip()
                    i_re = re.compile(i, re.IGNORECASE)
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


def tag_banner(protocol,banner,device_type,vendor):

    tag_list = list()
    
    banner = clean_banner(protocol,banner)
    device_rules = extra_rules(protocol,device_type,vendor)
    tag_list = recognition_banner(banner,device_rules,tag_list)

    if tag_list is not None:
        return tag_list

    return None


if __name__ == "__main__":
    banner = "postfix"
    tag = tag_rules("SMTP", banner, device_type)
    print(tag)