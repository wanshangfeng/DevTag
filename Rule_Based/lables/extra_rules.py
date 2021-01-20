import json


def generate_all_rules(filename,device_type,vendor):
    openfile = "rules/"+filename
    rules = list()
    with open(openfile,'r',encoding='utf-8') as f:
        for line in f:
            lines = json.loads(line)
            devi = lines['tag']['device_type']
            ven = lines['tag']['vendor']

            if device_type.lower() != "all":
                if devi.lower() != device_type.lower():
                    continue

            if vendor.lower() != "all":
                if ven.lower() != vendor.lower():
                    continue

            rules.append(lines)

    return rules



class FTP():

        def __init__(self):
            self.rule = list() 

        def generate_rules(self,device_type,vendor):
            self.rule = generate_all_rules("FTP_RULES.json",device_type,vendor)

            

class HTTP():

        def __init__(self):
            self.rule = list()
        
        def generate_rules(self,device_type,vendor):
            self.rule = generate_all_rules("HTTP_RULES.json",device_type,vendor)



class RTSP():

        def __init__(self):
            self.rule = list() 

        def generate_rules(self,device_type,vendor):
            self.rule = generate_all_rules("RTSP_RULES.json",device_type,vendor)


class SMTP():

        def __init__(self):
            self.rule = list() 

        def generate_rules(self,device_type,vendor):
            self.rule = generate_all_rules("SMTP_RULES.json",device_type,vendor)


class TELNET():

        def __init__(self):
            self.rule = list() 

        def generate_rules(self,device_type,vendor):
            self.rule = generate_all_rules("TELNET_RULES.json",device_type,vendor)

