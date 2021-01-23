import json
import path_config as path 


def generate_rules_from_protocol(protocol_rule_name,device_type,vendor):
    rules = list()
    with open(protocol_rule_name,'r',encoding='utf-8') as f:
        for line in f:
            line = line.lower()
            line_to_json = json.loads(line)
            devi = line_to_json['tag']['device_type']
            ven = line_to_json['tag']['vendor']

            if device_type.lower() != "all":
                if devi.lower() != device_type.lower():
                    continue

            if vendor.lower() != "all":
                if ven.lower() != vendor.lower():
                    continue

            rules.append(line_to_json)

    return rules



class FTP():

        def __init__(self):
            self.rule = list() 

        def generate_rules(self,device_type,vendor):
            self.rule = generate_rules_from_protocol(path.FTP_rules_data,device_type,vendor)

            

class HTTP():

        def __init__(self):
            self.rule = list()
        
        def generate_rules(self,device_type,vendor):
            self.rule = generate_rules_from_protocol(path.HTTP_rules_data,device_type,vendor)



class RTSP():

        def __init__(self):
            self.rule = list() 

        def generate_rules(self,device_type,vendor):
            self.rule = generate_rules_from_protocol(path.RTSP_rules_data,device_type,vendor)


class SMTP():

        def __init__(self):
            self.rule = list() 

        def generate_rules(self,device_type,vendor):
            self.rule = generate_rules_from_protocol(path.SMTP_rules_data,device_type,vendor)


class TELNET():

        def __init__(self):
            self.rule = list() 

        def generate_rules(self,device_type,vendor):
            self.rule = generate_rules_from_protocol(path.TELNET_rules_data,device_type,vendor)

