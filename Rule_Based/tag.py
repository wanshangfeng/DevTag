class DataTag(object):
    """docstring for """
    def __init__(self):
        self.application = None
        self.device_type = None
        self.os = None
        self.product = None
        self.vendor = None
        self.version = None
    
    def get_dict(self,tag):
        tag["application"] = tag["application"]
        tag["device_type"] = tag["device_type"]
        tag["os"] = tag["os"]
        tag["product"] = tag["product"]
        tag["vendor"] = tag["vendor"]
        tag["version"] = tag["version"]

        if len(tag) > 0:
            return tag
    
    def get_message(self):
        message_fields = [
             self.product,
             self.vendor,
             self.version,
             self.os,
             self.device_type
             self.application
        ]

        messages = [message for message in message_fields if message]
        if messages:
            return " ".join(messages)

        return None
        

