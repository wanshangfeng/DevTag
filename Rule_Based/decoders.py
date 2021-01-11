import json
import sys

class Decoder():

    def __init__(self, *args, **kwargs):
        pass

    def decode(self, s):
        raise NotImplementedError


class JSONDecoder(Decoder):

    def __init__(self, logger=None, *args, **kwargs):
        super(JSONDecoder, self).__init__(*args, **kwargs)
        self.logger = logger

    def decode(self, s):
        return json.loads(s)

class TSVDecoder(Decoder):

    def __init__(self, *args, **kwargs):
        super(TSVDecoder, self).__init__(*args, **kwargs)

    def decode(self, s):
        return s.rstrip().split("\t")


class NullDecoder(Decoder):

    def __init__(self, *args, **kwargs):
        super(NullDecoder, self).__init__(*args, **kwargs)

    def decode(self, s):
        return s