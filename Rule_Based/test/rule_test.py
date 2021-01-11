from lables.extra_rules import FTP
from lables.extra_rules import HTTP
from lables.extra_rules import IMAP
from lables.extra_rules import POP3
from lables.extra_rules import RTSP
from lables.extra_rules import SMTP
from lables.extra_rules import TELNET



def test_rules(protocol):
    rule = None

    if protocol == "FTP":
        rule = FTP()
    if protocol == "HTTP":
        rule = HTTP()
    if protocol == "IMAP":
        rule = IMAP()
    if protocol == "POP3":
        rule = POP3()
    if protocol == "RTSP":
        rule = RTSP()
    if protocol == "SMTP":
        rule = SMTP()
    if protocol == "TELNET":
        rule = TELNET()

    rule.generate_rules()

    print(len(rule.rule.regular))
    print(len(rule.rule.strings))
    print(len(rule.rule.word))
    return rule



if __name__ == "__main__":
    test_rules("FTP")
    test_rules("HTTP")
    test_rules("IMAP")
    test_rules("POP3")
    test_rules("RTSP")
    test_rules("SMTP")
    test_rules("TELNET")