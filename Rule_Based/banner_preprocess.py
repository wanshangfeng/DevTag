import re

def clean_banner(protocol,banner):
    banner = banner.lower()
    # banner = banner.replace(r"\r\n","")

    if protocol == "FTP" or protocol == "TELNET":
        return clean_TF(banner)

    if protocol == "RTSP":
        return clean_RTSP(banner)

    if protocol == "HTTP":
        return clean_HTTP(banner)

    return banner


time_pattern1 = "(0\d{1}|1\d{1}|2[0-3]):([0-5]\d{1})"
time_pattern2 = "(0\d{1}|1\d{1}|2[0-3]):[0-5]\d{1}:([0-5]\d{1})"
ip_pattern = "((?:(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))\.){3}(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d))))"
pattern_32 = "[0-9a-z]{32}"
pattern_20 = "[0-9a-z]{20}"
pattern_15 = "[0-9A-Z]{15}"
pattern_13 = "[0-9A-Z]{13}"      
pattern_10 = "[0-9]{10}"

def clean_TF(banner):

    p_ip = re.compile(ip_pattern)
    p_time1 = re.compile(time_pattern1)
    p_time2 = re.compile(time_pattern2)

    banner = p_time1.sub(" ",banner)
    banner = p_time2.sub(" ",banner)
    banner = p_ip.sub(" ",banner)

    return banner

def clean_RTSP(banner):

    p_32 = re.compile(pattern_32)
    p_10 = re.compile(pattern_10)
    p_20 = re.compile(pattern_20)
    p_15 = re.compile(pattern_15)
    p_13 = re.compile(pattern_13)
    p_time2 = re.compile(time_pattern2)

    banner = p_32.sub(" ",banner)
    banner = p_20.sub(" ",banner)
    banner = p_15.sub(" ",banner)
    banner = p_13.sub(" ",banner)
    banner = p_10.sub(" ",banner)

    banner = p_time2.sub(" ",banner)

    return banner


def clean_HTTP(banner):
    p_32 = re.compile(pattern_32)
    p_10 = re.compile(pattern_10)

    p_ip = re.compile(ip_pattern)
    p_time1 = re.compile(time_pattern1)
    p_time2 = re.compile(time_pattern2)

    banner = p_32.sub(" ",banner)
    banner = p_10.sub(" ",banner)

    banner = p_ip.sub(" ",banner)
    banner = p_time1.sub(" ",banner)
    banner = p_time2.sub(" ",banner)

    return banner
    