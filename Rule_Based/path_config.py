# coding: UTF-8
rules_data = 'rules'  # 规则数据集目录

FTP_rules_data = rules_data + '/FTP_RULES.json'   # ftp协议的规则数据
HTTP_rules_data = rules_data + '/HTTP_RULES.json'   # http协议的规则数据
RTSP_rules_data = rules_data + '/RTSP_RULES.json'   # rtsp协议的规则数据
SMTP_rules_data = rules_data + '/SMTP_RULES.json'   # smtp协议的规则数据
TELNET_rules_data = rules_data + '/TELNET_RULES.json'   # telnet协议的规则数据

test_path = 'test'   # 测试数据目录

ftp_test_data = test_path + '/test_ftp.json'      # ftp的banner测试数据
ftp_test_result = test_path + '/DevTag-ftp.json'  # ftp测试数据的结果

http_test_data = test_path + '/test_http.json'    # http的banner测试数据
http_test_result = test_path + '/DevTag-http.json'  # http测试数据的结果