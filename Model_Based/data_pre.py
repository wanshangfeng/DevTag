import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocessing(text):
    text = filter_tags(text)
    # 分词
    tokens = [word for word in word_tokenize(text)]
    # 去停词
    stop = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop]
    # 去特殊字符
    characters = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&',
                  '!', '*', '@', '#', '$', '%', '-', '...', '200', '|', '=', '+']  # ,'-'
    tokens = [word for word in tokens if word not in characters]
    # 大小写转换，统一为小写
    tokens = [word.lower() for word in tokens]
    # print(tokens)
    return tokens


# 过滤HTML中的标签
# 将HTML中标签等信息去掉
# @param htmlstr HTML字符串
def filter_tags(htmlstr):
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)                 # 匹配CDATA
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)     # style
    re_br = re.compile('<br\s*?/?>')            # 处理换行
    re_h = re.compile('</?\w+[^>]*>')           # HTML标签
    re_comment = re.compile('<!--[^>]*-->')     # HTML注释
    re_stopwords = re.compile('\u3000')         # 去除无用的'\u3000'字符
    re_chinese = re.compile(r'[\u4e00-\u9fa5]') # 汉字
    s = re_cdata.sub('', htmlstr)    # 去掉CDATA
    s = re_script.sub('', s)         # 去掉SCRIPT
    s = re_style.sub('', s)          # 去掉style
    s = re_br.sub('\n', s)           # 将br转换为换行
    s = re_h.sub('', s)              # 去掉HTML 标签
    s = re_comment.sub('', s)        # 去掉HTML注释
    s = re_stopwords.sub('', s)      # 去除无用的'\u3000'字符
    s = re_chinese.sub('', s)        # 排除汉字
    blank_line = re.compile('\n+')  # 去掉多余的空行
    s = blank_line.sub('\n', s)
    s = replaceCharEntity(s)        # 替换实体
    # 提取英文字符和数字
    s = re.sub(u"([^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a])", " ", s)
    return s


def replaceCharEntity(htmlstr):
    """使用正常的字符替换HTML中特殊的字符实体"""
    CHAR_ENTITIES = {'nbsp':' ','160':' ',
                  'lt':'<','60':'<',
                  'gt':'>','62':'>',
                  'amp':'&','38':'&',
                  'quot':'"','34':'"',}

    re_charEntity=re.compile(r'&#?(?P<name>\w+);')
    sz=re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()  # entity全称，如&gt;
        key = sz.group('name')  # 去除&;后entity,如&gt;为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            # 以空串代替
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr


def test_data_pre(path, to_path):
    """对测试数据做预处理"""
    filepath = [path]
    print("Data preprocessing...")
    for file in filepath:
        print('Processing ', file)
        with open(file, 'r', encoding='utf-8') as f, open(to_path, 'w', encoding='utf-8') as f2:
            for line in f.readlines():
                text = preprocessing(line)
                f2.write(' '.join(text)+'\n')


def all_data_pre():
    """读取原json数据集，预处理后得到训练集、测试集、验证集、类别集合。可根据需要进行修改。"""
    import numpy as np
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import path_config as path

    def save_labels(path_labels, labels):
        with open(path_labels, 'w', encoding='utf-8') as f:
            for line in labels:
                f.write(''.join(line) + '\n')

    def split_train_test_val(banner_data, tag_data):
        X_train, X_test, y_train, y_test = train_test_split(banner_data, tag_data, test_size=0.2, random_state=2)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=2)
        # print(len(X_train), len(y_train), len(X_test), len(y_test))
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.95, random_state=2)
        # print(len(X_train), len(y_train), len(X_test), len(y_test))
        return X_train, X_test, X_val, y_train, y_test, y_val

    def y_to_id(labels, y_train, y_test, y_val):
        le = LabelEncoder()
        le.fit(labels)
        y_train = le.transform(y_train).tolist()
        y_test = le.transform(y_test).tolist()
        y_val = le.transform(y_val).tolist()
        return y_train, y_test, y_val

    def save_xy(file, X, y):
        with open(file, 'w', encoding='utf-8') as f:
            for line1, line2 in list(zip(X, y)):
                f.write(' '.join(line1) + '\t' + str(line2) + '\n')

    banner_data = []
    tag_data = []
    filepath_list = ['data3_49000/somedata49000.json']  # 原始数据路径
    for file in filepath_list:  # 遍历文件夹
        print('正在处理', file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                dic = json.loads(line)
                text = preprocessing(dic['banner'])
                banner_data.append(text)
                tag_data.append(np.array(dic['device_type'].lower() + '/' + dic['brand'].lower() + '/' + dic['product'].lower()))
    labels = np.unique(tag_data)
    for label in labels:
        print(label, tag_data.count(label))
    save_labels(path.class_path, labels)
    # 打乱数据
    banner_data = np.array(banner_data)
    tag_data = np.array(tag_data)
    index = [i for i in range(len(banner_data))]
    np.random.shuffle(index)
    banner_data = banner_data[index]
    tag_data = tag_data[index]

    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test_val(banner_data, tag_data)
    print(len(X_train), len(X_test), len(X_val), labels.size)

    y_train3, y_test3, y_val3 = y_to_id(labels, y_train, y_test, y_val)
    save_xy(path.train_path, X_train, y_train3)
    save_xy(path.test_path, X_test, y_test3)
    save_xy(path.val_path, X_val, y_val3)