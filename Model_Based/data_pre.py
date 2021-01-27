import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import json

def extracTextFromHtml(html):
    html_a = html
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except:
        return ""
    if str(type(soup.script)) != str("<type 'NoneType'>"):
        for scripts in soup.find_all('script'):
            if scripts.string:
                html_a = html_a.replace(str(scripts.string), '')
    if str(type(soup.style)) != str("<type 'NoneType'>"):
        for styles in soup.find_all('style'):
            if styles.string:
                html_a = html_a.replace(str(styles.string), '')
    try:
        soup = BeautifulSoup(html_a, 'ihtml.parser')
    except:
        soup = BeautifulSoup(html_a, 'lxml')
    text = soup.get_text()
    return text


def preprocessing(text):
    # 从网页中提取文本信息
    text = extracTextFromHtml(text)
    text = ' '.join(text.split())
    # 去除时间、ip地址等信息
    time_pattern1 = "(0\d{1}|1\d{1}|2[0-3]):([0-5]\d{1})"
    time_pattern2 = "(0\d{1}|1\d{1}|2[0-3]):[0-5]\d{1}:([0-5]\d{1})"
    ip_pattern = "((?:(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))\.){3}(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d))))"
    p_ip = re.compile(ip_pattern)
    p_time1 = re.compile(time_pattern1)
    p_time2 = re.compile(time_pattern2)
    text = p_ip.sub(" ", text)
    text = p_time2.sub(" ", text)
    text = p_time1.sub(" ", text)

    text = text.replace(r'\r', ' ').replace(r'\n', ' ').replace(r'\t', ' ')  # 去除\r, \n, \t
    text = re.sub(u"([^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a])", ' ', text)  # 提取英文字符和数字


    tokens = [word for word in word_tokenize(text)]  # 分词
    stop = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop]  # 去停词
    tokens = [word.lower() for word in tokens]  # 大小写转换，统一为小写
    # print(tokens)
    return tokens


def test_data_pre(path, to_path):
    """对测试数据做预处理"""
    filepath = [path]
    print("Data preprocessing...")
    for file in filepath:
        print('Processing ', file)
        with open(file, 'r', encoding='utf-8') as f, open(to_path, 'w', encoding='utf-8') as f2:
            for line in f.readlines():
                dict = json.loads(line)
                text = preprocessing(dict['banner'])
                # print(text)
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
    filepath_list = ['dataset/data.json']  # 原始数据路径
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