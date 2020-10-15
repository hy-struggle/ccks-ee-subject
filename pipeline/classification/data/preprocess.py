# /usr/bin/env python
# coding=utf-8
"""preprocess"""
from pathlib import Path
import json
import re
import random
import logging

from utils import STR2IO, set_logger
SEED = 2020
SRC_DATA_DIR = Path('../ccks 4_1 Data')
DATA_DIR = Path('./')


def merge_label(content):
    """将文本相同的数据标签合并
    """
    result = []
    for s in content:
        # init
        tmp = [s[0], s[1], [s[2]], [[s[3]]]]
        # 文本相同则合并标签
        for compare in content:
            # 如果文本相同，且非自身，且类别不存在
            if s[1] == compare[1] and s[0] != compare[0] and compare[2] not in tmp[2]:
                tmp[2].append(compare[2])
                tmp[3].append([compare[3]])
            # 如果文本相同，且非自身，且类别已存在
            elif s[1] == compare[1] and s[0] != compare[0] and compare[2] in tmp[2]:
                # 获取对应位置
                idx = tmp[2].index(compare[2])
                # 将实体加入已有位置
                tmp[3][idx].append(compare[3])
        result.append(tmp)
    return result


def filter_chars(text):
    """过滤无用字符
    :param text: 文本
    """
    # 找出文本中所有非中，英和数字的字符
    add_chars = set(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9]', text))
    extra_chars = set(r"""!！￥$%*（）()-——【】:：“”";；'‘’，。？,.?、""")
    add_chars = add_chars.difference(extra_chars)

    # 替换特殊字符组合
    text = re.sub('{IMG:.?.?.?}', '', text)
    text = re.sub(r'<!--IMG_\d+-->', '', text)
    text = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text)  # 过滤网址
    text = re.sub('<a[^>]*>', '', text).replace("</a>", "")  # 过滤a标签
    text = re.sub('<P[^>]*>', '', text).replace("</P>", "")  # 过滤P标签
    text = re.sub('<strong[^>]*>', ',', text).replace("</strong>", "")  # 过滤strong标签
    text = re.sub('<br>', ',', text)  # 过滤br标签
    text = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text).replace("()", "")  # 过滤www开头的网址
    text = re.sub(r'\s', '', text)  # 过滤不可见字符
    text = re.sub('Ⅴ', 'V', text)

    # 清洗
    for c in add_chars:
        text = text.replace(c, '')
    return text


def get_train_data(content, mode='train'):
    """获取验证集
    :param content (List[List[str]]): 源文本
    """
    with open(DATA_DIR / f'{mode}.data', 'w', encoding='utf-8') as f:
        result = []
        # write to json
        for idx, c in enumerate(content):
            # 清洗数据
            text = filter_chars(c[1])
            # 抽取实体位置
            result.append({
                'id': c[0],
                'context': text,
                'entity_label': [STR2IO[tp] for tp in c[2]],
            })
        print(f'get {len(result)} {mode} samples.')
        json.dump(result, f, indent=4, ensure_ascii=False)


def get_test_data():
    """获取测试集
    """
    result = []
    with open(SRC_DATA_DIR / 'event_entity_dev_data.csv', 'r', encoding='utf-8') as f, \
            open(DATA_DIR / 'test.data', 'w', encoding='utf-8') as f_json:
        # get src data
        content = [line.strip().split('\t') for line in f]
        # write to json
        for idx, c in enumerate(content):
            # 清洗数据
            text = filter_chars(c[1])
            # write data
            result.append({
                'id': c[0],
                'context': text,
                'entity_label': []
            })
        print(f'get {len(result)} test samples.')
        json.dump(result, f_json, indent=4, ensure_ascii=False)


def get_data_mrc():
    """获取mrc格式数据集
    """
    with open(SRC_DATA_DIR / 'event_entity_train_data_label.csv', 'r', encoding='utf-8') as f:
        # get src data
        content = []
        count = 0
        for line in f:
            line_sp = line.strip().split('\t')
            if line_sp[2] == 'NaN' or line_sp[3] == 'NaN':
                count += 1
                # 只取8000条NaN样本
                if count < 8000:
                    content.append(line_sp)
            else:
                content.append(line_sp)

        logging.info('Merge label...')
        # 合并文本相同的数据
        content = merge_label(content)
        logging.info('-done')

        # shuffle&split
        random.seed(SEED)
        random.shuffle(content)
        train_content = content[:-900]
        val_content = content[-900:]

    # get train data
    get_train_data(train_content, mode='train')
    # get val data
    get_train_data(val_content, mode='val')
    # get test data
    get_test_data()


if __name__ == '__main__':
    set_logger(save=False)
    get_data_mrc()
