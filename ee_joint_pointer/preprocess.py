# /usr/bin/env python
# coding=utf-8
"""preprocess"""
from pathlib import Path
import json
import re
import random
import logging
from multiprocessing import Pool
import functools

from utils import set_logger

SRC_DATA_DIR = Path('./ccks 4_1 Data')
DATA_DIR = Path('./data')


def findall(p, s):
    """Yields all the positions of the pattern p in the string s.
    :param p: sub str
    :param s: father str
    :return (start position, end position)
    """
    i = s.find(p)
    while i != -1:
        yield (i, i + len(p) - 1)
        i = s.find(p, i + 1)


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


def merge_label(s, content):
    """将文本相同的数据标签合并
    """
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
    return tmp


def src2json():
    """scr data to json file
    """
    # get train data
    result = []
    with open(SRC_DATA_DIR / 'event_entity_train_data_label.csv', 'r', encoding='utf-8') as f, \
            open(DATA_DIR / 'train.data', 'w', encoding='utf-8') as f_train, \
            open(DATA_DIR / 'val.data', 'w', encoding='utf-8') as f_val:
        # get src data
        content = [line.strip().split('\t') for line in f]

        logging.info('Merge label...')
        # 多进程
        with Pool() as p:
            merge_label_func = functools.partial(merge_label, content=content)
            # 合并文本相同的数据
            # content = merge_label(content)
            content = p.map(func=merge_label_func, iterable=content)
        logging.info('-done')

        logging.info('Write train and val set to json file...')
        # write to json
        for idx, c in enumerate(content):
            if c[2][0] != 'NaN' and c[3][0] != 'NaN':
                # 清洗数据
                text = filter_chars(c[1])
                # 抽取实体位置
                # entities = findall(p=c[3], s=text)
                result.append({
                    'id': c[0],
                    'context': text,
                    'type': [tp for tp in c[2]],
                    # 每个类别对应一个实体列表，每个实体对应一个位置列表
                    'entity': [[loc for entity in entity_type for loc in list(findall(p=filter_chars(entity), s=text))]
                               for entity_type in c[3]]
                })
        # shuffle
        random.seed(2020)
        random.shuffle(result)

        print(f'get {len(result[:-900])} train samples.')
        json.dump(result[:-900], f_train, indent=4, ensure_ascii=False)
        print(f'get {len(result[-900:])} val samples.')
        json.dump(result[-900:], f_val, indent=4, ensure_ascii=False)

    # get test data
    result = []
    with open(SRC_DATA_DIR / 'event_entity_dev_data.csv', 'r', encoding='utf-8') as f, \
            open(DATA_DIR / 'test.data', 'w', encoding='utf-8') as f_test:
        # get src data
        content = [line.strip().split('\t') for line in f]
        # write to json
        for idx, c in enumerate(content):
            # 清洗数据
            text = filter_chars(c[1])
            result.append({
                'id': c[0],
                'context': text,
                'type': [],
                'entity': []
            })

        print(f'get {len(result)} test samples.')
        json.dump(result, f_test, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    set_logger(save=False)
    # src data to json
    src2json()
