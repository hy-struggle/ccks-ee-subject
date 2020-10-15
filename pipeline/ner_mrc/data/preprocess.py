# /usr/bin/env python
# coding=utf-8
"""preprocess"""
from pathlib import Path
import json
import re
import random
import logging

from utils import STR2IO, IO2QUERY, set_logger

SRC_DATA_DIR = Path('../ccks 4_1 Data')
DATA_DIR = Path('./')


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
    with open(DATA_DIR / f'{mode}.data', 'w', encoding='utf-8') as f, \
            open(DATA_DIR / f'{mode}.bio', 'w', encoding='utf-8') as f_bio:
        result = []
        # write to json
        for idx, c in enumerate(content):
            if c[2][0] != 'NaN' and c[3][0] != 'NaN':
                # 清洗数据
                text = filter_chars(c[1])
                # 写BIO文件
                # f_bio.write('{}\n'.format(' '.join(text)))
                # 抽取实体位置
                for tp_id, tp in enumerate(c[2]):
                    result.append({
                        'id': c[0],
                        'context': text,
                        'entity_label': STR2IO[tp],
                        'query': IO2QUERY[STR2IO[tp]],
                        'start_position': [loc[0] for entity in c[3][tp_id]
                                           for loc in list(findall(p=entity, s=text))],
                        'end_position': [loc[1] for entity in c[3][tp_id]
                                         for loc in list(findall(p=entity, s=text))]
                    })
        print(f'get {len(result)} {mode} samples.')
        json.dump(result, f, indent=4, ensure_ascii=False)


def get_test_data():
    """获取测试集
    """
    result = []
    # 获取cls任务结果
    with open(SRC_DATA_DIR / 'test_cls.data', 'r', encoding='utf-8') as f_cls:
        data_cls = json.load(f_cls)

    with open(DATA_DIR / 'test.data', 'w', encoding='utf-8') as f_json, \
            open(DATA_DIR / 'test.bio', 'w', encoding='utf-8') as f_bio:
        for d in data_cls:
            for cls in d['entity_label']:
                # 去掉NaN样本
                if cls != 'NaN':
                    f_bio.write(f"{d['id']}\t{' '.join(d['context'])}\n")
                    result.append({
                        'id': d['id'],
                        'context': d['context'],
                        'entity_label': STR2IO[cls],
                        'query': IO2QUERY[STR2IO[cls]],
                        'start_position': [],
                        'end_position': []
                    })
        print(f'get {len(result)} test samples.')
        json.dump(result, f_json, indent=4, ensure_ascii=False)


def get_data_mrc():
    """获取mrc格式数据集
    """
    with open(SRC_DATA_DIR / 'event_entity_train_data_label.csv', 'r', encoding='utf-8') as f:
        # get src data
        content_tmp = [line.strip().split('\t') for line in f]
        # 去掉NaN标注
        content = [c for c in content_tmp if c[2] != 'NaN' and c[3] != 'NaN']
        logging.info('Merge label...')
        # 合并文本相同的数据
        content = merge_label(content)
        logging.info('-done')

        # shuffle&split
        random.seed(2020)
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
    # get_data_mrc()
    get_test_data()