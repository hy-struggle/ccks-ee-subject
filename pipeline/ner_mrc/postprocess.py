#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""postprocess"""
import json

from utils import IO2STR
from utils import Params
from metrics import get_entities


def postprocess(params):
    """分析文本形式结果
    """
    # get text
    with open(params.data_dir / f'test.bio', 'r', encoding='utf-8') as f:
        sentences = [line.strip().split('\t')[-1].split(' ') for line in f]

    # 预测标签
    with open(params.data_dir / f'test_tags_pre.txt', 'r') as f:
        result = []
        for idx, line in enumerate(f):
            sample_list = []
            # get BIO-tag
            entities = get_entities(line.strip().split(' '))
            for entity in entities:
                label_type = IO2STR[entity[0]]
                start_ind = entity[1]
                end_ind = entity[2]
                # get en from sentence
                en = sentences[idx][start_ind:end_ind + 1]
                sample_list.append((label_type, ''.join(en)))

            result.append(sample_list)
    return result


def get_submit(params):
    # 获取测试集结果
    re_pre = postprocess(params)
    # 获取文本id
    with open(params.data_dir / f'test.bio', 'r', encoding='utf-8') as f:
        id_list = [line.strip().split('\t')[0] for line in f]
    # 写submit
    with open(params.params_path / 'submit.csv', 'w', encoding='utf-8') as f_sub:
        # 获取每条文本的预测结果
        for re_sample, idx in zip(re_pre, id_list):
            # 获取单个预测结果
            for re in set(re_sample):
                # TODO: 单字实体和字数大于20的实体不要
                if len(re[1]) != 1 and len(re[1]) <= 20:
                    f_sub.write(f'{idx}\t{re[0]}\t{re[1]}\n')


if __name__ == '__main__':
    params = Params()
    get_submit(params)