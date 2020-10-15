#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""postprocess"""
import json
import argparse
import pandas as pd

from utils import IO2STR
from utils import Params
from metrics import get_entities

# 设定参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='val', help="mode")
parser.add_argument('--ex_index', type=int, default=1, help="实验名称索引")
args = parser.parse_args()


def apply_fn(group):
    result = []
    # 获取该组的所有实体
    for tags, s2o in zip(group.tags, group.split_to_ori):
        entities = get_entities(eval(tags))
        for entity in entities:
            result.append((entity[0], eval(s2o)[entity[1]], eval(s2o)[entity[2]]))
    return result


def postprocess(params, mode):
    # get df
    pre_df = pd.read_csv(params.params_path / f'{mode}_tags_pre.csv', encoding='utf-8')
    pre_df = pd.DataFrame(pre_df.groupby('example_id').apply(apply_fn), columns=['entities']).reset_index()
    print(len(pre_df))
    # get text
    with open(params.data_dir / f'{args.mode}.data', 'r', encoding='utf-8') as f:
        text_data = json.load(f)
        sentences = [list(sample['context'].strip()) for sample in text_data]
        print(len(sentences))

    # 预测结果
    result = []
    for idx, entities in enumerate(pre_df['entities']):
        sample_list = []
        for entity in entities:
            label_type = IO2STR[entity[0]]
            start_ind = entity[1]
            end_ind = entity[2]
            en = sentences[idx][start_ind:end_ind + 1]
            sample_list.append((label_type, ''.join(en)))
        result.append(sample_list)

    if mode == 'test':
        # 获取文本id
        with open(params.data_dir / f'test.data', 'r', encoding='utf-8') as f:
            text_data = json.load(f)
            id_list = [sample['id'] for sample in text_data]
    else:
        id_list = list(range(len(result)))

    # 写submit
    with open(params.params_path / f'submit_{mode}.csv', 'w', encoding='utf-8') as f_sub:
        # 获取每条文本的预测结果
        for re_sample, idx in zip(result, id_list):
            # 获取单个预测结果
            for re in set(re_sample):
                # TODO: 规则去除单个实体和过长实体
                # if len(re[1]) != 1 and len(re[1]) < 20:
                    f_sub.write(f'{idx}\t{re[0]}\t{re[1]}\n')


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(ex_index=args.ex_index)
    postprocess(params, mode=args.mode)
