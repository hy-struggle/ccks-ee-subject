#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""postprocess"""
import json
import argparse

from utils import IO2STR
from utils import Params

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--mode', type=str, default='test', help='test or val')


def postprocess(params, mode):
    """分析文本形式结果
    """
    # get text
    with open(params.data_dir / f'{mode}.data', 'r', encoding='utf-8') as f_src, \
            open(params.params_path / f'{mode}_pre.data', 'r', encoding='utf-8') as f_pre:
        data_src = json.load(f_src)
        result = []
        for d_s, d_p in zip(data_src, f_pre):
            idx = d_s['id']
            text = d_s['context']
            tags = d_p.strip().split(',')
            result.append({
                'id': idx,
                'context': text,
                'entity_label': tags
            })
    with open(params.params_path / f'{mode}_result.data', 'w', encoding='utf-8') as f_re:
        json.dump(result, f_re, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params()
    postprocess(params, mode=args.mode)
