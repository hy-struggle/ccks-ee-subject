#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mrc result to bio"""


def pointer2bio(start_labels, end_labels, ne_cate):
    """convert (begin, end, span) label to bio label. for single sample.
    :return: bio_labels List[str]: 实体序列（单样本）
    """
    # init
    bio_labels = len(start_labels) * ["O"]

    # 取出start idx和end idx
    start_labels = [idx for idx, tmp in enumerate(start_labels) if tmp != 0]
    end_labels = [idx for idx, tmp in enumerate(end_labels) if tmp != 0]

    # 打start标
    for start_item in start_labels:
        bio_labels[start_item] = "B-{}".format(ne_cate)

    # 打I标
    for tmp_start in start_labels:
        # 取出在start position后的end position
        tmp_end = [tmp for tmp in end_labels if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            # 取出距离最近的end
            tmp_end = min(tmp_end)
        # 如果匹配则标记为实体
        if tmp_start != tmp_end:
            for i in range(tmp_start + 1, tmp_end + 1):
                bio_labels[i] = "I-{}".format(ne_cate)
        # TODO:忽略单字实体
        else:
            bio_labels[tmp_end] = "O"

    return bio_labels
