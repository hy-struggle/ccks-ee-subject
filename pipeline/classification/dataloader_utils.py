#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataloader.py utils"""

import json


def whitespace_tokenize(text):
    """
    Desc:
        runs basic whitespace cleaning and splitting on a piece of text.
    """
    text = text.strip()
    # 内容为空则返回空列表
    if not text:
        return []
    tokens = list(text)
    return tokens


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self,
                 context_item,
                 cate=None):
        self.context_item = context_item
        self.cate = cate


class InputFeatures(object):
    """
    Desc:
        a single set of features of data_src
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cate=None,
                 ):
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.cate = cate
        self.segment_ids = segment_ids


def read_examples(input_file):
    """read data_src to InputExamples
    :return examples (List[InputExample]):
    """
    # read json file
    with open(input_file, "r", encoding='utf-8') as f:
        input_data = json.load(f)
    # get InputExample class
    examples = []
    for entry in input_data:
        context_item = entry["context"]
        cate = entry["entity_label"]

        example = InputExample(context_item=context_item,
                               cate=cate)
        examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def convert_examples_to_features(params, examples, tokenizer, pad_sign=True):
    """convert examples to features.
    :param examples (List[InputExamples]):
    :param pad_sign: 是否补零
    :return: features (List[InputFeatures])
    """
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.tag_list)}
    features = []

    for (example_idx, example) in enumerate(examples):
        # List[str]
        context_doc = whitespace_tokenize(example.context_item)

        # 获取文本tokens
        # 标签不为空的样本
        # get context_tokens
        context_doc_tokens = []
        for token in context_doc:
            # tokenize
            tmp_subword_lst = tokenizer.tokenize(token)
            if len(tmp_subword_lst) == 1:
                context_doc_tokens.extend(tmp_subword_lst)  # context len
            else:
                raise ValueError("Please check the result of tokenizer!!!")

        # cut off
        if len(context_doc_tokens) > params.max_seq_length - 2:
            context_doc_tokens = context_doc_tokens[: params.max_seq_length - 2]

        # input_mask:
        #   the mask has 1 for real tokens and 0 for padding tokens.
        #   only real tokens are attended to.
        # segment_ids:
        #   segment token indices to indicate first and second portions of the inputs.
        input_tokens = []
        segment_ids = []
        input_mask = []

        input_tokens.append("[CLS]")
        segment_ids.append(0)
        input_mask.append(1)

        # context
        input_tokens.extend(context_doc_tokens)
        segment_ids.extend([0] * len(context_doc_tokens))
        input_mask.extend([1] * len(context_doc_tokens))

        # sep
        input_tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)

        # token to id
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

        # zero-padding up to the sequence length
        if len(input_ids) < params.max_seq_length and pad_sign:
            # 补零
            padding = [0] * (params.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

        # get category
        cate = [0] * len(tag2idx)
        for c in example.cate:
            cate[tag2idx[c]] = 1

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                cate=cate
            ))

    return features
