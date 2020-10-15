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
                 query_item,
                 context_item,
                 start_position=None,
                 end_position=None,
                 ne_cate=None):
        self.query_item = query_item
        self.context_item = context_item
        self.start_position = start_position
        self.end_position = end_position
        self.ne_cate = ne_cate


class InputFeatures(object):
    """
    Desc:
        a single set of features of data_src
    Args:
        start_position: start position is a list of symbol
        end_position: end position is a list of symbol
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 ne_cate,
                 start_position=None,
                 end_position=None,
                 ):
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.ne_cate = ne_cate
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


def read_mrc_ner_examples(input_file):
    """read MRC-NER data_src to InputExamples
    :return examples (List[InputExample]):
    """
    # read json file
    with open(input_file, "r", encoding='utf-8') as f:
        input_data = json.load(f)
    # get InputExample class
    examples = []
    for entry in input_data:
        query_item = entry["query"]
        context_item = entry["context"]
        start_position = entry["start_position"]
        end_position = entry["end_position"]
        ne_cate = entry["entity_label"]

        example = InputExample(query_item=query_item,
                               context_item=context_item,
                               start_position=start_position,
                               end_position=end_position,
                               ne_cate=ne_cate)
        examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def convert_examples_to_features(params, examples, tokenizer, pad_sign=True):
    """convert src data_src to features.
    :param examples (List[InputExamples]): data_src examples.
    :param pad_sign: 是否补零
    :return: features (List[InputFeatures])
    """
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.label_list)}
    features = []

    for (example_idx, example) in enumerate(examples):
        # tokenize query
        query_tokens = tokenizer.tokenize(example.query_item)
        # List[str]
        context_doc = whitespace_tokenize(example.context_item)
        # context max len
        max_tokens_for_doc = params.max_seq_length - len(query_tokens) - 3

        # init
        doc_start_pos = [0] * len(context_doc)
        doc_end_pos = [0] * len(context_doc)

        # 获取文本tokens
        # 标签不为空的样本
        if len(example.start_position) != 0 and len(example.end_position) != 0:
            # get gold label
            for start_item, end_item in zip(example.start_position, example.end_position):
                doc_start_pos[start_item] = 1
                doc_end_pos[end_item] = 1

        # get context_tokens
        context_doc_tokens = []
        for token in context_doc:
            # tokenize
            tmp_subword_lst = tokenizer.tokenize(token)
            if len(tmp_subword_lst) == 1:
                context_doc_tokens.extend(tmp_subword_lst)  # context len
            else:
                raise ValueError("Please check the result of tokenizer!!!")

        # sanity check
        assert len(context_doc_tokens) == len(doc_start_pos)
        assert len(context_doc_tokens) == len(doc_end_pos)
        assert len(doc_start_pos) == len(doc_end_pos)

        # cut off
        if len(context_doc_tokens) > max_tokens_for_doc:
            context_doc_tokens = context_doc_tokens[: max_tokens_for_doc]
            doc_start_pos = doc_start_pos[: max_tokens_for_doc]
            doc_end_pos = doc_end_pos[: max_tokens_for_doc]

        # input_mask:
        #   the mask has 1 for real tokens and 0 for padding tokens.
        #   only real tokens are attended to.
        # segment_ids:
        #   segment token indices to indicate first and second portions of the inputs.
        input_tokens = []
        segment_ids = []
        input_mask = []
        start_pos = []
        end_pos = []

        input_tokens.append("[CLS]")
        segment_ids.append(0)
        start_pos.append(0)
        end_pos.append(0)
        input_mask.append(1)

        # query
        input_tokens.extend(query_tokens)
        segment_ids.extend([0] * len(query_tokens))
        start_pos.extend([0] * len(query_tokens))
        end_pos.extend([0] * len(query_tokens))
        input_mask.extend([1] * len(query_tokens))

        # sep
        input_tokens.append("[SEP]")
        segment_ids.append(0)
        start_pos.append(0)
        end_pos.append(0)
        input_mask.append(1)

        # context
        input_tokens.extend(context_doc_tokens)
        segment_ids.extend([1] * len(context_doc_tokens))
        start_pos.extend(doc_start_pos)
        end_pos.extend(doc_end_pos)
        input_mask.extend([1] * len(context_doc_tokens))

        # sep
        input_tokens.append("[SEP]")
        segment_ids.append(1)
        start_pos.append(0)
        end_pos.append(0)
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
            start_pos += padding
            end_pos += padding

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_pos,
                end_position=end_pos,
                ne_cate=tag2idx[example.ne_cate]
            ))

    return features
