# /usr/bin/env python
# coding=utf-8
"""dataloader"""

import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from dataloader_utils import read_examples, convert_examples_to_features


class CustomDataLoader(object):
    """dataloader
    """

    def __init__(self, params):
        """
        :param data_processor: get data_src examples.
        :param mode: dataloader mode. 'train' or 'test'.
        """
        self.params = params

        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size

        self.data_dir = params.data_dir
        self.max_seq_length = params.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=str(params.bert_model_dir),
                                                       do_lower_case=True,
                                                       never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
        # 保存数据(Bool)
        self.data_cache = params.data_cache

    def convert_examples_to_features(self, data_sign):
        """convert to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        :return: features (List[InputFeatures]):
        """
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))

        # 数据保存路径
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        # 读取数据
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:
            if data_sign == "train":
                examples = read_examples(os.path.join(self.data_dir, "train.data"))
            elif data_sign == "val":
                examples = read_examples(os.path.join(self.data_dir, "val.data"))
            elif data_sign == "test":
                examples = read_examples(os.path.join(self.data_dir, "test.data"))
            else:
                raise ValueError("please notice that the data can only be train/val/test !!")
            # 生成数据
            features = convert_examples_to_features(self.params, examples, self.tokenizer)
            # save data
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign="train"):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        :return:
        """
        # InputExamples to InputFeatures
        features = self.convert_examples_to_features(data_sign=data_sign)

        # convert to tensor
        print('Convert to Tensor...')
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        cate = torch.tensor([f.cate for f in features], dtype=torch.long)
        dataset = TensorDataset(input_ids, input_mask, segment_ids, cate)

        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)  # RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size)
        elif data_sign == "test":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size)

        return dataloader


if __name__ == '__main__':
    from utils import Params

    params = Params()
    datalodaer = CustomDataLoader(params)
    f = datalodaer.get_dataloader(data_sign='train')
