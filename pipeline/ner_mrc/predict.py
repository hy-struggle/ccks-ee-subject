#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""do inference"""
import argparse
import logging
import os
import random

import torch

import utils
from utils import IO2QUERY
from transformers import RobertaConfig
from metrics_utils import pointer2bio
from dataloader import MRCNERDataLoader
from model import BertQueryNER

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--seed', type=int, default=2333, help="random seed for initialization")
parser.add_argument('--restore_file', default=None, required=False,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--mode', default='test', help="'val' or 'test'")


def predict(model, test_dataloader, params, mode):
    """预测并将结果输出至文件
    :param mode: 'val' or 'test'
    """
    model.eval()
    # init
    pre_result = []

    # idx to label
    cate_idx2label = {idx: value for idx, value in enumerate(params.label_list)}

    # get data
    for input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate in test_dataloader:
        # to device
        input_ids = input_ids.to(params.device)
        input_mask = input_mask.to(params.device)
        segment_ids = segment_ids.to(params.device)

        # inference
        with torch.no_grad():
            start_logits, end_logits = model(input_ids, segment_ids, input_mask)

        # predict label
        start_label = start_logits.detach().cpu().numpy().tolist()
        end_label = end_logits.detach().cpu().numpy().tolist()
        # mask
        input_mask = input_mask.to("cpu").detach().numpy().tolist()
        ner_cate = ner_cate.to("cpu").numpy().tolist()

        # get result
        for start_p, end_p, input_mask_s, ner_cate_s in zip(start_label, end_label,
                                                            input_mask, ner_cate):
            ner_cate_str = cate_idx2label[ner_cate_s]
            # 问题长度
            q_len = len(IO2QUERY[ner_cate_str])
            # 有效长度
            act_len = sum(input_mask_s[q_len + 2:-1])
            # 转换为BIO标注
            pre_bio_labels = pointer2bio(start_p[q_len+2:q_len+2+act_len],
                                         end_p[q_len+2:q_len+2+act_len],
                                         ne_cate=ner_cate_str)

            pre_result.append(pre_bio_labels)

    # write to file
    with open(params.data_dir / f'{mode}_tags_pre.txt', 'w', encoding='utf-8') as file_tags:
        for tag in pre_result:
            # 真实标签
            file_tags.write('{}\n'.format(' '.join(tag)))


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params()
    # Set the logger
    utils.set_logger(save=False)
    # 预测验证集还是测试集
    mode = args.mode
    # 设置模型使用的gpu
    torch.cuda.set_device(7)
    # 查看现在使用的设备
    logging.info('current device:{}'.format(torch.cuda.current_device()))

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # get dataloader
    dataloader = MRCNERDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    config = RobertaConfig.from_pretrained(params.bert_model_dir / 'config.json', output_hidden_states=True)
    model = BertQueryNER.from_pretrained(params.bert_model_dir,
                                         config=config, params=params)
    model.to(params.device)

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'), model)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode)
    logging.info("- done.")

    logging.info("Starting prediction...")
    predict(model, loader, params, mode)
    logging.info('- done.')
