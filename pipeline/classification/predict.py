#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""do inference"""
import argparse
import logging
import os
import random
from tqdm import tqdm

import torch
import numpy as np

import utils
from utils import IO2STR, STR2IO
from transformers import RobertaConfig
from dataloader import CustomDataLoader
from model import BertSequenceClassifier

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--seed', type=int, default=2333, help="random seed for initialization")
parser.add_argument('--restore_file', default=None, required=False,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--mode', default='test', help="'val' or 'test'")


def predict(model, dataloader, params, mode):
    """预测并将结果输出至文件
    :param mode: 'val' or 'test'
    """
    model.eval()
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.tag_list)}
    y_pred = []
    # get data
    for batch in tqdm(dataloader, unit='Batch'):
        # fetch the next training batch
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_id, _ = batch
        # inference
        with torch.no_grad():
            # (bs, tag_size)
            cls_pre = model(input_ids=input_ids,
                            attention_mask=input_mask, token_type_ids=segment_id)
        # predict
        pred = cls_pre.detach().cpu().numpy()
        # TODO: 规则获取pred
        pred_threshold = np.where(pred > params.threshold, 1, 0)
        for idx, p in enumerate(pred_threshold):
            # 如果有NaN类
            if p[tag2idx[STR2IO['NaN']]] == 1:
                # 如果NaN类概率最大
                if np.argmax(pred[idx]) == tag2idx[STR2IO['NaN']]:
                    pred_threshold[idx] = 0
                    pred_threshold[idx][tag2idx[STR2IO['NaN']]] = 1
                else:
                    pred_threshold[idx][tag2idx[STR2IO['NaN']]] = 0
            # 如果没有类别，选最大的
            if 1 not in p:
                pred_threshold[idx][np.argmax(pred[idx])] = 1

        y_pred.append(pred_threshold)

    # 将预测结果写入文件
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    idx2tag = {idx: tag for idx, tag in enumerate(params.tag_list)}
    with open(params.params_path / f'{mode}_pre.data', 'w', encoding='utf-8') as f:
        for sample in y_pred:
            tags = []
            for idx, p in enumerate(sample):
                if p == 1:
                    tags.append(IO2STR[idx2tag[idx]])
            f.write(",".join(tags) + '\n')


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
    dataloader = CustomDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    config = RobertaConfig.from_pretrained(params.bert_model_dir / 'config.json', output_hidden_states=True)
    model = BertSequenceClassifier.from_pretrained(params.bert_model_dir,
                                                   config=config, params=params)
    model.to(params.device)

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'), model)
    logging.info('- done.')

    # Create the input data pipeline
    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode)
    logging.info("- done.")

    logging.info("Starting prediction...")
    predict(model, loader, params, mode)
    logging.info('- done.')
