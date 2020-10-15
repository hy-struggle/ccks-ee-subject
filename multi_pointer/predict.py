#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""do inference"""
import argparse
import logging
import os
import random
from tqdm import tqdm

import torch
import pandas as pd

import utils
from evaluate import pointer2bio
from dataloader import NERDataLoader

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=int, default=1, help="实验名称索引")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--restore_file', default='best', required=False,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--mode', default='test', help="'val' or 'test'")


def predict(model, test_dataloader, params, mode):
    """预测并将结果输出至文件
    :param mode: 'val' or 'test'
    """
    model.eval()
    # init
    pre_result = pd.DataFrame()

    # idx to label
    cate_idx2label = {idx: int(idx + 1) for idx, _ in enumerate(params.label_list)}

    # get data
    for batch in tqdm(test_dataloader, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, start_pos, end_pos, split_to_ori, example_ids = batch
        # inference
        with torch.no_grad():
            start_logits, end_logits = model(input_ids, attention_mask=input_mask)

        # predict label
        start_label = start_logits.detach().cpu().numpy().transpose((0, 2, 1)).tolist()  # (bs, tag_size, seq_len)
        end_label = end_logits.detach().cpu().numpy().transpose((0, 2, 1)).tolist()
        # mask
        input_mask = input_mask.to("cpu").detach().numpy().tolist()
        split_to_ori = split_to_ori.to('cpu').numpy().tolist()  # (bs, max_len)
        example_ids = example_ids.to('cpu').numpy().tolist()  # (bs,)

        # get result
        for start_p_s, end_p_s, input_mask_s, s_t_o, example_id in zip(start_label, end_label, input_mask,
                                                                       split_to_ori, example_ids):
            # 有效长度
            act_len = sum(input_mask_s)
            for idx, (start_p, end_p) in enumerate(zip(start_p_s, end_p_s)):
                pre_bio_labels = pointer2bio(start_p[:act_len], end_p[:act_len],
                                             ne_cate=cate_idx2label[idx])
                # append to df
                pre_result = pre_result.append({
                    'example_id': int(example_id),
                    'tags': pre_bio_labels,
                    'split_to_ori': s_t_o[:act_len]
                }, ignore_index=True)

    pre_result.to_csv(path_or_buf=params.params_path / f'{mode}_tags_pre.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(args.ex_index)

    # 设置模型使用的gpu
    torch.cuda.set_device(args.device_id)
    # 查看现在使用的设备
    print('current device:', torch.cuda.current_device())
    # 预测验证集还是测试集
    mode = args.mode
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = NERDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    # Reload weights from the saved file
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode)
    logging.info('-done')

    logging.info("Starting prediction...")
    # Create the input data pipeline
    predict(model, loader, params, mode)
    logging.info('-done')
