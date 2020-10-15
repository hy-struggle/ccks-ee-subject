#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate"""
import logging
from tqdm import tqdm

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

import utils
from utils import STR2IO


def evaluate(args, model, eval_dataloader, params):
    model.eval()
    # 记录平均损失
    loss_avg = utils.RunningAverage()
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.tag_list)}
    # init
    y_true = []
    y_pred = []
    # get data
    for batch in tqdm(eval_dataloader, unit='Batch'):
        # fetch the next training batch
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_id, cate = batch
        # inference
        with torch.no_grad():
            # get loss
            loss = model(input_ids, token_type_ids=segment_id, attention_mask=input_mask,
                         cate=cate)
            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu.
            # update the average loss
            loss_avg.update(loss.item())

            # (bs, tag_size)
            cls_pre = model(input_ids=input_ids,
                            attention_mask=input_mask, token_type_ids=segment_id)
        # gold label
        gold = cate.to('cpu').numpy()  # (bs, tag_size)
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

        y_true.append(gold)
        y_pred.append(pred_threshold)

    # metrics
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    # f1, acc
    metrics = {'loss': loss_avg(), 'f1': f1, 'acc': acc}
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format('Val') + metrics_str)
    return metrics


if __name__ == '__main__':
    from utils import Params
    from transformers import RobertaConfig
    from model import BertSequenceClassifier
    from dataloader import CustomDataLoader
    import argparse

    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', action='store_true', help="是否多GPU")
    args = parser.parse_args()
    params = Params()
    # Prepare model
    logging.info('Init pre-train model...')
    config = RobertaConfig.from_pretrained(params.bert_model_dir / 'config.json', output_hidden_states=True)
    model = BertSequenceClassifier.from_pretrained(params.bert_model_dir,
                                                   config=config, params=params)
    model.to(params.device)
    # Load training data and val data
    dataloader = CustomDataLoader(params)
    val_loader = dataloader.get_dataloader(data_sign='val')
    evaluate(args, model, val_loader, params)
