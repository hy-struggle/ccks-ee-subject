#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""utils"""
import json
import os
from pathlib import Path
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.init as init

STR2IO = {'不能履职': '1', '资产负面': '2', '业务资产重组': '3', '歇业停业': '4',
          '实际控制人变更': '5', '资金紧张': '6', '业绩下滑': '7', '履行连带担保责任': '8',
          '涉嫌欺诈': '9', '涉嫌传销': '10', '提现困难': '11', '实控人股东变更': '12', '债务违约': '13',
          '信批违规': '14', '投诉维权': '15', '高管负面': '16', '实际控制人涉诉仲裁': '17', '财务信息造假': '18',
          '交易违规': '19', '股票转让-股权受让': '20', '商业信息泄露': '21', '评级调整': '22', '失联跑路': '23',
          '重组失败': '24', '资金账户风险': '25', '债务重组': '26', '涉嫌非法集资': '27', '财务造假': '28',
          '涉嫌违法': '29'}
IO2STR = {v: k for k, v in STR2IO.items()}
IO2QUERY = {'1': '高管逝世，去世，违纪，违规，失联，死亡，被捕',
            '2': '资金流向不明，被冻结，被查封，资金链断裂',
            '3': '资产重组，并购，合并，被收购',
            '4': '停业，被注销，闭店，吊销执照，歇业整顿，停产',
            '5': '实际控制人变更，出售股权，股权变更',
            '6': '偿债能力差，资金压力，资金紧张，资金缺口',
            '7': '净利下降，业绩下坡，盈利下滑',
            '8': '连带担保责任，连带清偿责任，违规担保',
            '9': '涉嫌欺诈，犯罪，虚假宣传，诈骗，非法交易，集资',
            '10': '涉嫌传销，直销，传销模式',  # 9的子集
            '11': '无法归还，提现失败，提现困难',
            '12': '控股股东变更，控制人变更，实控人变更',  # 和5差不多
            '13': '违约债券，债务违约，逾期债务，未能偿还',
            '14': '信息披露违规，披露推迟，未披露，信披违规',
            '15': '举报，投诉，信访，侵权，诈骗',
            '16': '高管被拘留，羁押，卷款，批准逮捕，判刑，被抓',  # 1的子集
            '17': '实控人，董事长被查，涉及诉讼，股份被冻结',
            '18': '财务造假，披露虚假信息，利润造假',
            '19': '内幕交易，违规交易，违规买卖，短线交易',
            '20': '转让股份，出售股权，入股，收购股份',
            '21': '信息泄露，数据泄露，商业机密，内幕信息泄露',
            '22': '评级下调，调整，调降',
            '23': '卷款跑路，人去楼空，停止运营，失联',
            '24': '重组被否，重组告吹，重组失败，未通过，被终止',
            '25': '套取资金，财产冻结，存款丢失，被盗',
            '26': '债务问题，资产重组，债务重组',  # 3的子集
            '27': '非法集资，非法吸收公共存款，非吸，自融',
            '28': '财务造假，欺诈上市，数据造假，虚增，粉饰业绩',
            '29': '涉嫌违法，立案侦查，法院处罚，刑事，起诉'
            }


class Params:
    """参数定义
    """

    def __init__(self):
        # 根路径
        self.root_path = Path(os.path.abspath(os.path.dirname(__file__)))
        self.data_dir = self.root_path / 'data'
        self.params_path = self.root_path / 'experiments'
        self.bert_model_dir = self.root_path.parent.parent.parent / 'pre_model_roberta_base'
        self.model_dir = self.root_path / 'model'

        # 读取保存的data
        self.data_cache = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()

        self.train_batch_size = 100
        self.val_batch_size = 100
        self.test_batch_size = 128

        # patience策略
        self.patience = 0.001
        self.patience_num = 3
        self.min_epoch_num = 3

        # 标签列表
        self.label_list = list(STR2IO.values())
        self.max_seq_length = 200

        self.dropout = 0.3
        self.weight_decay_rate = 0.05
        self.fin_tuning_lr = 5e-5
        self.downstream_lr = self.fin_tuning_lr * 10
        # 梯度截断
        self.clip_grad = 2
        self.warmup_prop = 0.1
        self.gradient_accumulation_steps = 2

        # 下游参数
        self.weight_start = 1.
        self.weight_end = 1.

    def get(self):
        """Gives dict-like access to Params instance by `params.show['learning_rate']"""
        return self.__dict__

    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """保存配置到json文件
        """
        params = {}
        with open(json_path, 'w') as f:
            for k, v in self.__dict__.items():
                if isinstance(v, (str, int, float, bool)):
                    params[k] = v
            json.dump(params, f, indent=4)


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    记录平均损失

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    # 如果是最好的checkpoint则以best为文件名保存
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def set_logger(save, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if save:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def initial_parameter(net, initial_method=None):
    r"""A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model or a List of Pytorch model
    :param str initial_method: one of the following initializations.

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        # classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            if len(m.weight.size()) > 1:
                init_method(m.weight.data)
            else:
                init.normal_(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    if isinstance(net, list):
        for n in net:
            n.apply(weights_init)
    else:
        net.apply(weights_init)