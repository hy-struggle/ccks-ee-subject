#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""model"""

import torch
import torch.nn as nn
import numpy as np

from NEZHA.model_NEZHA import BertPreTrainedModel, NEZHAModel
from utils import initial_parameter


class MultiLossLayer(nn.Module):
    def __init__(self, num_loss):
        super(MultiLossLayer, self).__init__()
        # sigmas^2
        self._sigmas_sq = torch.empty(num_loss)  # (num_loss,)
        # uniform init
        self._sigmas_sq = nn.Parameter(nn.init.uniform_(self._sigmas_sq, a=0.2, b=1.0), requires_grad=True)

    def get_loss(self, loss_set):
        """
        Args:
            loss_set: (num_loss,) multi-task loss
        """
        # 1/2σ^2
        factor = torch.div(1.0, torch.mul(2.0, self._sigmas_sq))  # (num_loss,)
        # loss part
        loss_part = torch.sum(torch.mul(factor, loss_set))  # (num_loss,)
        # regular part
        regular_part = torch.sum(torch.log(self._sigmas_sq))
        loss = loss_part + regular_part
        return loss


class MultiLabelClassifier(nn.Module):
    def __init__(self, seq_len, tag_size, hidden_size, drop_prop):
        """
        句子维度多标签分类
        Args:
            tag_size: 实体类别数
            seq_len: 序列长度
        """
        super(MultiLabelClassifier, self).__init__()
        self.seq_len = seq_len
        # 类别
        self.tag_size = tag_size
        self.drop_prop = drop_prop
        self.dropout = nn.Dropout(self.drop_prop)
        # 获取表征类别的向量
        self.seqvec2tagvec = nn.Linear(seq_len, tag_size)
        # 获取类别
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output):
        """
        Args:
            sequence_output: (bs, seq_len, hidden_size)
        Returns:
            cls_pre: 用于分类的预测结果，(bs, tag_size)
            tag_vec: 用于表征样本类别的向量，(bs, tag_size, hidden_size)
        """
        tag_vec = self.seqvec2tagvec(sequence_output.transpose(1, 2)).transpose(1, 2)  # (bs, tag_size, hidden_size)
        tag_vec = self.dropout(tag_vec)
        cls_pre = self.output(tag_vec).squeeze(-1)  # (bs, tag_size)
        return cls_pre, tag_vec


class BertJointExtractPointer(BertPreTrainedModel):
    def __init__(self, config, params):
        super(BertJointExtractPointer, self).__init__(config)
        # pre-train model layer
        self.bert = NEZHAModel(config)
        self.tag_size = len(params.label_list)

        # use to classify
        self.classifier = MultiLabelClassifier(seq_len=params.max_seq_length, tag_size=self.tag_size,
                                               hidden_size=config.hidden_size, drop_prop=params.dropout)
        # start and end position layer
        # use to extract
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)

        # 动态权重
        self.fusion_layers = params.fusion_layers
        self.dym_weight = nn.Parameter(torch.ones((self.fusion_layers, 1, 1, 1)),
                                       requires_grad=True)

        # loss weight
        self.weight_cls = params.weight_cls
        self.weight_ext = params.weight_ext
        self.threshold = params.pointer_threshold

        self.apply(self.init_bert_weights)
        self.init_param()

    def init_param(self):
        initial_parameter(self.classifier)
        initial_parameter(self.start_outputs)
        initial_parameter(self.end_outputs)
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        """
        获取动态权重融合后的bert output(num_layer维度)
        :param outputs: origin bert output
        :return sequence_output: 融合后的bert encoder output. (batch_size, seq_len, hidden_size[embedding_dim])
        """
        hidden_stack = torch.stack(outputs[0][-self.fusion_layers:],
                                   dim=0)  # (bert_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, cls_labels=None,
                cls_ids=None, start_positions=None, end_positions=None):
        """
        Args:
            cls_labels: (bs, tag_size)
            cls_ids: (bs,)
            start_positions: (batch x max_len)
            end_positions: (batch x max_len)
        Returns:
            cls_re: (one_sample_bs,)
            start_logits: (one_sample_bs, seq_len)
            end_logits: (one_sample_bs, seq_len)
        """
        # pretrain model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_all_encoded_layers=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        # BERT融合
        sequence_output = self.get_dym_layer(outputs)  # (batch_size, seq_len, hidden_size[embedding_dim])
        batch_size, seq_len, hid_size = sequence_output.size()

        loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        # 分类结果
        cls_pres, tag_vecs = self.classifier(sequence_output)  # (bs, tag_size), (bs, tag_size, hidden_size)

        # train
        if cls_ids is not None and start_positions is not None and end_positions is not None:
            # cls loss
            cls_loss = torch.mean(loss_fct(cls_pres, cls_labels.float()))
            # extract loss
            # 获取要融合的类别向量
            # (bs, hidden_size)
            tag_vec = tag_vecs[list(range(batch_size)), cls_ids.squeeze(-1)]
            # 融合类别向量
            sequence_output += tag_vec.unsqueeze(1).expand((batch_size, seq_len, hid_size))
            # get logits
            start_logits = self.start_outputs(sequence_output)  # batch x seq_len x 1
            end_logits = self.end_outputs(sequence_output)  # batch x seq_len x 1
            # s&e loss, (bs, seq_len)
            start_loss = loss_fct(start_logits.view(batch_size, -1), start_positions.view(batch_size, -1).float())
            end_loss = loss_fct(end_logits.view(batch_size, -1), end_positions.view(batch_size, -1).float())
            # mask loss
            extra_loss = torch.sum((start_loss + end_loss) * attention_mask)
            # calculate the average
            extra_loss /= torch.sum(attention_mask)
            # total loss
            total_loss = self.weight_cls * cls_loss + self.weight_ext * extra_loss
            return total_loss
        # inference(bs=1)
        else:
            # 一条样本对应的抽取序列batch
            one_sample_batch = []
            cls_re = np.where(cls_pres[0].detach().cpu().numpy() > self.threshold)[0]  # (one_sample_bs, )

            # 如果没有大于阈值的类别，则当作空样本处理
            if len(cls_re) == 0:
                start_pre = torch.zeros(sequence_output.squeeze(-1).size(), device=sequence_output.device)
                end_pre = torch.zeros(sequence_output.squeeze(-1).size(), device=sequence_output.device)
                cls_re = [np.argmax(cls_pres[0].detach().cpu().numpy())]

                return cls_re, start_pre, end_pre

            for idx in cls_re:
                # 选出类别表征向量
                tag_vec = tag_vecs[0][idx].expand((1, seq_len, hid_size))  # (1, seq_len, hidden_size)
                fusion_vec = sequence_output + tag_vec  # (1, seq_len, hidden_size)
                one_sample_batch.append(fusion_vec)
            one_sample_output = torch.cat(one_sample_batch, dim=0)  # (one_sample_bs, seq_len, hidden_size)

            # get logits
            start_logits = self.start_outputs(one_sample_output).squeeze(-1)  # (one_sample_bs, seq_len)
            end_logits = self.end_outputs(one_sample_output).squeeze(-1)
            # get output
            start_pre = torch.sigmoid(start_logits)  # one_sample_bs x seq_len
            end_pre = torch.sigmoid(end_logits)
            tensor_ones = torch.ones(start_pre.size(), device=start_pre.device)
            tensor_zeros = torch.zeros(start_pre.size(), device=start_pre.device)
            start_pre = torch.where(start_pre > self.threshold, tensor_ones, tensor_zeros)
            end_pre = torch.where(end_pre > self.threshold, tensor_ones, tensor_zeros)  # (one_sample_bs, seq_len)
            return cls_re, start_pre, end_pre
