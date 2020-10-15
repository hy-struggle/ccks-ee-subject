#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""Downstream task model"""

import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, RobertaModel
from utils import initial_parameter


class BertSequenceClassifier(BertPreTrainedModel):
    def __init__(self, config, params):
        super(BertSequenceClassifier, self).__init__(config)
        # pretrain model layer
        self.bert = RobertaModel(config)

        # 动态权重
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)),
                                       requires_grad=True)

        self.tag_size = len(params.tag_list)
        self.classifier = nn.Linear(config.hidden_size, self.tag_size)

        self.init_weights()
        self.init_param()

    def init_param(self):
        initial_parameter(self.classifier)
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        """
        获取动态权重融合后的bert output(num_layer维度)
        :param outputs: origin bert output
        :return: sequence_output: 融合后的bert encoder output. (batch_size, seq_len, hidden_size[embedding_dim])
        """
        hidden_stack = torch.stack(outputs[2][1:], dim=0)  # (bert_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cate=None):
        # pretrain model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        # BERT融合
        sequence_output = self.get_dym_layer(outputs)  # (batch_size, seq_len, hidden_size[embedding_dim])
        # sequence_output = outputs[0]  # batch x seq_len x hidden
        batch_size, seq_len, hid_size = sequence_output.size()
        # get [CLS] vector
        sequence_output = sequence_output[:, 0, :]

        logits = self.classifier(sequence_output)  # (bs, tag_size)

        # train
        if cate is not None:
            # s & e loss
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(batch_size, -1), cate.view(batch_size, -1).float())
            return loss
        # inference
        else:
            pre = torch.sigmoid(logits)
            return pre


if __name__ == '__main__':
    from transformers import RobertaConfig
    import utils

    params = utils.Params()
    # Prepare model
    config = RobertaConfig.from_pretrained(str(params.bert_model_dir / 'config.json'), output_hidden_states=True)
    model = BertSequenceClassifier.from_pretrained(str(params.bert_model_dir),
                                                   config=config, params=params)

    for n, _ in model.named_parameters():
        print(n)
