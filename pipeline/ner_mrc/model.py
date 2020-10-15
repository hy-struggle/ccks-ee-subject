#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""Downstream task model"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, RobertaModel
from utils import initial_parameter


class BertQueryNER(BertPreTrainedModel):
    def __init__(self, config, params):
        super(BertQueryNER, self).__init__(config)
        # nezha layer
        self.bert = RobertaModel(config)

        # start and end position layer
        self.start_outputs = nn.Linear(config.hidden_size, 2)
        self.end_outputs = nn.Linear(config.hidden_size, 2)

        # 动态权重
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)),
                                       requires_grad=True)

        # loss weight
        self.loss_wb = params.weight_start
        self.loss_we = params.weight_end

        self.init_weights()
        self.init_param()

    def init_param(self):
        initial_parameter(self.start_outputs)
        initial_parameter(self.end_outputs)
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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]] 
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]] 
        """
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

        # get logits
        start_logits = self.start_outputs(sequence_output)  # batch x seq_len x 2
        end_logits = self.end_outputs(sequence_output)  # batch x seq_len x 2

        # train
        if start_positions is not None and end_positions is not None:
            # mask
            start_positions = torch.where(attention_mask.view(-1) == 0,
                                          -torch.ones(batch_size * seq_len, device=start_logits.device),
                                          start_positions.view(-1).float()).long()
            end_positions = torch.where(attention_mask.view(-1) == 0,
                                        -torch.ones(batch_size * seq_len, device=start_logits.device),
                                        end_positions.view(-1).float()).long()
            # s & e loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))

            # total loss
            total_loss = self.loss_wb * start_loss + self.loss_we * end_loss
            return total_loss
        # inference
        else:
            start_logits = torch.argmax(F.softmax(start_logits, -1), dim=-1)
            end_logits = torch.argmax(F.softmax(end_logits, -1), dim=-1)
            return start_logits, end_logits


if __name__ == '__main__':
    from transformers import RobertaConfig
    import utils

    params = utils.Params()
    # Prepare model
    config = RobertaConfig.from_pretrained(str(params.bert_model_dir / 'config.json'), output_hidden_states=True)
    model = BertQueryNER.from_pretrained(str(params.bert_model_dir),
                                         config=config, params=params)

    for n, _ in model.named_parameters():
        print(n)
