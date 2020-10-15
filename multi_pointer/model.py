#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""model"""

import torch
import torch.nn as nn

# from transformers import BertPreTrainedModel, RobertaModel
from NEZHA.model_NEZHA import BertPreTrainedModel, NEZHAModel
from utils import initial_parameter


class BertMultiPointer(BertPreTrainedModel):
    def __init__(self, config, params):
        super(BertMultiPointer, self).__init__(config)
        # pretrain model layer
        self.bert = NEZHAModel(config)
        self.tag_size = len(params.label_list)

        # start and end position layer
        self.start_outputs = nn.Linear(config.hidden_size, self.tag_size)
        self.end_outputs = nn.Linear(config.hidden_size, self.tag_size)

        # 动态权重
        self.fusion_layers = params.fusion_layers
        self.dym_weight = nn.Parameter(torch.ones((self.fusion_layers, 1, 1, 1)),
                                       requires_grad=True)

        self.threshold = params.multi_threshold

        self.apply(self.init_bert_weights)
        self.init_param()

    def init_param(self):
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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None,
                start_positions=None, end_positions=None):
        """
        Args:
            start_positions: (batch x max_len x tag_size)
            end_positions: (batch x max_len x tag_size)
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

        # get logits
        start_logits = self.start_outputs(sequence_output)  # batch x seq_len x tag_size
        end_logits = self.end_outputs(sequence_output)  # batch x seq_len x tag_size

        # train
        if start_positions is not None and end_positions is not None:
            # s & e loss
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            # (bs, seq_len*tag_size)
            start_loss = loss_fct(start_logits.view(batch_size, -1), start_positions.view(batch_size, -1).float())
            end_loss = loss_fct(end_logits.view(batch_size, -1), end_positions.view(batch_size, -1).float())
            # mask loss
            total_loss = torch.sum((start_loss + end_loss).view(batch_size, seq_len, self.tag_size) *
                                   attention_mask.unsqueeze(-1).expand(batch_size, seq_len, self.tag_size))
            # calculate the average
            total_loss /= torch.sum(attention_mask) * self.tag_size
            return total_loss
        # inference
        else:
            start_pre = torch.sigmoid(start_logits)  # batch x seq_len x tag_size
            end_pre = torch.sigmoid(end_logits)

            # get output
            tensor_ones = torch.ones(start_pre.size(), device=start_pre.device)
            tensor_zeros = torch.zeros(start_pre.size(), device=start_pre.device)
            start_pre = torch.where(start_pre > self.threshold, tensor_ones, tensor_zeros)
            end_pre = torch.where(end_pre > self.threshold, tensor_ones, tensor_zeros)
            return start_pre, end_pre


if __name__ == '__main__':
    from transformers import RobertaConfig
    import utils

    params = utils.Params()
    # Prepare model
    config = RobertaConfig.from_pretrained(str(params.bert_model_dir / 'config.json'), output_hidden_states=True)
    model = BertMultiPointer.from_pretrained(str(params.bert_model_dir),
                                             config=config, params=params)

    for n, _ in model.named_parameters():
        print(n)
