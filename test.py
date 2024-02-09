# -*- coding: utf-8 -*-
# @Author: Hanbing Wang
# @Date  : 2023/1/6 21:41
# @Contact : 596183321@qq.com

import json
import torch
# file_path = './dataset/ai_train.json'
# with open(file_path, 'r', encoding='UTF-8') as f:
#     data = json.load(f)
#     for item in data:
#         words = item["str_words"]
#         labels_ner = item["tags_ner"]
#         labels_esi = item["tags_esi"]
#         labels_net = item["tags_net"]
#         entity = []
#         entities = []
#         temp = 'O'
#         for idx, label in enumerate(labels_net):
#             if label != 'O':
#                 temp = label
#                 word = words[idx]
#                 entity.append(word)
#             else:
#                 if len(entity) != 0:
#                     entities.append([" ".join(entity), temp])
#                     entity = []
#         print(words)
#         print(labels_net)
#         print(entities)
# attention_mask = torch.tensor([[1,0,0,0],[1,1,1,0]])
# a = attention_mask.view(-1) == 1
# print(a)

# from transformers import BertTokenizer, BertModel
# import torch

# tokenizer = BertTokenizer.from_pretrained('./cached_models/bert-base-cased')
# model = BertModel.from_pretrained('./cached_models/bert-base-cased')
#
# inputs = tokenizer("Hello, my dog is cute")
# outputs = model(**inputs)
#
# print(type(outputs))
# print(type(outputs[0]))

import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

