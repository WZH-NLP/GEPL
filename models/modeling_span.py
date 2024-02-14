# -*- coding:utf-8 -*-
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, BCELoss
from utils.loss_utils import ReverseLayerF

class Span_Detector(BertPreTrainedModel):
    def __init__(self, config, span_num_labels, device):
        super().__init__(config)

        self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.span_num_labels = span_num_labels
        # self.type_num_labels_src = type_num_labels_src+1
        # self.type_num_labels_tgt = type_num_labels_tgt+1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        # self.span = nn.Parameter(torch.randn(type_num_labels, config.hidden_size, span_num_labels))
        # self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels)
        self.classifier_bio_src = nn.Linear(config.hidden_size, span_num_labels)
        self.classifier_bio_tgt = nn.Linear(config.hidden_size, span_num_labels)
        # self.classifier_m_src = nn.Linear(config.hidden_size, type_num_labels_src)
        # self.classifier_m_tgt = nn.Linear(config.hidden_size, type_num_labels_tgt)
        # self.discriminator = nn.Linear(config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        ori_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_bio=None,
        mlm_labels=None,
        graph_prompt_labels=None,
        prompt_label_idx=None,
        O_id=None,
        graph_prompt=False,
        permute=False,
        test=False,
        tgt=True,
        reduction="none",
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        # print(outputs[0][0])
        # # print(outputs[1].size())
        # # print(len(outputs[2]))
        # print(outputs[2][-1][0])
        # print(outputs[2][0].size())
        # print(outputs[2][1].size())
        # exit()
        # print("input_ids",input_ids.size)
        final_embedding = outputs[0] # B, L, D
        # print("final_embedding",final_embedding.shape)
        # print("mlm_label")
        mlm_loss = 0
        if not test:  # 训练阶段时
            if graph_prompt:
                for index, hidden in enumerate(final_embedding):  # 对于每个句子
                    mlm_label = mlm_labels[index]
                    # print("mlm_label", mlm_label)
                    hidden = hidden[mlm_label > 0]
                    if len(
                            hidden) > 0:  # 可以过滤没有共现label的情况，目前只剩下有共现label但用-100填充的，分两种情况：一种区分填充的【-100，-100】，另一种区分【556，-100】
                        labels_graph = graph_prompt_labels[index]
                        prompt_label_idx_temp = prompt_label_idx[index]
                        # print("hidden", hidden.shape)
                        # print("labels_graph", labels_graph)
                        for index_, i in enumerate(hidden):  # 对每个位置分别计算概率,长度为5, 也起到了区分【-100，-100】的作用
                            label = torch.unsqueeze(labels_graph[index_], 0)
                            # label = labels[index_]
                            emb_id = prompt_label_idx_temp[index_].long()
                            # print("emb_id",emb_id)
                            if int(emb_id[1]) == -100:
                                emb_id = emb_id[:-1].long()
                            logits = torch.matmul(
                                hidden[index_, :],
                                self.bert.embeddings.word_embeddings.weight[emb_id].transpose(1, 0)
                                # mask_token位置的rep与三个候选labelrep相乘，计算每个候选label的概率
                            )
                            logits = torch.unsqueeze(logits, 0)
                            mlm_loss += self.criterion(logits, label)  # label表示第几个位置，除了第一个位置有区别之外，其他两个mask_token都是0

        sequence_output1 = self.dropout(final_embedding)
        # sequence_output2 = self.dropout(final_embedding)
        # reverse_feature = ReverseLayerF.apply(sequence_output2, alpha)
        # logits_domain = self.discriminator(reverse_feature) # B, L, 2
        # loss_fct = CrossEntropyLoss()
        # logits_size = logits_domain.size()
        # labels_domain = torch.zeros(logits_size[0]*logits_size[1]).long().to(self.device_)
        if tgt:
            logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
            # labels_domain = labels_domain + 1
            # loss_domain = loss_fct(logits_domain.view(-1, 2), labels_domain)

        else:
            logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
            # loss_domain = loss_fct(logits_domain.view(-1, 2), labels_domain)
        # print(logits_bio.shape)

        hidden_states = ()
        active_loss_hs = True
        if ori_mask is not None:
            active_loss_hs = ori_mask.view(-1) == 1
        for layer_emb in outputs[2]:
            rep_dim = layer_emb.size()[-1]
            active_hidden_states = layer_emb.view(-1, rep_dim)[active_loss_hs]
            hidden_states = hidden_states + (active_hidden_states,)
        # layer_emb = outputs[2][0]
        # rep_dim = layer_emb.size()[-1]
        # active_hidden_states = layer_emb.view(-1, rep_dim)[active_loss_hs]  # select embeddings of original inputs
        # hidden_states = hidden_states + (active_hidden_states,)

        # outputs = (logits_bio, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        outputs = (logits_bio, final_embedding,) + outputs[2:] + (hidden_states,) # add hidden states and attention if they are here
        # outputs = (logits_bio, final_embedding, ) + outputs[2:] + (outputs[2],)# add hidden states and attention if they are here
        # print(len(outputs))
        # print("hidden_state",outputs[2][2].shape)
        if labels_bio is not None:
            # logits = self.logsoftmax(logits)
            # Only keep active parts of the loss
            active_loss = True
            if ori_mask is not None:
                active_loss = ori_mask.view(-1) == 1
                # print("active_loss",active_loss, active_loss.shape)
                # print("attention_mask",attention_mask,attention_mask.shape)
            # print("logits_bio",logits_bio.shape)
            # print(logits_bio.view(-1, self.span_num_labels).shape)
            active_logits = logits_bio.view(-1, self.span_num_labels)[active_loss]
            # print('active_logits',active_logits.shape)
            # print("active_logits", active_logits, active_logits.shape)

            loss_fct = CrossEntropyLoss(reduction=reduction)
            if ori_mask is not None:
                active_labels = labels_bio.view(-1)[active_loss]
                loss_bio = loss_fct(active_logits, active_labels)
            else:
                loss_bio = loss_fct(logits_bio.view(-1, self.span_num_labels), labels_bio.view(-1))

            outputs = (loss_bio, active_logits,) + outputs + (mlm_loss,)
        # print(loss_bio.shape)
        # print(active_logits.shape)
        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # if not test:  # 训练阶段时
        #     if not graph_prompt and not permute:  # 若graph_prompt为false（不用graph_prompt,测试时统计co_occurs不需要用graph_prompt） 并且permute为fasle时计算
        #         # print(graph_prompt)
        #         preds_span_temp = logits_bio  # .detach()
        #         preds_span_temp = torch.argmax(preds_span_temp, dim=-1)
        #         # print(preds_type_temp.shape)
        #         co_occurs = []
        #         for pred in preds_type_temp:  # 统计label共现信息，去除O
        #             pred = pred.tolist()
        #             co_occur = []
        #             labels = list(set(pred))
        #             if O_id in labels:
        #                 labels.remove(O_id)
        #             # print("labels",labels)
        #             for idx, label in enumerate(labels):
        #                 if idx != len(labels) - 1:
        #                     remains = labels[idx + 1:]
        #                     for i in remains:
        #                         # 不考虑顺序
        #                         if [label, i] in co_occur or [i, label] in co_occur:  # 只要有一个就行， 否则的话加一个，加的是哪个不知道
        #                             continue
        #                         else:
        #                             co_occur.append([label, i])
        #             # print("co_occur",co_occur)
        #             co_occurs.append(co_occur)

        # if test or graph_prompt or permute:  # 1. test时 训练时计算mlm loss 或者训练时 permute 为true，三者只要一个为true
        #     return outputs
        # else:
        #     return co_occurs

        # if labels_bio is not None:
        #     # logits = self.logsoftmax(logits)
        #     # Only keep active parts of the loss
        #     active_loss = True
        #     if ori_mask is not None:
        #         active_loss = ori_mask.view(-1) == 1
        #         print("active_loss",active_loss, active_loss.shape)
        #         print("ori_mask",ori_mask,ori_mask.shape)
        #
        #     active_logits = logits_bio.view(-1, self.span_num_labels)[active_loss]
        #     print("active_logits", active_logits, active_logits.shape)
        #
        #     loss_fct = CrossEntropyLoss(reduction=reduction)
        #     if ori_mask is not None:
        #         active_labels = labels_bio.view(-1)[active_loss]
        #         loss_bio = loss_fct(active_logits, active_labels)
        #     else:
        #         loss_bio = loss_fct(logits_bio.view(-1, self.span_num_labels), labels_bio.view(-1))
        #
        #     outputs = (loss_bio, active_logits,) + outputs

        return outputs

    def loss(self, loss_bio, logits_type, tau=0.1, eps=0.0):
        # loss_bio: B*L
        # logits_type: B*L, C
        # print(loss_bio.shape)
        # print(logits_type.shape)
        logits_type = torch.softmax(logits_type.detach()/tau, dim=-1)
        weight = 1.0-logits_type[:, -1]+eps
        loss = torch.mean(loss_bio*weight)

        return loss

    def adv_attack(self, emb, loss, mu):
        loss_grad = torch.autograd.grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, dim=2))
        perturbed_sentence = emb + mu * (loss_grad/(loss_grad_norm.unsqueeze(2)+1e-5))
        
        return perturbed_sentence
