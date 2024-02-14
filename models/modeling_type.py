# -*- coding:utf-8 -*-
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss
import torch.nn.functional as F
import numpy as np

class Type_Classifier(BertPreTrainedModel):
    def __init__(self, config, type_num_labels_src, type_num_labels_tgt, device, domain):
        super().__init__(config)

        self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.span_num_labels = span_num_labels
        self.type_num_labels_src = type_num_labels_src+1
        self.type_num_labels_tgt = type_num_labels_tgt+1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        # self.W = nn.Parameter(torch.randn(config.hidden_size, 300))
        # self.base = nn.Parameter(torch.randn(8, 300))
        # self.span = nn.Parameter(torch.randn(type_num_labels, config.hidden_size, span_num_labels))
        # self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels)
        # self.classifier_bio_src = nn.Linear(config.hidden_size, span_num_labels)
        # self.classifier_bio_tgt = nn.Linear(config.hidden_size, span_num_labels)
        self.classifier_type_src = nn.Linear(config.hidden_size, type_num_labels_src+1)
        self.classifier_type_tgt = nn.Linear(config.hidden_size, type_num_labels_tgt+1)
        self.classifier_type = nn.Linear(config.hidden_size, type_num_labels_tgt)
        domain_map = {
                       "politics":torch.tensor([3,4,5,6]).long().to(device),
                       "science":torch.tensor([9,10,11,12]).long().to(device),
                       "music":torch.tensor([5,6,10,11]).long().to(device),
                       "literature":torch.tensor([5,7,8,9]).long().to(device),
                       "ai":torch.tensor([4,6,7,8]).long().to(device)
                   }
        self.label_ind_map = domain_map[domain]
        # self.label_ind_map = torch.tensor([4,6,7,8]).long().to(device) # ai
        # self.label_ind_map = torch.tensor([5,7,8,9]).long().to(device) # literature
        # self.label_ind_map = torch.tensor([5,6,10,11]).long().to(device) # music
        # self.label_ind_map = torch.tensor([9,10,11,12]).long().to(device) # science
        # self.label_ind_map = torch.tensor([3,4,5,6]).long().to(device) # politics
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
        labels_type=None,
        logits_bio=None,
        mlm_labels=None,
        graph_prompt_labels=None,
        prompt_label_idx=None,
        O_id=None,
        graph_prompt=False,
        permute=False,
        test=False,
        tgt=True,
        reduction="none"
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
        final_embedding = outputs[0] # B, L, D
        # print("final_embedding", final_embedding.size())
        # sequence_output1 = self.dropout(final_embedding)
        mlm_loss = 0
        if not test:# 训练阶段时
            if graph_prompt: #如果使用graph prompt
                for index, hidden in enumerate(final_embedding):# 对于每个句子
                    mlm_label = mlm_labels[index]
                    # print("mlm_label", mlm_label)
                    hidden = hidden[mlm_label > 0]
                    # print("hidden",hidden.shape)
                    if len(hidden) > 0:# 可以过滤没有共现label的情况，目前只剩下有共现label但用-100填充的，分两种情况：一种区分填充的【-100，-100】，另一种区分【556，-100】
                        labels_graph = graph_prompt_labels[index]
                        prompt_label_idx_temp = prompt_label_idx[index]
                        # print("hidden", hidden.shape)
                        # print("labels_graph", labels_graph)
                        for index_, i in enumerate(hidden):  # 对每个位置分别计算概率,长度为5, 也起到了区分【-100，-100】的作用
                            label = torch.unsqueeze(labels_graph[index_], 0)
                            # print("label", label)
                            # label = labels[index_]
                            emb_id = prompt_label_idx_temp[index_].long()
                            # print("emb_id",emb_id)
                            if int(emb_id[1]) == -100:
                                emb_id = emb_id[:-1].long()
                            # print("emb_id", emb_id)
                            logits = torch.matmul(
                                    hidden[index_, :],
                                    self.bert.embeddings.word_embeddings.weight[emb_id].transpose(1, 0)
                                    # mask_token位置的rep与三个候选labelrep相乘，计算每个候选label的概率
                                )
                            # print("logits", logits)
                            logits = torch.unsqueeze(logits, 0)
                            mlm_loss += self.criterion(logits, label)# label表示第几个位置，除了第一个位置有区别之外，其他两个mask_token都是0

        sequence_output2 = self.dropout(final_embedding)
        # bs, l, d = sequence_output2.size()
        # alpha1 = torch.bmm(sequence_output2, self.W.unsqueeze(0).repeat(bs, 1, 1)) # N, L, d
        # alpha2 = torch.bmm(alpha1, self.base.t().unsqueeze(0).repeat(bs, 1, 1)) # N, L, C_base
        # alpha = torch.softmax(alpha2, dim=-1) # N, L, C_base
        # seq_out = torch.bmm(alpha, self.base.unsqueeze(0).repeat(bs, 1, 1)) # N, L, d_new
        # seq_embed = sequence_output.view(-1, self.hidden_size) # B*L, D
        # seq_size = seq_embed.size()
        # logits = torch.bmm(seq_embed.unsqueeze(0).expand(self.type_num_labels, 
        #                 seq_size[0], seq_size[1]), self.span).permute(1, 0, 2) # B*L, type_num, span_num
        type_num_labels = self.type_num_labels_tgt
        if tgt:
            # logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
            logits_type = self.classifier_type_tgt(sequence_output2)
        else:
            # logits_bio = self.classifier_bio_src(sequence_output1) # B, L, C
            # logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
            # logits_type = self.classifier_type_tgt(sequence_output2)[:,:,self.label_ind_map]
            logits_type = self.classifier_type_src(sequence_output2)
            type_num_labels = self.type_num_labels_src
        # logits = self.classifier_bio_tgt(sequence_output1) # B, L, C
        # print(logits.size())

        hidden_states = ()
        active_loss_hs = True
        if ori_mask is not None:
            active_loss_hs = ori_mask.view(-1) == 1
        for layer_emb in outputs[2]: #outputs[2] is hidden states of 13 layers
            rep_dim = layer_emb.size()[-1]
            active_hidden_states = layer_emb.view(-1, rep_dim)[active_loss_hs]# select embeddings of original inputs
            hidden_states = hidden_states + (active_hidden_states,)
        # layer_emb = outputs[2][0]
        # rep_dim = layer_emb.size()[-1]
        # active_hidden_states = layer_emb.view(-1, rep_dim)[active_loss_hs]# select embeddings of original inputs
        # hidden_states = hidden_states + (active_hidden_states,)

        # outputs = (logits_type, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        outputs = (logits_type, final_embedding,) + outputs[2:] + (hidden_states,)  # add hidden states and attention if they are here
        # outputs = (logits_type, final_embedding,) + outputs[2:] + (outputs[2],)
        # logits_bio = torch.argmax(logits_bio, dim=-1)

        if labels_type is not None:
            # logits = self.logsoftmax(logits)
            # Only keep active parts of the loss
            active_loss = True
            if ori_mask is not None:
                active_loss = ori_mask.view(-1) == 1

            # bio_labels = logits_bio.view(-1)
            # label_mask = bio_labels < 2
            # active_loss = active_loss&label_mask
            
            active_logits = logits_type.view(-1, type_num_labels)[active_loss]

            loss_fct = CrossEntropyLoss(reduction=reduction)
            # if attention_mask is not None:
            active_labels = labels_type.view(-1)[active_loss]
            if len(active_labels) == 0:
                loss_type = torch.tensor(0).float().to(self.device_)
            else:
                loss_type = loss_fct(active_logits, active_labels)
            # else:
            #     loss_type = loss_fct(logits_type.view(-1, type_num_labels), labels_type.view(-1))

            outputs = (loss_type, active_logits,) + outputs + (mlm_loss,)

        if not test:  # 训练阶段时
            if not graph_prompt and not permute: #若graph_prompt为false（不用graph_prompt,测试时统计co_occurs不需要用graph_prompt） 并且permute为fasle时计算
                # print(graph_prompt)
                preds_type_temp = logits_type#.detach()
                # print("logits_type",logits_type.shape)
                preds_type_temp = torch.argmax(preds_type_temp, dim=-1)
                # print(preds_type_temp.shape)
                co_occurs=[]
                for pred in preds_type_temp:# 每句话 统计label共现信息，去除O
                    pred = pred.tolist()
                    co_occur = []
                    labels = list(set(pred))
                    if O_id in labels:
                        labels.remove(O_id)
                    # print("labels",labels)
                    for idx, label in enumerate(labels):
                        if idx != len(labels) - 1:
                            remains = labels[idx+1:]
                            for i in remains:
                                # 不考虑顺序
                                if [label, i] in co_occur or [i, label] in co_occur:# 只要有一个就行， 否则的话加一个，加的是哪个不知道
                                    continue
                                else:
                                    co_occur.append([label, i])
                    # print("co_occur",co_occur)
                    co_occurs.append(co_occur)

            # print("co_occurs", np.array(co_occurs).size())

        if test or graph_prompt or permute:# 1. test时 训练时计算mlm loss 或者训练时 permute 为true，三者只要一个为true
            return outputs
        else:
            return co_occurs
        # return outputs

    def loss(self, loss_type, logits_bio, tau=0.1, eps=0.0):
        # loss_type: B*L 
        # logits_bio: B*L, C
        # delta: music 0.1
        logits_bio = torch.softmax(logits_bio.detach()/tau, dim=-1)
        weight = 1.0-logits_bio[:, -1]+eps
        loss = torch.mean(loss_type*weight)
        return loss


    def mix_up(self, src_rep, tgt_rep, src_label, tgt_label, alpha, beta):
        # remove 'O'
        num_labels_src = self.type_num_labels_src-1
        num_labels_tgt = self.type_num_labels_tgt-1
        src_tgt_map = self.label_ind_map

        rep_dim = src_rep.size()[-1]
        src_rep = src_rep.view(-1, rep_dim) # B*L, d
        # print("src_rep", src_rep)
        tgt_rep = tgt_rep.view(-1, rep_dim) # B*L, d
        # print("src_label", src_label)
        src_label = src_label.view(-1) # B*L

        tgt_label = tgt_label.view(-1) # B*L
        # Select entity tokens
        mask_src = (src_label<num_labels_src)&(src_label>=0)
        mask_tgt = (tgt_label<num_labels_tgt)&(tgt_label>=0)
        src_sel = src_rep[mask_src] # N1, d
        # print("mask_src", mask_src)
        # print("src_sel", src_sel)
        tgt_sel = tgt_rep[mask_tgt] # N2, d
        src_label_sel = src_label[mask_src] # N1
        tgt_label_sel = tgt_label[mask_tgt] # N2
        N1 = src_sel.size()[0]
        N2 = tgt_sel.size()[0]
        # print("N1", N1)
        # print("N2", N2)
        # print(src_sel)
        src_exp = src_sel.unsqueeze(0).repeat(N2, 1, 1).view(N1*N2, -1)
        tgt_exp = tgt_sel.unsqueeze(1).repeat(1, N1, 1).view(N1*N2, -1)
        src_label_exp = src_label_sel.unsqueeze(0).repeat(N2, 1).view(-1).unsqueeze(1) # N1*N2, 1
        tgt_label_exp = tgt_label_sel.unsqueeze(1).repeat(1, N1).view(-1).unsqueeze(1) # N1*N2, 1
        src_onehot_ = torch.zeros(N1*N2, num_labels_src).to(self.device_).scatter_(1, src_label_exp, 1)
        tgt_onehot = torch.zeros(N1*N2, num_labels_tgt).to(self.device_).scatter_(1, tgt_label_exp, 1)
        src_onehot = torch.zeros(N1*N2, num_labels_tgt).to(self.device_)
        src_onehot[:, src_tgt_map] = src_onehot_
        mix_rep = alpha*src_exp + beta*tgt_exp
        mix_label = alpha*src_onehot + beta*tgt_onehot
        # print(mix_label)
        # exit()
        logits_type = F.log_softmax(self.classifier_type(mix_rep), dim=-1) # N1*N2, C

        loss_fct = KLDivLoss()

        loss = loss_fct(logits_type, mix_label)

        return loss

    def adv_attack(self, emb, loss, mu):
        loss_grad = torch.autograd.grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, dim=2))
        perturbed_sentence = emb + mu * (loss_grad/(loss_grad_norm.unsqueeze(2)+1e-5))
        
        return perturbed_sentence
