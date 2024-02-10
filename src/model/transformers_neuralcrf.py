# 
# @author: Allan
#

import torch
import torch.nn as nn

from src.model.module.bilstm_encoder import BiLSTMEncoder
from src.model.module.linear_crf_inferencer import LinearCRF
from src.model.module.linear_encoder import LinearEncoder
from src.model.embedder import TransformersEmbedder
from typing import Tuple
from overrides import overrides

from src.data.data_utils import START_TAG, STOP_TAG, PAD


class TransformersCRF(nn.Module):

    def __init__(self, config):
        super(TransformersCRF, self).__init__()
        self.bert_base_cased_path = './pretrained/bert-base-cased'
        # self.embedder = TransformersEmbedder(transformer_model_name=config.embedder_type,
        #                                      parallel_embedder=config.parallel_embedder)
        self.embedder = TransformersEmbedder(transformer_model_name=self.bert_base_cased_path,
                                             parallel_embedder=config.parallel_embedder)
        if config.hidden_dim > 0:
            self.encoder = BiLSTMEncoder(label_size=config.label_size, input_dim=self.embedder.get_output_dim(),
                                         hidden_dim=config.hidden_dim, drop_lstm=config.dropout)
        else:
            self.encoder = LinearEncoder(label_size=config.label_size, input_dim=self.embedder.get_output_dim())
        self.inferencer = LinearCRF(label_size=config.label_size, label2idx=config.label2idx, add_iobes_constraint=config.add_iobes_constraint,
                                    idx2labels=config.idx2labels)
        self.pad_idx = config.label2idx[PAD]
        self.criterion = nn.CrossEntropyLoss()


    @overrides
    def forward(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    orig_to_tok_index: torch.Tensor,
                    input_mask: torch.Tensor,
                    labels: torch.Tensor,
                    mlm_labels: torch.Tensor,
                    graph_labels: torch.Tensor,
                    graph_prompt_labels:torch.Tensor,
                    prompt_label_idx:torch.Tensor) -> torch.Tensor:#prompt_label_idx: torch.Tensor,
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size) # 表示一句话有多少字构成
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        # print('words', words, words.shape)
        word_rep = self.embedder(words, orig_to_tok_index, input_mask) #原输入字的表示，加上graph_prompt的表示
        # print("orig_to_tok_index", orig_to_tok_index, orig_to_tok_index.shape)
        # print("word_rep", word_rep, word_rep.shape)
        # mlm task
        # print("mlm_labels", mlm_labels, mlm_labels.shape)
        hidden_states = self.embedder(words, mlm_labels, input_mask) #取出mask_token的表示，但是要去掉后面的0
        # print("hidden_states", hidden_states, hidden_states.shape)
        # prompt_label_idx = [torch.tensor([0,1]).long()]
        # for index, i in enumerate(prompt_label_idx):
            # print(self.embedder.model.embeddings.word_embeddings.weight[i])# .transpose(1, 0)
            # print(i)

        mlm_loss = 0
        for index, hidden in enumerate(hidden_states):# 对于每个句子
            graph_label = graph_labels[index]
            # print(graph_label)
            hidden = hidden[graph_label > 0]#.view(words.size(0), 3, -1)  # words.size(0)表示batch_size大小 3是3个prompt token的位置
            labels_graph = graph_prompt_labels[index]
            prompt_label_idx_temp = prompt_label_idx[index]
            # print("hidden", hidden)
            # print("labels", labels_graph)
            if len(hidden) > 0:# 可以过滤没有共现label的情况，目前只剩下有共现label但用-100填充的，分两种情况：一种区分填充的【-100，-100】，另一种区分【556，-100】
                for index_, i in enumerate(hidden):  # 对每个位置分别计算概率,长度为5, 也起到了区分【-100，-100】的作用
                    label = torch.unsqueeze(labels_graph[index_], 0)
                    # label = labels[index_]
                    emb_id = prompt_label_idx_temp[index_].long()
                    # print("emb_id",emb_id)
                    if int(emb_id[1]) == -100:
                        emb_id = emb_id[:-1].long()
                    # print("label", label)
                    # print("emb_id", emb_id)
                    logits = torch.matmul(
                            hidden[index_, :],
                            self.embedder.model.embeddings.word_embeddings.weight[emb_id].transpose(1, 0)
                            # mask_token位置的rep与三个候选labelrep相乘，计算每个候选label的概率
                        )
                    logits = torch.unsqueeze(logits, 0)
                    # print("logits", logits)
                    # print("label", label)
                    mlm_loss += self.criterion(logits, label)# label表示第几个位置，除了第一个位置有区别之外，其他两个mask_token都是0

        # hidden_states = hidden_states[graph_labels > 0].view(words.size(0), 3, -1)# words.size(0)表示batch_size大小 3是3个prompt token的位置
        # # print("graph_labels", graph_labels, graph_labels.shape)
        # # print("hidden_states", hidden_states, hidden_states.shape)
        #
        # logits = [
        #     torch.mm(
        #         hidden_states[:, index, :],
        #         self.embedder.model.embeddings.word_embeddings.weight[i].transpose(1, 0)
        #         # mask_token位置的rep与三个候选labelrep相乘，计算每个候选label的概率
        #     )
        #     for index, i in enumerate(prompt_label_idx)  # 对每个位置分别计算概率,长度为5
        # ]

        # print('word_rep', word_rep, word_rep.shape)
        lstm_scores = self.encoder(word_rep, word_seq_lens)# word_rep中句子的长度一致
        # print('lstm_scores', lstm_scores, lstm_scores.shape)
        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)# 即word_seq_lens, 增加graph prompt之后words还包含sep和无用label
        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev).view(1, sent_len).expand(batch_size, sent_len)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        # print("mask", mask.shape)
        unlabed_score, labeled_score =  self.inferencer(lstm_scores, word_seq_lens, labels, mask)
        return unlabed_score - labeled_score + 0.1 * mlm_loss

    def decode(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    orig_to_tok_index: torch.Tensor,
                    input_mask,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        word_rep = self.embedder(words, orig_to_tok_index, input_mask)
        features = self.encoder(word_rep, word_seq_lens)
        bestScores, decodeIdx = self.inferencer.decode(features, word_seq_lens)
        return bestScores, decodeIdx
