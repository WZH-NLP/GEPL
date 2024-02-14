# -*- coding:utf-8 -*-
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import json
import os

from utils.data_utils import load_and_cache_examples, tag_to_id, get_chunks
from flashtool import Logger
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
# )
# logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
# logging_fh.setLevel(logging.DEBUG)
# logger.addHandler(logging_fh)
# logger.warning(
#     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
#     args.local_rank,
#     device,
#     args.n_gpu,
#     bool(args.local_rank != -1),
#     args.fp16,
# )
def add_graph_prompt(args, inputs, co_occurs, tokenizer, cooccur):
    input_ids = inputs["input_ids"]
    input_ids_final = []
    for index, input_id in enumerate(input_ids):
        input_id =  input_id.tolist()
        # print("input_id",input_id)
        if 0 in input_id:
            input_id_temp=input_id[:input_id.index(0)]
            co_occur = co_occurs[index]
            for occur in co_occur:
            # for i in range(len(co_occur)):
            #     occur = random.choice(co_occur)
            #     co_occur.remove(occur)
                if len(input_id_temp) + 5 < args.max_seq_length - 1:
                    # print("occur",occur)
                    # print("co_occur",co_occur)
                    if [occur[0],occur[1]] in cooccur or [occur[1],occur[0]] in cooccur:
                        input_id_temp.append(occur[0])
                        input_id_temp.append(tokenizer.convert_tokens_to_ids('is'))
                        input_id_temp.append(tokenizer.convert_tokens_to_ids('related'))
                        input_id_temp.append(tokenizer.convert_tokens_to_ids('to'))
                        input_id_temp.append(occur[1])
                    else:
                        input_id_temp.append(occur[0])
                        input_id_temp.append(tokenizer.convert_tokens_to_ids('not'))
                        input_id_temp.append(tokenizer.convert_tokens_to_ids('related'))
                        input_id_temp.append(tokenizer.convert_tokens_to_ids('to'))
                        input_id_temp.append(occur[1])
                    input_id_temp.append(tokenizer.convert_tokens_to_ids("[SEP]"))

            padding_lenth = args.max_seq_length - len(input_id_temp)
            # if padding_lenth >= 0:
            input_id_temp += [0] * padding_lenth
        else:
            input_id_temp = input_id
        # print("input_id_1", input_id_temp)
        # print("input_id_temp_padding", input_id_temp)
        input_ids_final.append(input_id_temp)

    return torch.tensor(input_ids_final, dtype=torch.long).to(args.device)

def evaluate(vals, args, span_model, type_model, tokenizer, id_to_label_span, \
    pad_token_label_id, best, best_bio, mode, logger, prefix="", verbose=True):
    
    _, eval_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode=mode)
    span_to_id = {id_to_label_span[id_]:id_ for id_ in id_to_label_span}
    non_entity_id = span_to_id["O"]
    num_class = len(span_to_id)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    with open(args.data_dir + args.dataset + "_tag_to_id.json", 'r') as f:
        data = json.load(f)
        types = data["type"]
        O_id = types['O']

    cooccurs_file_src = os.path.join(
        args.data_dir,
        "{}_{}_cooccur.npy".format(
            args.src_dataset, args.dataset
        ),
    )
    cooccur = np.load(cooccurs_file_src).tolist()
    cooccur_id = []
    for i in cooccur:
        id_0 = types[i[0]]
        id_1 = types[i[1]]
        cooccur_id.append([id_0, id_1])
    # print("cooccur_id", cooccur_id)


    logger.info("***** Running evaluation %s *****", prefix)
    if verbose:
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_type = None
    preds_bio = None
    span_label_ids = None
    type_label_ids = None
    # type_label_ids = None
    out_label_ids_type = None
    out_label_ids_bio = None
    att_mask = None
    span_model.eval()
    type_model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "ori_mask": batch[6], "labels_type": batch[3], "logits_bio": outputs_span[2],"O_id":O_id, "tgt": True}
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "ori_mask": batch[6], "labels_type": batch[3],"O_id": O_id, "tgt": True}
            outputs_type = type_model(**inputs) # outputs_type为co_occurs
            # print("outputs_type",outputs_type)

            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_bio": batch[2], "O_id": O_id, "tgt": True, "reduction": "none"}
            # outputs_span = span_model(**inputs)

            # inputs_graph_span = add_graph_prompt(args, inputs, outputs_span, tokenizer, cooccur_id)  # .to(args.device)

            inputs_graph = add_graph_prompt(args, inputs, outputs_type, tokenizer, cooccur_id)  # .to(args.device)

            inputs = {"input_ids": inputs_graph, "attention_mask": batch[1], "ori_mask": batch[6], "labels_bio": batch[2], "tgt": True, "reduction": "none"}
            outputs_span = span_model(**inputs)

            inputs = {"input_ids": inputs_graph, "attention_mask": batch[1], "ori_mask": batch[6],
                      "labels_type": batch[3], "logits_bio": outputs_span[2], "test": True, "tgt": True}
            outputs_type = type_model(**inputs)  # outputs_type为co_occurs


            # inputs_graph_span = add_graph_prompt(args, inputs, outputs_span, tokenizer, cooccur_id)  # .to(args.device)
            # inputs = {"input_ids": inputs_graph_span, "attention_mask": batch[1], "ori_mask": batch[6], "logits_bio": batch[2], "test": True, "tgt": True}
            # outputs_span = span_model(**inputs)  # outputs_type为co_occurs

            # inputs = {"input_ids": inputs_graph, "attention_mask": batch[1], "ori_mask": batch[6], "labels_type": batch[3],
            #           "logits_bio": outputs_span[2], "tgt": True}
            # outputs_type = type_model(**inputs)  # outputs_type为co_occurs

            span_logits = outputs_span[2]
            type_logits = outputs_type[2]
            # loss1 = outputs_span[0]
            loss1 = span_model.loss(outputs_span[0], outputs_type[1])
            # loss2 = outputs_type[0]
            loss2 = type_model.loss(outputs_type[0], outputs_span[1])
            # loss = outputs_span[0]+outputs_type[0]
            loss = loss1+loss2

            if args.n_gpu > 1:
                loss = loss.mean()
                # meta_loss = meta_loss.mean()

            eval_loss += loss.item()
            # meta_eval_loss += meta_loss.item()
        nb_eval_steps += 1
        
        if preds_type is None:
            preds_type = type_logits.detach() # B, L, C
            # print(preds_type)
            preds_bio = span_logits.detach() # B, L, C
            # meta_preds = meta_logits.detach().cpu().numpy()
            # span_label_ids = batch[2] # B, L
            # type_label_ids = batch[3] # B, L
            out_label_ids_bio = batch[2] # B, L
            out_label_ids_type = batch[3] # B, L 每一句话中的所有type label
            # att_mask = batch[1].unsqueeze(1).expand(span_logits.size()[:3])
        else:
            preds_type = torch.cat((preds_type, type_logits.detach()), dim=0)
            preds_bio = torch.cat((preds_bio, span_logits.detach()), dim=0)
            # span_label_ids = torch.cat((span_label_ids, batch[2]), dim=0)
            # type_label_ids = torch.cat((type_label_ids, batch[3]), dim=0)
            out_label_ids_bio = torch.cat((out_label_ids_bio, batch[2]), dim=0)
            out_label_ids_type = torch.cat((out_label_ids_type, batch[3]), dim=0)
            # att_mask = torch.cat((att_mask, batch[1].unsqueeze(1).expand(span_logits.size()[:3])), dim=0)

    # preds: nb, type_num, L, span_num
    # out_label_ids: nb*type_num, L
    
    eval_loss = eval_loss/nb_eval_steps
    # meta_eval_loss = meta_eval_loss/nb_eval_steps
    # print(preds)
    # task_preds = np.argmax(task_preds, axis=-1)
    # meta_preds = np.argmax(meta_preds, axis=-1)
    # a, b = out_label_ids.size()
    preds_type = torch.argmax(preds_type, dim=-1)
    # print(preds_type[1])
    # print(preds_type.shape)

    preds_bio = torch.argmax(preds_bio, dim=-1)
    # print(preds_bio.shape)
    # att_mask = att_mask.view(a,b) 
    # print(preds_type)

    # label_map = {i: label for i, label in enumerate(labels)}
    # label_map = id_to_label
    # task_preds_list = [[] for _ in range(out_label_ids.shape[0])]
    # meta_preds_list = [[] for _ in range(out_label_ids.shape[0])]
    # out_id_list = [[] for _ in range(out_label_ids.shape[0])]
    # task_preds_id_list = [[] for _ in range(out_label_ids.shape[0])]
    # meta_preds_id_list = [[] for _ in range(out_label_ids.shape[0])]
    out_id_list_type = [[] for _ in range(out_label_ids_type.shape[0])]
    preds_id_list_type = [[] for _ in range(out_label_ids_type.shape[0])]# B

    out_id_list_bio = [[] for _ in range(out_label_ids_bio.shape[0])]
    preds_id_list_bio = [[] for _ in range(out_label_ids_bio.shape[0])]

    for i in range(out_label_ids_type.shape[0]):# for each sentence
        for j in range(out_label_ids_type.shape[1]):# for each word in a sentence
            if out_label_ids_type[i, j] != pad_token_label_id: # 若label不是padding
                out_id_list_type[i].append(out_label_ids_type[i][j])# gT
                preds_id_list_type[i].append(preds_type[i][j])# 预测值

    for i in range(out_label_ids_bio.shape[0]):
        for j in range(out_label_ids_bio.shape[1]):
            if out_label_ids_bio[i, j] != pad_token_label_id:
                out_id_list_bio[i].append(out_label_ids_bio[i][j])
                preds_id_list_bio[i].append(preds_bio[i][j])

    correct_preds, total_correct, total_preds = 0., 0., 0. # i variables
    correct_preds_bio, total_correct_bio, total_preds_bio = 0., 0., 0. # i variables
    correct_preds_type, total_correct_type, total_preds_type = 0., 0., 0. # i variables

    types_list = []

    data_path = os.path.join(args.data_dir, "{}_tag_to_id.json".format(args.dataset))
    data_path_src = os.path.join(args.data_dir, "{}_tag_to_id.json".format(args.src_dataset))

    with open(data_path, 'r') as f:
        data = json.load(f)# {"ner":{}, "span":{}, "type":{}}
        types = data["type"]
    with open(data_path_src, 'r') as f:
        data = json.load(f)# {"ner":{}, "span":{}, "type":{}}
        types_src = data["type"]

    for i, idx in types_src.items():
        if i not in types_list and i != "O":
            types_list.append(i)
    for i, idx in types.items():
        if i not in types_list and i != "O":
            types_list.append(i)

    correct_label, total_label, total_pred_label, f1 = [0] * len(types_list), [0] * len(types_list), [0] * len(types_list), [0] * len(types_list)

    label_id_map = {label: idx for idx, label in enumerate(types_list)}

    # print("EVAL:")
    # wrong = []
    for ground_truth_id_type, predicted_id_type, ground_truth_id_bio, predicted_id_bio in zip(out_id_list_type, \
                                                            preds_id_list_type, out_id_list_bio, preds_id_list_bio):#out_id_list_type是去除padding token的gT
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks, lab_chunks_bio = get_chunks(ground_truth_id_type, ground_truth_id_bio, tag_to_id(args.data_dir, args.dataset))
        # print("ground_truth:")
        # print(lab_chunks_bio)
        # print("lab_chunks",lab_chunks) lab_chunks [('election', 2, 6), ('misc', 7, 8), ('politicalparty', 14, 17), ('politicalparty', 18, 19), ('politicalparty', 21, 23)]

        lab_chunks      = set(lab_chunks)
        lab_chunks_bio  = set(lab_chunks_bio)
        lab_pred_chunks, lab_pred_chunks_bio = get_chunks(predicted_id_type, predicted_id_bio, tag_to_id(args.data_dir, args.dataset))
        # print("pred:")
        # print(lab_pred_chunks_bio)
        # print(lab_pred_chunks)

        lab_pred_chunks = set(lab_pred_chunks)
        lab_pred_chunks_bio = set(lab_pred_chunks_bio)

        lab_pred_type_chunks, _ = get_chunks(predicted_id_type, ground_truth_id_bio, tag_to_id(args.data_dir, args.dataset))
        lab_pred_type_chunks = set(lab_pred_type_chunks)

        # Updating the i variables
        correct_preds += len(lab_chunks & lab_pred_chunks)# 预测正确的
        total_preds   += len(lab_pred_chunks)# 所有预测的，用于计算p
        total_correct += len(lab_chunks)# groundtruth的长度，用于计算recall

        # print("lab_chunks", lab_chunks)
        # print("lab_pred_chunks", lab_pred_chunks)
        # print("and", lab_chunks & lab_pred_chunks)
        # wrong_pred = list(lab_pred_chunks - (lab_chunks & lab_pred_chunks))# 预测错误的label
        # wrong.extend(wrong_pred)
        correct_list = list(lab_chunks & lab_pred_chunks)
        for i in correct_list:
            i_label = i[0]
            i_label_id = label_id_map[i_label]
            correct_label[i_label_id] += 1

        lab_chunks_list = list(lab_chunks)
        for i in lab_chunks_list:
            i_label = i[0]
            i_label_id = label_id_map[i_label]
            total_label[i_label_id] += 1

        lab_pred_chunks_list = list(lab_pred_chunks)
        for i in lab_pred_chunks_list:
            i_label = i[0]
            i_label_id = label_id_map[i_label]
            total_pred_label[i_label_id] += 1

        # Updating the i variables
        correct_preds_bio += len(lab_chunks_bio & lab_pred_chunks_bio)
        total_preds_bio   += len(lab_pred_chunks_bio)
        total_correct_bio += len(lab_chunks_bio)

        # Updating the i variables
        correct_preds_type += len(lab_chunks & lab_pred_type_chunks)
        total_preds_type   += len(lab_pred_type_chunks)
        total_correct_type += len(lab_chunks)

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0
    vals.append(new_F)

    p_bio   = correct_preds_bio / total_preds_bio if correct_preds_bio > 0 else 0
    r_bio   = correct_preds_bio / total_correct_bio if correct_preds_bio > 0 else 0
    new_F_bio  = 2 * p_bio * r_bio / (p_bio + r_bio) if correct_preds_bio > 0 else 0

    p_type   = correct_preds_type / total_preds_type if correct_preds_type > 0 else 0
    r_type   = correct_preds_type / total_correct_type if correct_preds_type > 0 else 0
    new_F_type  = 2 * p_type * r_type / (p_type + r_type) if correct_preds_type > 0 else 0

    for idx, i in enumerate(f1):
        p = correct_label[idx] / total_pred_label[idx] if correct_label[idx] > 0 else 0
        r = correct_label[idx] / total_label[idx] if correct_label[idx] > 0 else 0
        new_F_temp = 2 * p * r / (p + r) if correct_label[idx] > 0 else 0
        f1[idx] = new_F_temp

    is_updated = False
    if new_F > best[-1]:
        best = [p, r, new_F]
        is_updated = True


    if new_F_bio > best_bio[-1]:
        best_bio = [p_bio, r_bio, new_F_bio]
        # is_updated = True

    results = {
       "loss": eval_loss,
       "precision": p,
       "recall": r,
       "f1": new_F,
       "best_precision": best[0],
       "best_recall": best[1],
       "best_f1": best[-1],
       "precision_bio": p_bio,
       "recall_bio": r_bio,
       "f1_bio": new_F_bio,
       "best_precision_bio": best_bio[0],
       "best_recall_bio": best_bio[1],
       "best_f1_bio": best_bio[-1],
       "precision_type": p_type,
       "recall_type": r_type,
       "f1_type": new_F_type,
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    # logger.info("***** Eval results for each label *****")
    # re = {label: f1[idx] for idx, label in enumerate(types_list)}
    re = {label: f1[label_id_map[label]] for label in label_id_map}
    logger.info("results for each labels")
    logger.info(re)

    correct_re = {label: correct_label[label_id_map[label]] for label in label_id_map}
    total_preds_re = {label: total_pred_label[label_id_map[label]] for label in label_id_map}
    total_re = {label: total_label[label_id_map[label]] for label in label_id_map}
    logger.info("correct_re")
    logger.info(correct_re)
    logger.info("total_preds_re")
    logger.info(total_preds_re)
    logger.info("total_re")
    logger.info(total_re)


    # logger.info("***** Meta Eval results %s *****", prefix)
    # for key in sorted(meta_results.keys()):
    #     logger.info("  %s = %s", key, str(meta_results[key]))

    return best, best_bio, is_updated
