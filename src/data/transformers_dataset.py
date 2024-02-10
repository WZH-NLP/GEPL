# 
# @author: Allan
#

import random
from tqdm import tqdm
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizer
import collections
import numpy as np
from termcolor import colored
from scipy import stats
from src.data.data_utils import convert_iobes, build_label_idx, check_all_labels_in_dict
import bert_score
from src.data import Instance
from src.data.search_space_manager import SearchSpaceManager
from src.config import Config
import sys
import os

Feature = collections.namedtuple('Feature', 'input_ids attention_mask token_type_ids orig_to_tok_index word_seq_len label_ids mlm_labels graph_labels graph_prompt_labels prompt_label_idx')#
Feature.__new__.__defaults__ = (None,) * 10



def maybe_show_prompt(id, word, prompt, mod):
    if id % mod == 0:
        print(colored(f"Instance {id}: {word}", "blue"))
        print(colored(f"Prompt {id}: {prompt}\n", "yellow"))

def convert_instances_to_feature_tensors(instances: List[Instance],
                                         tokenizer: PreTrainedTokenizer,
                                         label2idx: Dict[str, int],
                                         prompt: str = None, # "max", "random", "sbert", "bertscore"
                                         template: str = None, # "no_context", "context", "context_all", "structure", "structure_all"
                                         prompt_candidates_from_outside: List[str] = None,
                                         constrained_instances: bool = False):
    
    
    features = []
    candidates = [] # usually whole train dataset = prompt_candidates_from_outside

    graph_prompt = 1

    if prompt_candidates_from_outside is not None and prompt is not None:
        candidates = prompt_candidates_from_outside
    else:
        candidates = instances

    ## Construct entity dictionary for "max" or "random".
    entity_dict = {}
    for inst in candidates:
        for entity, label in inst.entities:
            if label not in entity_dict:
                entity_dict[label] = {}
            if entity not in entity_dict[label]:
                entity_dict[label][entity] = [inst]
            else:
                entity_dict[label][entity].append(inst)

    ## Popular Entity
    if prompt == "max":
        max_entities = {}
        for label in entity_dict:
            for x in sorted(entity_dict[label].items(), key=lambda kv: len(kv[1]), reverse=True)[0:1]:
                max_entities[label] = [x[0], tuple(x[1])[0]]
                
    if prompt == "sbert" or prompt == "bertscore":
        search_space = []
        search_space_dict = {}
        trans_func = None
        for inst in candidates:
            search_space.append(" ".join(inst.ori_words))
            search_space_dict[" ".join(inst.ori_words)] = inst
        if prompt == "sbert":
            search_model = SentenceTransformer('all-roberta-large-v1')

            def _trans_func(insts):
                if len(insts) == 0:
                    return None
                sub_search_space = list(map(lambda i: " ".join(i.ori_words), insts))
                return sub_search_space, search_model.encode(sub_search_space, convert_to_tensor=True)

            trans_func = _trans_func
            if not constrained_instances:
                global_corpus_embeddings = search_model.encode(search_space, convert_to_tensor=True)
        if prompt == "bertscore":
            bert_score_model_type = bert_score.lang2model["en"]
            num_layers = bert_score.model2layers[bert_score_model_type]
            bert_score_model = bert_score.get_model(bert_score_model_type, num_layers, False)

            def _trans_func(insts):
                if len(insts) == 0:
                    return None
                sub_search_space = list(map(lambda i: " ".join(i.ori_words), insts))
                return sub_search_space

            trans_func = _trans_func

        if constrained_instances:
            manager = SearchSpaceManager(candidates, trans_func)

    num_to_examine = 10 # Number of sample prompts we want to see
    step_sz = len(instances) // num_to_examine

    if prompt:
        print(colored("Some sample prompts used: ", "red"))

    top_k_correct_selection_count = 0
    scores = []
    count200 = 0
    count128 = 0
    countall = 0
    for idx, inst in enumerate(instances):
        words = inst.ori_words
        orig_to_tok_index = []
        tokens = []
        mlm_labels = []
        for i, word in enumerate(words):
            orig_to_tok_index.append(len(tokens))# 【0，x】x是每一个word对应的sub-token数量
            word_tokens = tokenizer.tokenize(" " + word)
            for sub_token in word_tokens:
                tokens.append(sub_token)
        # select graph prompt when calculating probability
        if graph_prompt:
            if len(inst.co_occurs) > 0:# 这个位置从co_occurs中选会泄露测试信息
                base = 0
                for co in inst.co_occurs:
                    # if len(co) == 5:  # is connected to
                    mlm_labels.append(len(tokens)+base+4)# 将3个mask_token的rep挑选出来，+4是因为需要提出前面的[sep] label [sep] mask mask mask [sep] label [sep] .. [sep] label [sep]
                    mlm_labels.append(len(tokens)+base+5)#
                    mlm_labels.append(len(tokens)+base+6)
                    base += 9
                    # else:
                    #     orig_to_tok_index.append(
                    #         len(tokens) + base + 4)  # 将3个mask_token的rep挑选出来，+4是因为需要提出前面的[sep] label [sep] mask mask mask mask [sep] label [sep] .. [sep] label [sep]
                    #     orig_to_tok_index.append(len(tokens) + base + 5)  #
                    #     orig_to_tok_index.append(len(tokens) + base + 6)
                    #     orig_to_tok_index.append(len(tokens) + base + 7)
                    #     base += 10
        # print("orig_to_tok_index", len(orig_to_tok_index))
        # mlm_labels = [0] * len(words) + [1] * 3 * len(inst.co_occurs)
        # mlm_labels = torch.Tensor(np.array(mlm_labels)).long()
        labels = inst.labels
        label_ids = [label2idx[label] for label in labels] if labels else [-100] * len(words)# 一个word对应一个label，在后面模型转化的时候
        # print("label_ids_before", len(label_ids))
        if prompt is None:
            input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
        elif prompt == "sbert":
            prompt_tokens = []
            query = " ".join(inst.ori_words)
            query_embedding = search_model.encode(query, convert_to_tensor=True)

            if constrained_instances:
                comparison_sets = []
                # try combined set
                combined_set = manager.superset_labels_search_space(inst)
                if combined_set is None:
                    for lb in set(label for entity, label in inst.entities):
                        comparison_sets.append(manager.single_label_search_space(lb))
                else:
                    comparison_sets.append(combined_set)

                results = []
                for sub_search_space, corpus_embeddings in comparison_sets:
                    # We use cosine-similarity and torch.topk to find the highest 5 scores
                    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
                    top_results = torch.topk(cos_scores, k=1)
                    results.extend(zip(top_results[0], map(lambda x: sub_search_space[x], top_results[1])))
            else:
                results = []
                cos_scores = util.pytorch_cos_sim(query_embedding, global_corpus_embeddings)[0]
                top_results = torch.topk(cos_scores, k=1)
                results.extend(zip(top_results[0], map(lambda x: search_space[x], top_results[1])))

            for score, selected_id in results:
                prompt_words = search_space_dict[selected_id].ori_words
                prompt_entities = search_space_dict[selected_id].entities
                # stats
                if prompt_candidates_from_outside is not None and idx % step_sz == 0:
                    print("[debug] Query: " + " ".join(inst.ori_words))
                    print("[debug] Selected: " + selected_id)
                    print("[debug] Score: %f" % score)
                    print("[debug] Prompt Entities: ", prompt_entities)
                if search_space_dict[selected_id] == inst:
                    top_k_correct_selection_count += 1
                scores.append(score.item())
                # stats end

                if template == "context_all":
                    for i, word in enumerate(prompt_words):
                        instance_tokens = tokenizer.tokenize(" " + word)
                        for sub_token in instance_tokens:
                            prompt_tokens.append(sub_token)

                    for entity in prompt_entities:
                        entity_tokens = tokenizer.tokenize(" " + entity[0])
                        for sub_token in entity_tokens:
                            prompt_tokens.append(sub_token)

                        prompt_tokens.append("is")
                        prompt_tokens.append(entity[1])
                        prompt_tokens.append(".")

                elif template == "structure_all":
                    instance_prompt_tokens = []
                    for i, word in enumerate(prompt_words):
                        instance_tokens = tokenizer.tokenize(" " + word)
                        for sub_token in instance_tokens:
                            instance_prompt_tokens.append(sub_token)

                    for entity in prompt_entities:
                        entity_tokens = tokenizer.tokenize(" " + entity[0])
                        start_ind = instance_prompt_tokens.index(entity_tokens[0])
                        end_ind = instance_prompt_tokens.index(entity_tokens[-1])

                        instance_prompt_tokens.insert(end_ind + 1, ']')
                        instance_prompt_tokens.insert(end_ind + 1, entity[1])
                        instance_prompt_tokens.insert(end_ind + 1, '|')
                        instance_prompt_tokens.insert(start_ind, '[')
                    prompt_tokens.extend(instance_prompt_tokens)

                elif template == 'lexical_all':
                    instance_prompt_tokens = []
                    for i, word in enumerate(prompt_words):
                        instance_tokens = tokenizer.tokenize(" " + word)
                        for sub_token in instance_tokens:
                            instance_prompt_tokens.append(sub_token)

                    for entity in prompt_entities:
                        entity_tokens = tokenizer.tokenize(" " + entity[0])

                        start_ind = instance_prompt_tokens.index(entity_tokens[0])
                        end_ind = instance_prompt_tokens.index(entity_tokens[-1])

                        instance_prompt_tokens[start_ind] = entity[1]
                        del instance_prompt_tokens[start_ind + 1:end_ind + 1]
                    prompt_tokens.extend(instance_prompt_tokens)

            maybe_show_prompt(idx, words, prompt_tokens, step_sz)
            input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token] + prompt_tokens + [tokenizer.sep_token])
            # if len(input_ids)>200:
            #     count200 += 1
            # elif len(input_ids) > 100:
            #     count128 += 1
            # countall += 1

        elif prompt == "bertscore":
            prompt_tokens = []
            query = " ".join(inst.ori_words)

            if constrained_instances:
                comparison_sets = []
                # try combined set
                combined_set = manager.superset_labels_search_space(inst)
                if combined_set is None:
                    for lb in set(label for entity, label in inst.entities):
                        comparison_sets.append(manager.single_label_search_space(lb))
                else:
                    comparison_sets.append(combined_set)

                results = []
                for sub_search_space in comparison_sets:
                    queries = [query] * len(sub_search_space)
                    # We use cosine-similarity and torch.topk to find the highest 5 scores
                    P, R, F1 = bert_score.score(sub_search_space, queries, model_type=(bert_score_model_type, bert_score_model), verbose=idx % step_sz == 0)
                    top_results = torch.topk(F1, k=1)
                    results.extend(zip(top_results[0], map(lambda x: sub_search_space[x], top_results[1])))
            else:
                results = []
                queries = [query] * len(search_space)
                # We use cosine-similarity and torch.topk to find the highest 5 scores
                P, R, F1 = bert_score.score(search_space, queries, model_type=(bert_score_model_type, bert_score_model), verbose=idx % step_sz == 0)
                top_results = torch.topk(F1, k=1)
                results.extend(zip(top_results[0], map(lambda x: search_space[x], top_results[1])))


            for score, selected_id in results:
                prompt_words = search_space_dict[selected_id].ori_words
                prompt_entities = search_space_dict[selected_id].entities
                # stats
                if prompt_candidates_from_outside is not None and idx % step_sz == 0:
                    print("[debug] Query: " + " ".join(inst.ori_words))
                    print("[debug] Selected: " + selected_id)
                    print("[debug] Score: %f" % score)
                    print("[debug] Prompt Entities: ", prompt_entities)
                if search_space_dict[selected_id] == inst:
                    top_k_correct_selection_count += 1
                scores.append(score.item())
                # stats end

                if template == "context_all":
                    for i, word in enumerate(prompt_words):
                        instance_tokens = tokenizer.tokenize(" " + word)
                        for sub_token in instance_tokens:
                            prompt_tokens.append(sub_token)

                    for entity in prompt_entities:
                        entity_tokens = tokenizer.tokenize(" " + entity[0])
                        for sub_token in entity_tokens:
                            prompt_tokens.append(sub_token)

                        prompt_tokens.append("is")
                        prompt_tokens.append(entity[1])
                        prompt_tokens.append(".")

                elif template == "structure_all":
                    instance_prompt_tokens = []
                    for i, word in enumerate(prompt_words):
                        instance_tokens = tokenizer.tokenize(" " + word)
                        for sub_token in instance_tokens:
                            instance_prompt_tokens.append(sub_token)

                    for entity in prompt_entities:
                        entity_tokens = tokenizer.tokenize(" " + entity[0])
                        start_ind = instance_prompt_tokens.index(entity_tokens[0])
                        end_ind = instance_prompt_tokens.index(entity_tokens[-1])

                        instance_prompt_tokens.insert(end_ind + 1, ']')
                        instance_prompt_tokens.insert(end_ind + 1, entity[1])
                        instance_prompt_tokens.insert(end_ind + 1, '|')
                        instance_prompt_tokens.insert(start_ind, '[')
                    prompt_tokens.extend(instance_prompt_tokens)

                elif template == 'lexical_all':
                    instance_prompt_tokens = []
                    for i, word in enumerate(prompt_words):
                        instance_tokens = tokenizer.tokenize(" " + word)
                        for sub_token in instance_tokens:
                            instance_prompt_tokens.append(sub_token)

                    for entity in prompt_entities:
                        entity_tokens = tokenizer.tokenize(" " + entity[0])

                        start_ind = instance_prompt_tokens.index(entity_tokens[0])
                        end_ind = instance_prompt_tokens.index(entity_tokens[-1])

                        instance_prompt_tokens[start_ind] = entity[1]
                        del instance_prompt_tokens[start_ind + 1:end_ind + 1]
                    prompt_tokens.extend(instance_prompt_tokens)

            maybe_show_prompt(idx, words, prompt_tokens, step_sz)
            input_ids = tokenizer.convert_tokens_to_ids(
                [tokenizer.cls_token] + tokens + [tokenizer.sep_token] + prompt_tokens + [tokenizer.sep_token])


        elif prompt == "max":
            prompt_tokens = []
            for entity_label in max_entities:
                if template in ["no_context", "context", "context_all"]:
                    if template in ["context", "context_all"]:
                        instance_words = max_entities[entity_label][1].ori_words
                        for i, word in enumerate(instance_words):
                            instance_tokens = tokenizer.tokenize(" " + word)
                            for sub_token in instance_tokens:
                                prompt_tokens.append(sub_token)

                    if template in ["no_context", "context"]:
                        entity_tokens = tokenizer.tokenize(" " + max_entities[entity_label][0])
                        for sub_token in entity_tokens:
                            prompt_tokens.append(sub_token)

                        prompt_tokens.append("is")
                        prompt_tokens.append(entity_label)
                        prompt_tokens.append(".")
                        prompt_tokens.append(tokenizer.sep_token)

                    elif template in ["context_all"]:
                        for entity in max_entities[entity_label][1].entities:
                            entity_tokens = tokenizer.tokenize(" " + entity[0])
                            for sub_token in entity_tokens:
                                prompt_tokens.append(sub_token)

                            prompt_tokens.append("is")
                            prompt_tokens.append(entity[1])
                            prompt_tokens.append(".")
                        prompt_tokens.append(tokenizer.sep_token)

                if template in ["structure", "structure_all",'lexical','lexical_all']:
                    instance_prompt_tokens = []
                    instance_words = max_entities[entity_label][1].ori_words
                    for i, word in enumerate(instance_words):
                        instance_tokens = tokenizer.tokenize(" " + word)
                        for sub_token in instance_tokens:
                            instance_prompt_tokens.append(sub_token)

                    if template == "structure":
                        entity_tokens = tokenizer.tokenize(" " + max_entities[entity_label][0])
                        start_ind = instance_prompt_tokens.index(entity_tokens[0])
                        end_ind = instance_prompt_tokens.index(entity_tokens[-1])
                        instance_prompt_tokens.insert(end_ind + 1, ']')
                        instance_prompt_tokens.insert(end_ind + 1, entity_label)
                        instance_prompt_tokens.insert(end_ind + 1, '|')
                        instance_prompt_tokens.insert(start_ind, '[')

                    elif template == "structure_all":
                        for entity in max_entities[entity_label][1].entities:
                            entity_tokens = tokenizer.tokenize(" " + entity[0])
                            start_ind = instance_prompt_tokens.index(entity_tokens[0])
                            end_ind = instance_prompt_tokens.index(entity_tokens[-1])
                            instance_prompt_tokens.insert(end_ind + 1, ']')
                            instance_prompt_tokens.insert(end_ind + 1, entity[1])
                            instance_prompt_tokens.insert(end_ind + 1, '|')
                            instance_prompt_tokens.insert(start_ind, '[')
                    
                    elif template =='lexical':
                        entity_tokens = tokenizer.tokenize(" " + max_entities[entity_label][0])
                        start_ind = instance_prompt_tokens.index(entity_tokens[0])
                        end_ind = instance_prompt_tokens.index(entity_tokens[-1])

                        instance_prompt_tokens[start_ind] = entity_label
                        del instance_prompt_tokens[start_ind+1:end_ind+1]

                    
                    elif template=='lexical_all':
                        for entity in max_entities[entity_label][1].entities:
                            entity_tokens = tokenizer.tokenize(" " + entity[0])
                            start_ind = instance_prompt_tokens.index(entity_tokens[0])
                            end_ind = instance_prompt_tokens.index(entity_tokens[-1])
                            instance_prompt_tokens[start_ind] = entity[1]
                            del instance_prompt_tokens[start_ind + 1:end_ind + 1]

                    prompt_tokens.extend(instance_prompt_tokens)
                    prompt_tokens.append(tokenizer.sep_token)

            if graph_prompt:
                if len(inst.co_occurs) > 0:
                    graph_prompt_tokens = []
                    graph_prompt_labels = []
                    for co in inst.co_occurs:
                        if co[2] == 'is':  # is connected to
                            graph_prompt_tokens.append(tokenizer.sep_token)# [sep] label [sep] mask mask mask [sep] label [sep]
                            graph_prompt_tokens.append(co[0])
                            graph_prompt_tokens.append(tokenizer.sep_token)
                            graph_prompt_tokens.append(tokenizer.mask_token)
                            graph_prompt_tokens.append(tokenizer.mask_token)
                            graph_prompt_tokens.append(tokenizer.mask_token)
                            graph_prompt_tokens.append(tokenizer.sep_token)
                            graph_prompt_tokens.append(co[1])
                            graph_prompt_tokens.append(tokenizer.sep_token)

                            graph_prompt_labels.extend([0,0,0])

                            # temp = tokenizer.convert_tokens_to_ids(co[2:])# 这个地方不确定是否需要” “隔开
                            # graph_prompt_labels.extend(temp)# extend or append?

                            # print("4", graph_prompt_labels)
                            # print("graph_prompt_tokens", graph_prompt_tokens)
                            # print("graph_prompt_labels", graph_prompt_labels)
                        else:
                            graph_prompt_tokens.append(tokenizer.sep_token)
                            graph_prompt_tokens.append(co[0])
                            graph_prompt_tokens.append(tokenizer.sep_token)
                            graph_prompt_tokens.append(tokenizer.mask_token)
                            graph_prompt_tokens.append(tokenizer.mask_token)
                            graph_prompt_tokens.append(tokenizer.mask_token)
                            graph_prompt_tokens.append(tokenizer.sep_token)
                            graph_prompt_tokens.append(co[1])
                            graph_prompt_tokens.append(tokenizer.sep_token)

                            graph_prompt_labels.extend([1, 0, 0])
                            # temp = tokenizer.convert_tokens_to_ids(co[2:])# 这个地方不确定是否需要” “隔开
                            # graph_prompt_labels.extend(temp)  # extend or append?
                        # label_ids.extend(graph_prompt_labels)
                        # graph_prompt_labels = []

                else:
                    graph_prompt_tokens = []
                    graph_prompt_labels = []
            else:
                graph_prompt_tokens = []
            # print("label_ids", len(label_ids))
            # print(label_ids)
            # print([tokenizer.cls_token] + tokens + graph_prompt_tokens + prompt_tokens)
            maybe_show_prompt(idx, words, prompt_tokens, step_sz)
            input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + graph_prompt_tokens + prompt_tokens)
            #print('input_ids',len(input_ids))

        elif prompt == "random":
            prompt_tokens = []
            for entity_label in entity_dict:
                if template in ["no_context", "context", "context_all"]:
                    entity = random.choice(tuple(entity_dict[entity_label]))
                    instance = random.choice(entity_dict[entity_label][entity])

                    if template in ["context", "context_all"]:
                        instance_words = instance.ori_words
                        for i, word in enumerate(instance_words):
                            instance_tokens = tokenizer.tokenize(" " + word)
                            for sub_token in instance_tokens:
                                prompt_tokens.append(sub_token)

                    if template in ["no_context", "context"]:
                        entity_tokens = tokenizer.tokenize(" " + entity)
                        for sub_token in entity_tokens:
                            prompt_tokens.append(sub_token)

                        prompt_tokens.append("is")
                        prompt_tokens.append(entity_label)
                        prompt_tokens.append(".")
                        prompt_tokens.append(tokenizer.sep_token)

                    elif template in ["context_all"]:
                        for entity in instance.entities:
                            entity_tokens = tokenizer.tokenize(" " + entity[0])
                            for sub_token in entity_tokens:
                                prompt_tokens.append(sub_token)

                            prompt_tokens.append("is")
                            prompt_tokens.append(entity[1])
                            prompt_tokens.append(".")
                        prompt_tokens.append(tokenizer.sep_token)

                if template in ["structure", "structure_all","lexical","lexical_all"]:
                    entity = random.choice(tuple(entity_dict[entity_label]))
                    instance = random.choice(entity_dict[entity_label][entity])

                    instance_prompt_tokens = []
                    instance_words = instance.ori_words
                    for i, word in enumerate(instance_words):
                        instance_tokens = tokenizer.tokenize(" " + word)
                        for sub_token in instance_tokens:
                            instance_prompt_tokens.append(sub_token)


                    if template == "structure":
                        entity_tokens = tokenizer.tokenize(" " + entity)
                        start_ind = instance_prompt_tokens.index(entity_tokens[0])
                        end_ind = instance_prompt_tokens.index(entity_tokens[-1])

                        instance_prompt_tokens.insert(end_ind + 1, ']')
                        instance_prompt_tokens.insert(end_ind + 1, entity_label)
                        instance_prompt_tokens.insert(end_ind + 1, '|')
                        instance_prompt_tokens.insert(start_ind, '[')

                    elif template == "structure_all":
                        for entity in instance.entities:
                            entity_tokens = tokenizer.tokenize(" " + entity[0])
                            start_ind = instance_prompt_tokens.index(entity_tokens[0])
                            end_ind = instance_prompt_tokens.index(entity_tokens[-1])

                            instance_prompt_tokens.insert(end_ind + 1, ']')
                            instance_prompt_tokens.insert(end_ind + 1, entity[1])
                            instance_prompt_tokens.insert(end_ind + 1, '|')
                            instance_prompt_tokens.insert(start_ind, '[')

                    elif template =='lexical':
                        entity_tokens = tokenizer.tokenize(" " + entity)
                        start_ind = instance_prompt_tokens.index(entity_tokens[0])
                        end_ind = instance_prompt_tokens.index(entity_tokens[-1])
                        instance_prompt_tokens[start_ind] = entity_label
                        del instance_prompt_tokens[start_ind + 1:end_ind + 1]

                    elif template=='lexical_all':
                        for entity in instance.entities:
                            entity_tokens = tokenizer.tokenize(" " + entity[0])
                            start_ind = instance_prompt_tokens.index(entity_tokens[0])
                            end_ind = instance_prompt_tokens.index(entity_tokens[-1])
                            instance_prompt_tokens[start_ind] = entity[1]
                            del instance_prompt_tokens[start_ind + 1:end_ind + 1]

                    prompt_tokens.extend(instance_prompt_tokens)
                    prompt_tokens.append(tokenizer.sep_token)

            maybe_show_prompt(idx, words, prompt_tokens, step_sz)
            input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token] + prompt_tokens)

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        prompt_label_idx = [list(sorted([tokenizer.convert_tokens_to_ids('is'), tokenizer.convert_tokens_to_ids('not')])),#
               list([tokenizer.convert_tokens_to_ids('connected'), -100]),list([tokenizer.convert_tokens_to_ids('to'), -100])] * len(inst.co_occurs)
        # print(inst.co_occurs)
        # print("set", len(set))
        # print("graph_prompt_labels",len(graph_prompt_labels))
        # prompt_label_idx = [
        #     torch.Tensor(i).long() for i in set
        # ]
        # prompt_label_idx = set

        # print(type(input_ids))
        # print(type(input_mask))
        # print(type(orig_to_tok_index))
        # print(type(segment_ids))
        # print(type(len(orig_to_tok_index)))
        # print(type(label_ids))
        # print(type(prompt_label_idx))
        # print(type(mlm_labels))
        # prompt_label_idx = torch.Tensor(prompt_label_idx)

        if len(input_ids) > 512:
            continue
        else:
            # print('----------------------------------')
            features.append(Feature(input_ids=input_ids,
                                    attention_mask=input_mask,
                                    orig_to_tok_index=orig_to_tok_index,
                                    token_type_ids=segment_ids,
                                    word_seq_len=len(orig_to_tok_index),
                                    label_ids=label_ids,
                                    mlm_labels = mlm_labels,
                                    graph_prompt_labels=graph_prompt_labels,
                                    prompt_label_idx=prompt_label_idx
                                    ))

    # graph_prompt_labels = graph_prompt_labels,
    # prompt_label_idx = prompt_label_idx
    # print("大于200", count200 / countall )
    # print("大于128", count128 / countall)
    if prompt_candidates_from_outside is None and (prompt == "sbert" or prompt == "bertscore"):
        print(colored("[Info] Top 1 selection precision: %.2f" % (top_k_correct_selection_count / len(instances)), 'yellow'))

    if len(scores) > 0 and (prompt == "sbert" or prompt == "bertscore"):
        print("[debug] Score Stats:", stats.describe(scores))
        print("[debug] Scores:", scores)
        print("##################################")

    if prompt_candidates_from_outside is None and prompt is not None:
        return features, candidates
    else:
        return features

class TransformersNERDataset(Dataset):

    def __init__(self, file: str,
                 tokenizer: PreTrainedTokenizer,
                 is_train: bool,
                 sents: List[List[str]] = None,
                 label2idx: Dict[str, int] = None,
                 number: int = -1,
                 percentage: int = 100,
                 prompt: str = None,
                 template: str = None,
                 prompt_candidates_from_outside: List[str] = None):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        ## read all the instances. sentences and labels
        self.percentage = percentage
        insts = self.read_txt(file=file, number=number) if sents is None else self.read_from_sentences(sents)
        self.insts = insts
        # print('---------')
        # print(insts)
        if is_train:
            print(f"[Data Info] Using the training set to build label index")
            assert label2idx is None
            # conf = Config()
            # dataset = conf.dataset
            # if dataset == 'politics' or dataset == 'science':
            print('file:',file)
            if 'music' in file:
                # print("--------------")
            # if "conll" not in file:
                dev_file = 'dataset/' + 'music' + '/dev.txt'
                test_file = 'dataset/' + 'music' + '/test.txt'
                insts_dev = self.read_txt(file=dev_file, number=number)
                insts_test = self.read_txt(file=test_file, number=number)
                insts_dev.extend(insts)
                insts_dev.extend(insts_test)
                idx2labels, label2idx = build_label_idx(insts_dev)
            elif 'science' in file:
                # print('=====================')
                dev_file = 'dataset/' + 'science' + '/dev.txt'
                test_file = 'dataset/' + 'science' + '/test.txt'
                insts_dev = self.read_txt(file=dev_file, number=number)
                insts_test = self.read_txt(file=test_file, number=number)
                insts_dev.extend(insts)
                insts_dev.extend(insts_test)
                idx2labels, label2idx = build_label_idx(insts_dev)
            elif 'politics' in file:
                # print("politicspoliticspoliticspoliticspoliticspolitics")
                dev_file = 'dataset/' + 'politics' + '/dev.txt'
                test_file = 'dataset/' + 'politics' + '/test.txt'
                insts_dev = self.read_txt(file=dev_file, number=number)
                insts_test = self.read_txt(file=test_file, number=number)
                insts_dev.extend(insts)
                insts_dev.extend(insts_test)
                idx2labels, label2idx = build_label_idx(insts_dev)
            elif 'literature' in file:
                # print('literatureliteratureliteratureliteratureliterature')
                dev_file = 'dataset/' + 'literature' + '/dev.txt'
                test_file = 'dataset/' + 'literature' + '/test.txt'
                insts_dev = self.read_txt(file=dev_file, number=number)
                insts_test = self.read_txt(file=test_file, number=number)
                insts_dev.extend(insts)
                insts_dev.extend(insts_test)
                idx2labels, label2idx = build_label_idx(insts_dev)
            elif 'ai' in file:
                # print('aiaiaiaiaiiaiaiaiaia')
                dev_file = 'dataset/' + 'ai' + '/dev.txt'
                test_file = 'dataset/' + 'ai' + '/test.txt'
                insts_dev = self.read_txt(file=dev_file, number=number)
                insts_test = self.read_txt(file=test_file, number=number)
                insts_dev.extend(insts)
                insts_dev.extend(insts_test)
                idx2labels, label2idx = build_label_idx(insts_dev)
            else:
                ## build label to index mapping. e.g., B-PER -> 0, I-PER -> 1
                idx2labels, label2idx = build_label_idx(insts)

            # idx2labels = ['<PAD>', 'O', 'B-election', 'I-election', 'E-election', 'S-misc', 'B-politicalparty', 'I-politicalparty',
            #  'E-politicalparty', 'S-politicalparty', 'S-organisation', 'B-misc', 'E-misc', 'B-politician',
            #  'E-politician', 'B-organisation', 'I-organisation', 'E-organisation', 'B-person', 'E-person',
            #  'S-politician', 'B-event', 'E-event', 'S-country', 'B-country', 'E-country', 'S-location', 'B-location',
            #  'E-location', 'I-misc', 'I-politician', 'I-location', 'I-event', 'I-person', 'S-person', 'I-country',
            #  'S-event', '<START>', '<STOP>']
            # label2idx = {'<PAD>': 0, 'O': 1, 'B-election': 2, 'I-election': 3, 'E-election': 4, 'S-misc': 5, 'B-politicalparty': 6, 'I-politicalparty': 7, 'E-politicalparty': 8, 'S-politicalparty': 9, 'S-organisation': 10, 'B-misc': 11, 'E-misc': 12, 'B-politician': 13, 'E-politician': 14, 'B-organisation': 15, 'I-organisation': 16, 'E-organisation': 17, 'B-person': 18, 'E-person': 19, 'S-politician': 20, 'B-event': 21, 'E-event': 22, 'S-country': 23, 'B-country': 24, 'E-country': 25, 'S-location': 26, 'B-location': 27, 'E-location': 28, 'I-misc': 29, 'I-politician': 30, 'I-location': 31, 'I-event': 32, 'I-person': 33, 'S-person': 34, 'I-country': 35, 'S-event': 36, '<START>': 37, '<STOP>': 38}

            # idx2labels, label2idx = build_label_idx(insts)

            self.idx2labels = idx2labels
            self.label2idx = label2idx
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            # check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)

        if is_train and prompt is not None:
            self.insts_ids, self.prompt_candidates = convert_instances_to_feature_tensors(insts, tokenizer, label2idx, prompt=prompt, template=template)
        else:
            self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, label2idx, prompt=prompt, template=template, prompt_candidates_from_outside=prompt_candidates_from_outside)
            self.prompt_candidates = None
        # print('insts_ids',len(self.insts_ids))
        self.tokenizer = tokenizer


    def read_from_sentences(self, sents: List[List[str]]):
        """
        sents = [['word_a', 'word_b'], ['word_aaa', 'word_bccc', 'word_ccc']]
        """
        insts = []
        for sent in sents:
            insts.append(Instance(words=sent, ori_words=sent))
        return insts


    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding")
        print(f"[Data Info] Modify src/data/transformers_dataset.read_txt function if you have other requirements")
        insts = []

        # cooccurs_file_src = os.path.join(
        #     args.data_dir,
        #     "conll2003_{}_cooccur.npy".format(
        #         args.dataset
        #     ),
        # )
        dataset = file.split("/")[1]
        cooccurs_file_src = 'dataset/' + dataset + "/conll2003_{}_cooccur.npy".format(dataset)
        cooccur = np.load(cooccurs_file_src).tolist()
        print("cooccur",cooccur)

        with open(file, 'r', encoding='utf-8') as f:
            words = []
            ori_words = []
            labels = []
            entities = []
            entity = []
            entity_label = []
            co_occurs = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    labels = convert_iobes(labels)
                    if len(entity) != 0:
                        entities.append([" ".join(entity),entity_label[0]])
                    if len(np.array(entities).shape) > 1:
                        for index, entity in enumerate(entities):
                            if index != len(entities) - 1:
                                for entity_temp in entities[index + 1:]:
                                    if entity[1] != entity_temp[1]:
                                        if [entity[1], entity_temp[1]] in cooccur or [entity_temp[1], entity[1]] in cooccur: # 说明两个label之间有关联
                                            if [entity[1], entity_temp[1], 'is', 'connected', 'to'] not in co_occurs and [entity_temp[1], entity[1], 'is', 'connected', 'to'] not in co_occurs:# 这个关系的两种情况都没有时再加入
                                                co_occurs.append([entity[1], entity_temp[1], 'is', 'connected', 'to'])
                                        else:
                                            if [entity[1], entity_temp[1], 'not', 'connected', 'to'] not in co_occurs and [entity_temp[1], entity[1], 'not', 'connected', 'to'] not in co_occurs:
                                                co_occurs.append([entity[1], entity_temp[1], 'not', 'connected', 'to'])
                            else:
                                continue
                    if len(set(labels)) > 1:
                        insts.append(Instance(words=words, ori_words=ori_words, labels=labels, entities=entities, co_occurs=co_occurs))
                    words = []
                    ori_words = []
                    labels = []
                    entities = []
                    entity = []
                    entity_label = []
                    co_occurs = []
                    if len(insts) == number:
                        break
                    continue
                ls = line.split()
                word, label = ls[0],ls[-1]
                ori_words.append(word)
                words.append(word)
                labels.append(label)


                if label.startswith("B"):
                    entity.append(word)
                    entity_label.append(label.split('-')[1])
                elif label.startswith("I"):
                    entity.append(word)
                else:
                    if len(entity) != 0:
                        entities.append([" ".join(entity), entity_label[0]])
                        entity = []
                        entity_label = []
        # print(insts)
        # print('--------------------------------')

        numbers = int(len(insts) * self.percentage / 100)
        percentage_insts = insts[:numbers]
        # print(percentage_insts)

        print("number of sentences: {}".format(len(percentage_insts)))
        return percentage_insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch:List[Feature]):
        word_seq_len = [len(feature.orig_to_tok_index) for feature in batch]
        max_seq_len = max(word_seq_len)# 最大字长度
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])# 最大token长度
        max_graph_prompt_len = max([len(feature.mlm_labels) for feature in batch])# 最大graph prompt长度
        max_label_len = max([len(feature.graph_prompt_labels) for feature in batch])

        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            mask = feature.attention_mask + [0] * padding_length
            type_ids = feature.token_type_ids + [self.tokenizer.pad_token_type_id] * padding_length
            padding_word_len = max_seq_len - len(feature.orig_to_tok_index)
            orig_to_tok_index = feature.orig_to_tok_index + [0] * padding_word_len
            label_ids = feature.label_ids + [0] * padding_word_len

            #
            padding_length_mlm_labels = max_graph_prompt_len - len(feature.mlm_labels)
            mlm_labels = feature.mlm_labels + [0] * padding_length_mlm_labels
            graph_labels = np.ones(max_graph_prompt_len) * (-1)
            mask_pos = np.where(np.array(mlm_labels) > 0)[0]
            graph_labels[mask_pos] = 1

            graph_prompt_labels = feature.graph_prompt_labels
            prompt_label_idx = feature.prompt_label_idx
            padding_length_labels = max_label_len - len(graph_prompt_labels)
            graph_prompt_labels = graph_prompt_labels + [-100] * padding_length_labels
            prompt_label_idx = prompt_label_idx + [[-100, -100]] * padding_length_labels
            # print(graph_prompt_labels, len(graph_prompt_labels))
            # print(prompt_label_idx,len(prompt_label_idx))


            # print("input_ids", len(input_ids))
            # print("orig_to_tok_index", len(orig_to_tok_index))
            # print("prompt_label_idx",prompt_label_idx, len(prompt_label_idx))
            # print("mlm_labels", len(mlm_labels))

            batch[i] = Feature(input_ids=np.asarray(input_ids),
                               attention_mask=np.asarray(mask), token_type_ids=np.asarray(type_ids),
                               orig_to_tok_index=np.asarray(orig_to_tok_index),
                               word_seq_len =feature.word_seq_len,
                               label_ids=np.asarray(label_ids),
                               mlm_labels=np.asarray(mlm_labels),
                               graph_labels=np.asarray(graph_labels),
                               graph_prompt_labels=np.asarray(graph_prompt_labels),
                               prompt_label_idx=np.asarray(prompt_label_idx))
        results = Feature(*(default_collate(samples) for samples in zip(*batch)))
        return results
