# -*- coding:utf-8 -*
import logging
import os
import json
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sentence_transformers import SentenceTransformer, util
from src.data.search_space_manager import SearchSpaceManager
from termcolor import colored
import numpy as np

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, spans, types, entities, co_occurs):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.spans = spans
        self.types = types
        self.entities = entities
        self.co_occurs = co_occurs

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, full_label_ids, span_label_ids, type_label_ids, label_mask, ori_mask, mlm_label, prompt_label_idx, graph_prompt_labels):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.full_label_ids = full_label_ids
        self.span_label_ids = span_label_ids
        self.type_label_ids = type_label_ids
        self.label_mask = label_mask
        self.ori_mask = ori_mask
        self.mlm_label = mlm_label
        self.prompt_label_idx = prompt_label_idx
        self.graph_prompt_labels = graph_prompt_labels

def read_examples_from_file(args, data_dir, mode):
    if mode == "train":
        if args.proportion == 1:
            file_path = os.path.join(data_dir, "{}_{}.json".format(args.dataset, mode))
        else:
            file_path = os.path.join(data_dir, "{}_{}_{}.json".format(args.dataset, mode, args.proportion))
    else:
        file_path = os.path.join(data_dir, "{}_{}.json".format(args.dataset, mode))
    print(file_path,'#####################')
    guid_index = 1
    examples = []

    if args.proportion == 1:
        cooccurs_file_src = os.path.join(
            args.data_dir,
            "{}_{}_cooccur.npy".format(
                args.src_dataset, args.dataset
            ),
        )
    else:
        cooccurs_file_src = os.path.join(
            args.data_dir,
            "{}_{}_cooccur_{}.npy".format(
                args.src_dataset, args.dataset, args.proportion
            ),
        )

    cooccur = np.load(cooccurs_file_src).tolist()

    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data:
            words = item["str_words"]
            labels_ner = item["tags_ner"]
            labels_esi = item["tags_esi"]
            labels_net = item["tags_net"]
            entity = []
            entities = []
            co_occurs = []
            temp = 'O'
            for idx, label in enumerate(labels_net):
                if label != 'O':
                    temp = label
                    word = words[idx]
                    entity.append(word)
                else:
                    if len(entity) != 0:
                        entities.append([" ".join(entity), temp])
                        entity = []
            # print(entities)
            # print(np.array(entities).shape, len(np.array(entities).shape))
            if np.array(entities).shape[0] > 1: #若句子中entity的个数大于1（包含重复entity）
                for index, entity in enumerate(entities):
                    if index != len(entities) - 1: #不是最后一个
                        for entity_temp in entities[index + 1:]:
                            if entity[1] != entity_temp[1]:
                                if [entity[1], entity_temp[1]] in cooccur or [entity_temp[1], entity[1]] in cooccur:  # 说明两个label之间有关联
                                    if [entity[1], entity_temp[1], 'is', 'related', 'to'] not in co_occurs and [entity_temp[1], entity[1], 'is', 'related', 'to'] not in co_occurs:  # 这个关系的两种情况都没有时再加入
                                        co_occurs.append([entity[1], entity_temp[1], 'is', 'related', 'to'])
                                else:
                                    if [entity[1], entity_temp[1], 'not', 'related', 'to'] not in co_occurs and [entity_temp[1], entity[1], 'not', 'related', 'to'] not in co_occurs:
                                        co_occurs.append([entity[1], entity_temp[1], 'not', 'related', 'to'])
                    else:
                        continue
            # print(co_occurs)
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels_ner, spans=labels_esi, types=labels_net, entities=entities, co_occurs=co_occurs))
            guid_index += 1
    examples_src = []
    examples_inter = []
    if mode == "train":
        file_path = os.path.join(data_dir, "{}_{}.json".format(args.src_dataset, mode))
        guid_index = 1
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item["str_words"]
                labels_ner = item["tags_ner"]
                labels_esi = item["tags_esi"]
                labels_net = item["tags_net"]
                entity = []
                entities = []
                co_occurs = []
                temp = 'O'
                for idx, label in enumerate(labels_net):
                    if label != 'O':
                        temp = label
                        word = words[idx]
                        entity.append(word)
                    else:
                        if len(entity) != 0:
                            entities.append([" ".join(entity), temp])
                            entity = []

                if np.array(entities).shape[0] > 1:
                    for index, entity in enumerate(entities):
                        if index != len(entities) - 1:
                            for entity_temp in entities[index + 1:]:
                                if entity[1] != entity_temp[1]:
                                    if [entity[1], entity_temp[1]] in cooccur or [entity_temp[1], entity[
                                        1]] in cooccur:  # 说明两个label之间有关联
                                        if [entity[1], entity_temp[1], 'is', 'related', 'to'] not in co_occurs and [
                                            entity_temp[1], entity[1], 'is', 'related',
                                            'to'] not in co_occurs:  # 这个关系的两种情况都没有时再加入
                                            co_occurs.append([entity[1], entity_temp[1], 'is', 'related', 'to'])
                                    else:
                                        if [entity[1], entity_temp[1], 'not', 'related', 'to'] not in co_occurs and [
                                            entity_temp[1], entity[1], 'not', 'related', 'to'] not in co_occurs:
                                            co_occurs.append([entity[1], entity_temp[1], 'not', 'related', 'to'])
                        else:
                            continue
                # print(co_occurs)
                examples_src.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels_ner,
                                             spans=labels_esi, types=labels_net, entities=entities, co_occurs=co_occurs))
                guid_index += 1

    return examples, examples_src

def maybe_show_prompt(id, word, prompt, mod):
    if id % mod == 0:
        print(colored(f"Instance {id}: {word}", "blue"))
        print(colored(f"Prompt {id}: {prompt}\n", "yellow"))

def get_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value==item]

def convert_examples_to_features(
    tag_to_id,
    examples,
    max_seq_length,
    tokenizer,
    prompt='sbert',
    template='context_all',
    prompt_or_not=False,
    graph_prompt=True,
    prompt_candidates_from_outside=None,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    show_exnum = -1,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    extra_long_samples = 0
    span_non_id = tag_to_id["span"]["O"]
    type_non_id = tag_to_id["type"]["O"]

    if prompt_or_not:
        constrained_instances = False
        if prompt_candidates_from_outside is None:
            candidates = examples
        else:
            candidates = prompt_candidates_from_outside

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

        #为每个example增加prompt， 注意最后为每个函数输入参数定义prompt
        if prompt == "sbert" or prompt == "bertscore":
            search_space = []
            search_space_dict = {}
            trans_func = None
            for inst in candidates:
                search_space.append(" ".join(inst.words))
                search_space_dict[" ".join(inst.words)] = inst
            if prompt == "sbert":
                search_model = SentenceTransformer('all-roberta-large-v1')

                def _trans_func(insts):
                    if len(insts) == 0:
                        return None
                    sub_search_space = list(map(lambda i: " ".join(i.words), insts))
                    return sub_search_space, search_model.encode(sub_search_space, convert_to_tensor=True)

                trans_func = _trans_func
                if not constrained_instances:
                    # print("===")
                    global_corpus_embeddings = search_model.encode(search_space, convert_to_tensor=True)
                    # print("---")
            # if prompt == "bertscore":
            #     bert_score_model_type = bert_score.lang2model["en"]
            #     num_layers = bert_score.model2layers[bert_score_model_type]
            #     bert_score_model = bert_score.get_model(bert_score_model_type, num_layers, False)
            #
            #     def _trans_func(insts):
            #         if len(insts) == 0:
            #             return None
            #         sub_search_space = list(map(lambda i: " ".join(i.words), insts))
            #         return sub_search_space
            #
            #     trans_func = _trans_func

            if constrained_instances:
                manager = SearchSpaceManager(examples, trans_func)

    top_k_correct_selection_count = 0
    scores = []

    num_to_examine = 10  # Number of sample prompts we want to see
    step_sz = len(examples) // num_to_examine
    account_128 = 0

    for (ex_index, example) in enumerate(examples):
        # print(ex_index)
        if ex_index % 100000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        span_label_ids = []
        type_label_ids = []
        label_mask = []
        # print(len(example.words), len(example.labels))
        for word, span_label, type_label in zip(example.words, example.spans, example.types):
            # print(word, label)
            span_label = tag_to_id["span"][span_label]
            type_label = tag_to_id["type"][type_label]
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if len(word_tokens) > 0:
                span_label_ids.extend([span_label] + [pad_token_label_id] * (len(word_tokens) - 1))
                type_label_ids.extend([type_label] + [pad_token_label_id] * (len(word_tokens) - 1))
                label_mask.extend([1] + [0]*(len(word_tokens) - 1))
            # full_label_ids.extend([label] * len(word_tokens))

        # print(len(tokens), len(label_ids), len(full_label_ids))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 2# if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:#若大于最大允许长度，则truncated，这里处理的是原输入
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            span_label_ids = span_label_ids[: (max_seq_length - special_tokens_count)]
            type_label_ids = type_label_ids[: (max_seq_length - special_tokens_count)]
            label_mask = label_mask[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        span_label_ids += [pad_token_label_id]
        type_label_ids += [pad_token_label_id]
        label_mask += [0]

        input_ids_ori = tokenizer.convert_tokens_to_ids(tokens)
        ori_mask = [1 if mask_padding_with_zero else 0] * len(input_ids_ori)
        padding_length_ori = max_seq_length - len(input_ids_ori)
        if pad_on_left:
            ori_mask = ([0 if mask_padding_with_zero else 1] * padding_length_ori) + ori_mask
        else:
            ori_mask += [0 if mask_padding_with_zero else 1] * padding_length_ori
        # print(tokens)
        # Graph prompt


        # print("mlm_label", mlm_label, len(mlm_label))
        # print("prompt_label_idx", prompt_label_idx, len(prompt_label_idx))
        # print("graph_prompt_labels", graph_prompt_labels, len(graph_prompt_labels))

        if len(tokens) > max_seq_length - 1 :#若大于最大允许长度，则truncated，这里处理的是原输入
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            span_label_ids = span_label_ids[: (max_seq_length - special_tokens_count)]
            type_label_ids = type_label_ids[: (max_seq_length - special_tokens_count)]
            label_mask = label_mask[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1

        # print(tokens)
        if prompt_or_not:
            #######################为每个instance增加prompt##########
            if prompt == "sbert":
                prompt_tokens = []
                query = " ".join(example.words)
                query_embedding = search_model.encode(query, convert_to_tensor=True)

                if constrained_instances:
                    comparison_sets = []
                    # try combined set
                    combined_set = manager.superset_labels_search_space(example)
                    if combined_set is None:
                        for lb in set(label for entity, label in inst.entities):#entities数据需要产生
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
                    # top_results = torch.topk(cos_scores, k=2)
                    # results.extend(zip(top_results[0][1].view(1), map(lambda x: search_space[x], top_results[1][1].view(1))))

                for score, selected_id in results:
                    prompt_words = search_space_dict[selected_id].words
                    prompt_entities = search_space_dict[selected_id].entities#entities确实需要产生
                    # stats
                    if prompt_candidates_from_outside is not None and ex_index % step_sz == 0:
                        print("[debug] Query: " + " ".join(example.words))
                        print("[debug] Selected: " + selected_id)
                        # print("[debug] prompt_words: " + prompt_words)
                        print("[debug] Score: %f" % score)
                        print("[debug] Prompt Entities: ", prompt_entities)
                    # if search_space_dict[selected_id] == inst:
                    #     top_k_correct_selection_count += 1
                    # print("prompt_words",prompt_words)
                    # print("selected_id",selected_id)
                    # print("example", example.words)
                    scores.append(score.item())
                    # stats end

                    if template == "context_all":
                        for i, word in enumerate(prompt_words):
                            instance_tokens = tokenizer.tokenize(" " + word)
                            for sub_token in instance_tokens:
                                prompt_tokens.append(sub_token)

                        for entity in prompt_entities:
                            # print(prompt_entities)
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

                maybe_show_prompt(ex_index, example.words, prompt_tokens, step_sz)

                if max_seq_length - len(tokens) - 2 > 0:
                    if len(prompt_tokens) > max_seq_length - len(tokens) - 2:
                        # print(prompt_tokens)
                        prompt_tokens = prompt_tokens[: (max_seq_length - len(tokens) - 2)]
                    prompt_tokens += [sep_token]
                else:
                    prompt_tokens = []
                # print("prompt_tokens", prompt_tokens)
                # print("example", example.words)
                # input_ids = tokenizer.convert_tokens_to_ids(
                #     [tokenizer.cls_token] + tokens + [tokenizer.sep_token] + prompt_tokens + [tokenizer.sep_token])

            elif prompt == "max":
                prompt_tokens = []
                for entity_label in max_entities:
                    if template in ["no_context", "context", "context_all"]:
                        if template in ["context", "context_all"]:
                            instance_words = max_entities[entity_label][1].words
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

                maybe_show_prompt(ex_index, example.words, prompt_tokens, step_sz)

                if prompt == "max":
                    if max_seq_length - len(tokens) - 1 > 0:
                        if len(prompt_tokens) > max_seq_length - len(tokens) - 1:
                            # print(prompt_tokens)
                            prompt_tokens = prompt_tokens[: (max_seq_length - len(tokens) - 1)]
                        # prompt_tokens += [sep_token]
                    else:
                        prompt_tokens = []
                else:
                    if max_seq_length - len(tokens) - 2 > 0:
                        if len(prompt_tokens) > max_seq_length - len(tokens) - 2:
                            # print(prompt_tokens)
                            prompt_tokens = prompt_tokens[: (max_seq_length - len(tokens) - 2)]
                        prompt_tokens += [sep_token]
                    else:
                        prompt_tokens = []
            ################################
        else:
            prompt_tokens = []

        # print("example_words:", example.words)
        # print("prompt_words:", prompt_words)
        # print("score:", score)
        #
        # print("tokens", tokens)
        # print("prompt_tokens", prompt_tokens)

        tokens +=  prompt_tokens
        span_label_ids += [pad_token_label_id] * len(prompt_tokens)
        type_label_ids += [pad_token_label_id] * len(prompt_tokens)
        label_mask += [0] * len(prompt_tokens)
        # print(len(tokens))
        if len(tokens)>127:
            account_128 += 1

        if graph_prompt:
            if len(example.co_occurs) > 0:
                graph_prompt_words = []
                graph_prompt_labels = []
                # print("co_occurs",example.co_occurs)
                for co in example.co_occurs:
                    if (len(tokens)+len(graph_prompt_words)) + 5 < max_seq_length - 1:
                        if co[2] == 'is':
                            graph_prompt_words.append(co[0])
                            graph_prompt_words.append(tokenizer.mask_token)
                            graph_prompt_words.append(tokenizer.mask_token)
                            graph_prompt_words.append(tokenizer.mask_token)
                            graph_prompt_words.append(co[1])
                            graph_prompt_labels.extend([0, 0, 0])
                        else:
                            graph_prompt_words.append(co[0])
                            graph_prompt_words.append(tokenizer.mask_token)
                            graph_prompt_words.append(tokenizer.mask_token)
                            graph_prompt_words.append(tokenizer.mask_token)
                            graph_prompt_words.append(co[1])
                            graph_prompt_labels.extend([1, 0, 0])
                        graph_prompt_words += [sep_token]
                    else:
                        continue
            else:
                graph_prompt_words = []
                graph_prompt_labels = []
            # MTD的graph prompt暂时加到这里，label还没加，co中的is connected to可以换成一个int来表示是否连接，反正label是固定的
        else:
            graph_prompt_words = []
            graph_prompt_labels = []

        tokens += graph_prompt_words
        span_label_ids += [pad_token_label_id] * len(graph_prompt_words)
        type_label_ids += [pad_token_label_id] * len(graph_prompt_words)
        label_mask += [0] * len(graph_prompt_words)
        # print(tokens)
        if graph_prompt:
            mlm_label = [1 if i == tokenizer.mask_token else -100 for i in tokens]

            prompt_label_idx = [list(sorted([tokenizer.convert_tokens_to_ids('is'), tokenizer.convert_tokens_to_ids('not')])),  #
                                list([tokenizer.convert_tokens_to_ids('related'), -100]),
                                list([tokenizer.convert_tokens_to_ids('to'), -100])] * len(example.co_occurs)

            mlm_label += [-100] * (max_seq_length-len(mlm_label))
            prompt_label_idx += [[-100, -100]] * (max_seq_length-len(prompt_label_idx))
            graph_prompt_labels += [-100] * (max_seq_length-len(graph_prompt_labels))
        else:
            mlm_label = []
            prompt_label_idx = []

        # example_dict = {}
        # example_dict[" ".join(example.words)] = [" ".join(prompt_words) + " ".join(graph_prompt_words), score]
        # path_example = './dataset/examples/music_1st_conll.txt'
        # with open(path_example, 'a') as f:
        #     f.write(str(example_dict))
        #     f.write('\n')

        # print("mlm_label", mlm_label)
        # print("prompt_label_idx", prompt_label_idx)
        # print("graph_prompt_labels", graph_prompt_labels)
        #
        # print(tokens)
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            span_label_ids += [pad_token_label_id]
            type_label_ids += [pad_token_label_id]
            label_mask += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            span_label_ids += [pad_token_label_id]
            type_label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            label_mask += [0]
        else:
            tokens = [cls_token] + tokens
            span_label_ids = [pad_token_label_id] + span_label_ids
            type_label_ids = [pad_token_label_id] + type_label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            label_mask += [0]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # print(len(tokens))
        # print(len(input_ids))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        # print("padding_length", padding_length)
        # print("input_ids", len(input_ids))
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            # ori_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + ori_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            span_label_ids = ([pad_token_label_id] * padding_length) + span_label_ids
            type_label_ids = ([pad_token_label_id] * padding_length) + type_label_ids
            label_mask = ([0] * padding_length) + label_mask
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            span_label_ids += [pad_token_label_id] * padding_length
            type_label_ids += [pad_token_label_id] * padding_length
            label_mask += [0] * padding_length
        
        # print(len(input_ids))
        # print(len(label_ids))
        # print(max_seq_length)
        # print("input_ids",input_ids)
        # print("tokens", tokens)
        # print("mlm_label", mlm_label)
        # print("prompt_label_idx", len(prompt_label_idx))
        # print("graph_prompt_labels", len(graph_prompt_labels))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(span_label_ids) == max_seq_length
        assert len(type_label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length
        assert len(ori_mask) == max_seq_length

        # assert len(mlm_label) == max_seq_length
        # assert len(prompt_label_idx) == max_seq_length
        # assert len(graph_prompt_labels) == max_seq_length


        # print("input_ids",input_ids)
        # print("tokens", tokens)
        # print("mlm_label", len(mlm_label))
        # print("prompt_label_idx", len(prompt_label_idx))
        # print("graph_prompt_labels", len(graph_prompt_labels))

        if ex_index < show_exnum:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("span_label_ids: %s", " ".join([str(x) for x in span_label_ids]))
            logger.info("type_label_ids: %s", " ".join([str(x) for x in type_label_ids]))
            logger.info("label_mask: %s", " ".join([str(x) for x in label_mask]))
            logger.info("ori_mask: %s", " ".join([str(x) for x in ori_mask]))
        # input_ids, input_mask, segment_ids, label_ids, full_label_ids, span_label_ids, type_label_ids
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, \
                label_ids=None, full_label_ids=None, span_label_ids=span_label_ids, type_label_ids=type_label_ids, label_mask=label_mask, ori_mask=ori_mask,
                          mlm_label=mlm_label, prompt_label_idx=prompt_label_idx, graph_prompt_labels=graph_prompt_labels)
        )
        # print(f.ori_mask for f in features)
    logger.info("Extra long example %d of %d", extra_long_samples, len(examples))
    print("长度超过128的比例为",account_128 / len(examples))
    return features

def load_and_cache_examples(args, tokenizer, pad_token_label_id, mode):

    tags_to_id = tag_to_id(args.data_dir, args.dataset)
    tags_to_id_src = tag_to_id(args.data_dir, args.src_dataset)
    
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # print(args.prompt_or_not)
    if args.prompt_or_not:
        # Load data features from cache or dataset file
        if args.proportion == 1:
            cached_features_file = os.path.join(
                args.data_dir,
                "{}_{}_{}_{}_{}.pt".format(
                    args.dataset, mode, args.prompt, args.template, args.max_seq_length
                ),
            )
        else:
            cached_features_file = os.path.join(
                args.data_dir,
                "{}_{}_{}_{}_{}_{}.pt".format(
                    args.dataset, mode, args.prompt, args.template, args.max_seq_length, args.proportion
                ),
            )

        cached_features_file_src = None

        if mode == "train":
            cached_features_file_src = os.path.join(
                args.data_dir,
                "{}_{}_{}_{}_{}.pt".format(
                    args.src_dataset, mode, args.prompt, args.template, args.max_seq_length
                ),
            )
    else:
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "{}_{}.pt".format(
                args.dataset, mode
            ),
        )

        cached_features_file_src = None

        if mode == "train":
            cached_features_file_src = os.path.join(
                args.data_dir,
                "{}_{}.pt".format(
                    args.src_dataset, mode
                ),
            )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        # print(features)
        if mode == "train":
            logger.info("Loading source domain features from cached file %s", cached_features_file_src)
            features_src = torch.load(cached_features_file_src)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples, examples_src = read_examples_from_file(args, args.data_dir, mode)
            # print("****************************")
            features = convert_examples_to_features(
                tags_to_id,
                examples,
                args.max_seq_length,
                tokenizer,
                prompt=args.prompt,
                template=args.template,
                prompt_or_not=args.prompt_or_not,
                graph_prompt=True,
                prompt_candidates_from_outside=examples_src,
                cls_token_at_end = bool(args.model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token = tokenizer.cls_token,
                cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0,
                sep_token = tokenizer.sep_token,
                sep_token_extra = bool(args.model_type in ["roberta"]),
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left = bool(args.model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id = pad_token_label_id,
            )
            # print("-----------------------------")
            features_src = convert_examples_to_features(
                tags_to_id_src,
                examples_src,
                args.max_seq_length,
                tokenizer,
                prompt=args.prompt,
                template=args.template,
                prompt_or_not=args.prompt_or_not,
                graph_prompt=True,
                cls_token_at_end = bool(args.model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token = tokenizer.cls_token,
                cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0,
                sep_token = tokenizer.sep_token,
                sep_token_extra = bool(args.model_type in ["roberta"]),
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left = bool(args.model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id = pad_token_label_id,
            )
        else:
            examples, _ = read_examples_from_file(args, args.data_dir, mode)
            prompt_candidates, _ = read_examples_from_file(args, args.data_dir, "train")
            features = convert_examples_to_features(
                tags_to_id,
                examples,
                args.max_seq_length,
                tokenizer,
                prompt=args.prompt,
                template=args.template,
                prompt_or_not=args.prompt_or_not,
                graph_prompt=False,
                prompt_candidates_from_outside=prompt_candidates,
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.model_type in ["roberta"]),
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(args.model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=pad_token_label_id,
            )

            # features_src = convert_examples_to_features(
            #     tags_to_id_src,
            #     examples_src,
            #     args.max_seq_length,
            #     tokenizer,
            #     prompt=args.prompt,
            #     template=args.template,
            #     prompt_or_not=args.prompt_or_not,
            #     prompt_candidates_from_outside=prompt_candidates_src,
            #     cls_token_at_end=bool(args.model_type in ["xlnet"]),
            #     # xlnet has a cls token at the end
            #     cls_token=tokenizer.cls_token,
            #     cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            #     sep_token=tokenizer.sep_token,
            #     sep_token_extra=bool(args.model_type in ["roberta"]),
            #     # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            #     pad_on_left=bool(args.model_type in ["xlnet"]),
            #     # pad on the left for xlnet
            #     pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            #     pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            #     pad_token_label_id=pad_token_label_id,
            # )

        
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            if mode == "train":
                logger.info("Saving features into cached file %s", cached_features_file_src)
                torch.save(features_src, cached_features_file_src)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # print(features)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_span_label_ids = torch.tensor([f.span_label_ids for f in features], dtype=torch.long)
    all_type_label_ids = torch.tensor([f.type_label_ids for f in features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long)
    all_ori_mask = torch.tensor([f.ori_mask for f in features], dtype=torch.long)
    all_prompt_label_idx = torch.tensor([f.prompt_label_idx for f in features], dtype=torch.long)
    all_mlm_label = torch.tensor([f.mlm_label for f in features], dtype=torch.long)
    all_graph_prompt_labels = torch.tensor([f.graph_prompt_labels for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_span_label_ids, all_type_label_ids, all_label_mask, all_ids, all_ori_mask, all_prompt_label_idx, all_mlm_label, all_graph_prompt_labels)
    
    dataset_src = None
    if mode == "train":
        # Convert to Tensors and build dataset
        all_input_ids_src = torch.tensor([f.input_ids for f in features_src], dtype=torch.long)
        all_input_mask_src = torch.tensor([f.input_mask for f in features_src], dtype=torch.long)
        all_segment_ids_src = torch.tensor([f.segment_ids for f in features_src], dtype=torch.long)
        all_span_label_ids_src = torch.tensor([f.span_label_ids for f in features_src], dtype=torch.long)
        all_type_label_ids_src = torch.tensor([f.type_label_ids for f in features_src], dtype=torch.long)
        all_label_mask_src = torch.tensor([f.label_mask for f in features_src], dtype=torch.long)
        all_ids_src = torch.tensor([f for f in range(len(features_src))], dtype=torch.long)
        all_ori_mask_src = torch.tensor([f.ori_mask for f in features_src], dtype=torch.long)
        all_prompt_label_idx_src = torch.tensor([f.prompt_label_idx for f in features_src], dtype=torch.long)
        all_mlm_label_src = torch.tensor([f.mlm_label for f in features_src], dtype=torch.long)
        all_graph_prompt_labels_src = torch.tensor([f.graph_prompt_labels for f in features_src], dtype=torch.long)

        dataset_src = TensorDataset(all_input_ids_src, all_input_mask_src, all_span_label_ids_src, all_type_label_ids_src, all_label_mask_src, all_ids_src, all_ori_mask_src, all_prompt_label_idx_src, all_mlm_label_src,
                                    all_graph_prompt_labels_src)
    
    return dataset_src, dataset

def get_labels(path=None, dataset_src=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        labels_ner = {}
        labels_span = {}
        labels_type = {}
        non_entity_id = None
        with open(path+dataset+"_tag_to_id.json", "r") as f:
            data = json.load(f)
            spans = data["span"]
            for l, idx in spans.items():
                labels_span[idx] = l
            types = data["type"]
            for l, idx in types.items():
                labels_type[idx] = l

        labels_type_src = {}
        with open(path+dataset_src+"_tag_to_id.json", "r") as f:
            data = json.load(f)
            # spans = data["span"]
            # for l, idx in spans.items():
            #     labels_span[idx] = l
            types = data["type"]
            for l, idx in types.items():
                labels_type_src[idx] = l

        # if "O" not in labels:
        #     labels = ["O"] + labels
        return labels_span, labels_type, labels_type_src
    else:
        return None, None, None

def tag_to_id(path=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        with open(path+dataset+"_tag_to_id.json", 'r') as f:
            data = json.load(f)
        return data # {"ner":{}, "span":{}, "type":{}}
    else:
        return None

def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq_type, seq_bio, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    assert len(seq_bio) == len(seq_type)
    spans = tags["span"]
    default = spans["O"]
    bgn = spans["B"]
    inner = spans["I"]
    idx_to_tag = {idx: tag for tag, idx in spans.items()}
    types = tags["type"]
    idx_to_type = {idx: t for t, idx in types.items()}
    chunks = []
    chunks_bio = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq_bio):
        if tok == default and chunk_start is not None:
            chunk = (chunk_start, i)
            chunks_bio.append(chunk)
            if chunk_type != "O":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
            chunk_start = None

        elif tok == bgn:
            if chunk_start is not None:
                chunk = (chunk_start, i)
                chunks_bio.append(chunk)
                if chunk_type != "O":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                chunk_start = None
            chunk_start = i
        else:
            pass
        chunk_type = idx_to_type[seq_type[i].item()]

    if chunk_start is not None:
        chunk = (chunk_start, len(seq_bio))
        chunks_bio.append(chunk)
        if chunk_type != "O":
            chunk = (chunk_type, chunk_start, len(seq_bio))
            chunks.append(chunk)
    return chunks, chunks_bio


if __name__ == '__main__':
    save(args)
