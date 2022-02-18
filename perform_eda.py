import json
import random
import os
import logging
import pickle
import string
import re
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict as ddict
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import eda
import re

alpha_sr = 0.3
alpha_ri = 0.0
alpha_rs = 0.0
p_rd = 0.0
num_aug = 4

# Adapted from https://github.com/AntheaLi/cs224nProject/tree/fb08ba61f9d4d86c8e6a2a48f6cfe989a0f3a65b
# Different from the other project implementation:
# 1. No train_fraction. Do data augmentation on 100% of data.
# 2. No answer_words. Do not avoid eda operation on words from answer_dict['text']. 
#    Only avoid eda operation on stop words, which is what the original implementation does.

# same function from util.py
def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if len(qa['answers']) == 0:
                    data_dict['question'].append(question)
                    data_dict['context'].append(context)
                    data_dict['id'].append(qa['id'])
                else:
                    for answer in qa['answers']:
                        data_dict['question'].append(question)
                        data_dict['context'].append(context)
                        data_dict['id'].append(qa['id'])
                        data_dict['answer'].append(answer)

    id_map = ddict(list)
    for idx, qid in enumerate(data_dict['id']):
        id_map[qid].append(idx)

    data_dict_collapsed = {'question': [], 'context': [], 'id': []}
    if data_dict['answer']:
        data_dict_collapsed['answer'] = []
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed['question'].append(data_dict['question'][ex_ids[0]])
        data_dict_collapsed['context'].append(data_dict['context'][ex_ids[0]])
        data_dict_collapsed['id'].append(qid)
        if data_dict['answer']:
            all_answers = [data_dict['answer'][idx] for idx in ex_ids]
            data_dict_collapsed['answer'].append({'answer_start': [answer['answer_start'] for answer in all_answers],
                                                  'text': [answer['text'] for answer in all_answers]})
    return data_dict_collapsed

def clean_line(sentence):
    line = sentence.strip()
    for s in string.punctuation:
        line = line.replace(s, "")

    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    line = re.sub(' +',' ',line) #delete extra spaces

    if len(line) == 0:
        return sentence

    if line[0] == ' ':
        line = line[1:]
    return line


def shuffle_augment_data(new_data_dict_collapsed):
    num_augment_sample = len(new_data_dict_collapsed['question'])
    shuffle_idx = [i for i in range(num_augment_sample)]
    random.shuffle(shuffle_idx)
    shuffle_new_data_dict_collapsed = {'question': [], 'context': [], 'id': [], 'answer': []}
    for i in shuffle_idx:
        shuffle_new_data_dict_collapsed['question'].append(new_data_dict_collapsed['question'][i])
        shuffle_new_data_dict_collapsed['context'].append(new_data_dict_collapsed['context'][i])
        shuffle_new_data_dict_collapsed['id'].append(new_data_dict_collapsed['id'][i])
        shuffle_new_data_dict_collapsed['answer'].append(new_data_dict_collapsed['answer'][i])
    return shuffle_new_data_dict_collapsed


def data_augmentation(dataset_name, data_dict_collapsed):
    question_list = data_dict_collapsed['question']
    context_list = data_dict_collapsed['context']
    id_list = data_dict_collapsed['id']
    answer_list = data_dict_collapsed['answer']

    new_data_dict_collapsed = {'question': [], 'context': [], 'id': [], 'answer': []}

    for idx, answer_dict in enumerate(answer_list):
        text = answer_dict['text']

       # operate eda on every context
        context = clean_line(context_list[idx])
        aug_contexts = eda.eda(context, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug)
        counter = 0
        for idx_context, aug_context in enumerate(aug_contexts):
            aug_context = clean_line(aug_context)
            new_answer_dict = {'answer_start': [], 'text': []}
            for each_answer in text:
                new_each_answer = clean_line(each_answer)
                start = aug_context.find(new_each_answer)
                if start != -1:
                    new_answer_dict['answer_start'].append(start)
                    new_answer_dict['text'].append(new_each_answer)
                else:
                    counter += 1
                    print("not found original answer: ", counter)

            if len(new_answer_dict['text']) != 0:
                new_data_dict_collapsed['question'].append(clean_line(question_list[idx]))
                new_data_dict_collapsed['context'].append(aug_context)
                new_data_dict_collapsed['answer'].append(new_answer_dict)
                new_data_dict_collapsed['id'].append(str(idx_context)+"eda"+id_list[idx])

    # Save augmented data to JSON file
    save_json_file = open("datasets/eda_"+dataset_name+".json", "w+")
    save_json_file.write(json.dumps(new_data_dict_collapsed))
    save_json_file.close()

    return shuffle_augment_data(new_data_dict_collapsed)


def perform_eda(path, dataset_name):
    data_dict_collapsed = read_squad(path)
    new_data_dict_collapsed = data_augmentation(dataset_name, data_dict_collapsed)
    print("="*20)
    print("Data augmentation(eda) is finished for file ", path)
    print("Number of original samples: ", len(data_dict_collapsed['question']))
    print("Total number of samples after augmentation: ", len(new_data_dict_collapsed['question']))
    print("="*20 + "\t")
    return new_data_dict_collapsed










