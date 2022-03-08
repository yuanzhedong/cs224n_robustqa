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

# Adapted from https://github.com/AntheaLi/cs224nProject/tree/fb08ba61f9d4d86c8e6a2a48f6cfe989a0f3a65b
# Different from the other project implementation:
# 1. No train_fraction. Do data augmentation on 100% of data. Plus hyperparameters can be applied to augmentation to adjust for percentage.
# 2. No answer_words. Do not avoid eda operation on words from answer_dict['text']. 
#    Only avoid eda operation on stop words, which is what the original implementation does.
#    --> added it back for now for easier identification of answer starting point. will try to remove it later.

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


def data_augmentation(args, dataset_name, data_dict_collapsed):
    # parameters from the other project that uses eda for data augmentation
    # alpha_sr = 0.3
    # alpha_ri = 0.0
    # alpha_rs = 0.0
    # alpha_rd = 0.0
    # num_aug = 4

    #number of augmented sentences to generate per original sentence
    num_aug = args.num_aug

    #how much to replace each word by synonyms
    alpha_sr = args.alpha_sr

    #how much to insert new words that are synonyms
    alpha_ri = args.alpha_ri

    #how much to swap words
    alpha_rs = args.alpha_rs

    #how much to delete words
    alpha_rd = args.alpha_rd

    question_list = data_dict_collapsed['question']
    context_list = data_dict_collapsed['context']
    id_list = data_dict_collapsed['id']
    answer_list = data_dict_collapsed['answer']

    new_data_dict_collapsed = {'question': [], 'context': [], 'id': [], 'answer': []}

    found_answer_counter = 0
    lost_answer_counter = 0
    for idx, answer_dict in enumerate(answer_list):
        answer_words = set()
        text = answer_dict['text']

        # Add all words in 3 answers into the a words list, which is a word list that eda should avoid operate on, just like stop words
        for each_answer in text:
            words = clean_line(each_answer).split(" ")
            for word in words:
                if word:
                    answer_words.add(word)

       # operate eda on every context
        context = clean_line(context_list[idx])
        aug_contexts = eda.eda(context, answer_words, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug)
        for idx_context, aug_context in enumerate(aug_contexts):
            aug_context = clean_line(aug_context)
            new_answer_dict = {'answer_start': [], 'text': []}
            for each_answer in text:
                new_each_answer = clean_line(each_answer)
                start = aug_context.find(new_each_answer) # The find() method finds the first occurrence of the specified value. The find() method returns -1 if the value is not found.
                if start != -1:
                    #print("found!!!")
                    found_answer_counter += 1
                    new_answer_dict['answer_start'].append(start)
                    new_answer_dict['text'].append(new_each_answer)
                else:
                    lost_answer_counter += 1
                    #print("not found original answer: ", lost_answer_counter, each_answer, "\n", aug_context, "\n")

            if len(new_answer_dict['text']) != 0:
                new_data_dict_collapsed['question'].append(clean_line(question_list[idx]))
                new_data_dict_collapsed['context'].append(aug_context)
                new_data_dict_collapsed['answer'].append(new_answer_dict)
                new_data_dict_collapsed['id'].append(str(idx_context)+"translate"+id_list[idx])
    print("lost answer:", lost_answer_counter, "; found answer:", found_answer_counter)
    # Save augmented data to JSON file
    save_json_file = open("datasets/eda_"+dataset_name+".json", "w+")
    save_json_file.write(json.dumps(new_data_dict_collapsed))
    save_json_file.close()

    return shuffle_augment_data(new_data_dict_collapsed)


def perform_eda(args, path, dataset_name):
    data_dict_collapsed = read_squad(path)
    new_data_dict_collapsed = data_augmentation(args, dataset_name, data_dict_collapsed)
    print("="*20)
    print("Data augmentation(eda) is finished for file ", path)
    print("Number of original samples: ", len(data_dict_collapsed['question']))
    print("Total number of samples after augmentation: ", len(new_data_dict_collapsed['question']))
    print("="*20 + "\t")
    return new_data_dict_collapsed







