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
import util

from BackTranslation import BackTranslation
trans = BackTranslation(url=[
      'translate.google.com',
      'translate.google.co.kr',
    ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})

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

    line = line.replace('""', " ")
    line = line.replace("'", " ")
    line = line.replace("`", " ")
    line = line.replace("\\", "")

    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    #line = line.lower()

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

    question_list = data_dict_collapsed['question']
    context_list = data_dict_collapsed['context']
    id_list = data_dict_collapsed['id']
    answer_list = data_dict_collapsed['answer']

    new_data_dict_collapsed = {'question': [], 'context': [], 'id': [], 'answer': []}

    print("back-translating for", dataset_name, "...")
    total = len(answer_list)
    for idx, answer_dict in enumerate(answer_list): # answers for each context
        print(f'{idx}/{total}')
        context = context_list[idx]
        text = answer_dict['text']
        answer_start = answer_dict['answer_start']
        
        answer_starts_sizes = [] # (starts, size)
        context_broken = [] # we do not translate answers --> remove answers from context before translation
        context_start_idx = 0
        for i, each_answer in enumerate(text):
            # {"question": "Who calls Gramps and Pud from the other side?", "id": "6d849cc2e70742a1b7d4ad20b90bba61", "answers": [{"answer_start": 2875, "text": "Granny Nellie"}, {"answer_start": 2875, "text": "Granny Nellie"}]}
            context_broken.append(context[context_start_idx:answer_start[i]])
            context_start_idx = answer_start[i] + len(each_answer) + 1
            answer_starts_sizes.append((answer_start[i], len(each_answer)))
        context_broken.append(context[context_start_idx:])

       # operate back translation on every context
        aug_contexts = []
        # trans_fr = []
        trans_es = []
        for context_part in context_broken:
            context_part = clean_line(context_part)
            if len(context_part.split()) <= 5: # incomplete phrases do not get translated -> get errors from google trans
                # trans_fr.append(clean_line(context_part))
                trans_es.append(clean_line(context_part))
            else:
                #print(context_part)
                # using chinese as media is not stable, sometimes translation fail on long and weird texts
                # -> error happens within site-packages/googletrans/client.py, hard to fix
                # -> change to french
                # google trans performs the best on spanish
                # added sleeping=1 so not to get "429" from ['translate.google.com']
                # trans_fr.append(clean_line(trans.translate(context_part, src='en', tmp = 'fr', sleeping=0.5).result_text))
                trans_es.append(clean_line(trans.translate(context_part, src='en', tmp = 'es', sleeping=1).result_text)) # sleep = 1 works, sleep = 0.5 fails
        # aug_contexts.append(trans_fr)
        aug_contexts.append(trans_es)

        # print("")
        # print("text", text)
        # print("original:", context)
        for idx_context, aug_context in enumerate(aug_contexts):

            new_answer_dict = {'answer_start': [], 'text': []}
            
            aug_context_string = ""
            for i, each_answer in enumerate(text):
                new_each_answer = clean_line(each_answer)
                aug_context_string += aug_context[i] + " "
                new_answer_dict['answer_start'].append(len(aug_context_string))
                aug_context_string += new_each_answer + " "
                new_answer_dict['text'].append(new_each_answer) 
            aug_context_string += aug_context[-1]

            # print("aug_context_string:", aug_context_string)

            new_data_dict_collapsed['question'].append(clean_line(question_list[idx]))
            new_data_dict_collapsed['context'].append(aug_context_string)
            new_data_dict_collapsed['answer'].append(new_answer_dict)
            new_data_dict_collapsed['id'].append(str(idx_context)+"translate"+id_list[idx])

    # Save augmented data to JSON file
    save_json_file = open("datasets/back_translate_"+dataset_name+".json", "w+")
    save_json_file.write(json.dumps(new_data_dict_collapsed))
    save_json_file.close()

    return shuffle_augment_data(new_data_dict_collapsed)


def perform_back_translate(args, path, dataset_name):
    data_dict_collapsed = read_squad(path)
    new_data_dict_collapsed = data_augmentation(args, dataset_name, data_dict_collapsed)

    print("="*20)
    print("Data augmentation(back translation) is finished for file ", path)
    print("Number of original samples: ", len(data_dict_collapsed['question']))

    # merge with original unaugmented
    new_data_dict_collapsed = util.merge(data_dict_collapsed, new_data_dict_collapsed)

    print("Total number of samples after augmentation: ", len(new_data_dict_collapsed['question']))
    print("="*20 + "\t")
    return new_data_dict_collapsed







