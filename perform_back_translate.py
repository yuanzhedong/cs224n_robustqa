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

# back translation using google trans
from BackTranslation import BackTranslation

trans = BackTranslation(
    url=[
        "translate.google.com",
        "translate.google.co.kr",
    ],
    proxies={"http": "127.0.0.1:1234", "http://host.name": "127.0.0.1:4012"},
)


# back translation using transformer
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)  # back translation on device: cuda:0

# src = 'en'  # source language
# trg = 'de'  # target language
# mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
# model = MarianMTModel.from_pretrained(mname).to(device).half()  # fp16 should save lots of memory
# tok = MarianTokenizer.from_pretrained(mname)
# translations = []
# for src_text_list in chunks(data, 8): # copy paste chunks fn from run_eval.py, consider wrapping tqdm_notebook
#     batch = tok.prepare_translation_batch(src_text_list).to(device)
#     gen = model.generate(**batch)
#     german: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
#     translations.extend(german)

MAX_LENGTH = 100  # on VM, otherwise CUDA out of memory, using .half() for models also to save memory

fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
fr_tokenizer = MarianTokenizer.from_pretrained(
    fr_model_name
)  # 'MarianTokenizer' object has no attribute 'to'
fr_model = MarianMTModel.from_pretrained(fr_model_name).to(device).half()
en_model_name = "Helsinki-NLP/opus-mt-fr-en"
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name).to(device).half()


def _split_segement(sentences):
    """
    Split the long sentences into multiple sentences whose lengths are less than MAX_LENGTH.
    :param sentences: the list of tokenized sentences from source text
    :return: the list of sentences with proper length
    :rtype: list
    """
    sentences_list = []
    block = ""
    for sentence in sentences:
        if len((block.rstrip() + " " + sentence).encode("utf-8")) > MAX_LENGTH:
            sentences_list.append(block.rstrip())
            block = sentence
        else:
            block = block + sentence + " "
    sentences_list.append(block.rstrip())
    return sentences_list


def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Generate translation using model
    translated = model.generate(
        **tokenizer(src_texts, return_tensors="pt", padding=True).to(device)
    )

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts


def back_translate(texts, source_lang="en", target_lang="fr"):
    texts = texts[0]  # we are only inputing ["text"]
    # check the length of text
    if len(texts) > MAX_LENGTH:
        back_translated_texts = []
        original_sentences = _split_segement(sent_tokenize(texts))
        for sentence in original_sentences:
            fr_texts = translate(
                [sentence], fr_model, fr_tokenizer, language=target_lang
            )
            back_translated = translate(
                fr_texts, en_model, en_tokenizer, language=source_lang
            )
            back_translated_texts.append(back_translated[0])
        back_text = " ".join(back_translated_texts)
        back_text.rstrip()
        return back_text
    else:
        # Translate from source to target language (fr)
        fr_texts = translate(texts, fr_model, fr_tokenizer, language=target_lang)

        # Translate from target language back to source language (en)
        back_translated_texts = translate(
            fr_texts, en_model, en_tokenizer, language=source_lang
        )

        return back_translated_texts[0]  # back_translated_texts is list of strings


##########################################################################################

# same function from util.py
def read_squad(path):
    path = Path(path)
    with open(path, "rb") as f:
        squad_dict = json.load(f)
    data_dict = {"question": [], "context": [], "id": [], "answer": []}
    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]
                if len(qa["answers"]) == 0:
                    data_dict["question"].append(question)
                    data_dict["context"].append(context)
                    data_dict["id"].append(qa["id"])
                else:
                    for answer in qa["answers"]:
                        data_dict["question"].append(question)
                        data_dict["context"].append(context)
                        data_dict["id"].append(qa["id"])
                        data_dict["answer"].append(answer)

    id_map = ddict(list)
    for idx, qid in enumerate(data_dict["id"]):
        id_map[qid].append(idx)

    data_dict_collapsed = {"question": [], "context": [], "id": []}
    if data_dict["answer"]:
        data_dict_collapsed["answer"] = []
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed["question"].append(data_dict["question"][ex_ids[0]])
        data_dict_collapsed["context"].append(data_dict["context"][ex_ids[0]])
        data_dict_collapsed["id"].append(qid)
        if data_dict["answer"]:
            all_answers = [data_dict["answer"][idx] for idx in ex_ids]
            data_dict_collapsed["answer"].append(
                {
                    "answer_start": [answer["answer_start"] for answer in all_answers],
                    "text": [answer["text"] for answer in all_answers],
                }
            )
    return data_dict_collapsed


def clean_line(sentence):
    line = sentence.strip()

    line = line.replace('""', " ")
    line = line.replace("'", " ")
    line = line.replace("`", " ")
    line = line.replace("\\", "")

    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    # line = line.lower()

    line = re.sub(" +", " ", line)  # delete extra spaces

    if len(line) == 0:
        return sentence

    if line[0] == " ":
        line = line[1:]
    return line


def shuffle_augment_data(new_data_dict_collapsed):
    num_augment_sample = len(new_data_dict_collapsed["question"])
    shuffle_idx = [i for i in range(num_augment_sample)]
    random.shuffle(shuffle_idx)
    shuffle_new_data_dict_collapsed = {
        "question": [],
        "context": [],
        "id": [],
        "answer": [],
    }
    for i in shuffle_idx:
        shuffle_new_data_dict_collapsed["question"].append(
            new_data_dict_collapsed["question"][i]
        )
        shuffle_new_data_dict_collapsed["context"].append(
            new_data_dict_collapsed["context"][i]
        )
        shuffle_new_data_dict_collapsed["id"].append(new_data_dict_collapsed["id"][i])
        shuffle_new_data_dict_collapsed["answer"].append(
            new_data_dict_collapsed["answer"][i]
        )
    return shuffle_new_data_dict_collapsed


def data_augmentation(args, dataset_name, data_dict_collapsed):

    question_list = data_dict_collapsed["question"]
    context_list = data_dict_collapsed["context"]
    id_list = data_dict_collapsed["id"]
    answer_list = data_dict_collapsed["answer"]

    new_data_dict_collapsed = {"question": [], "context": [], "id": [], "answer": []}

    print("back-translating for", dataset_name, "...")
    total = len(answer_list)
    for idx, answer_dict in enumerate(answer_list):  # answers for each context
        print(f"{idx}/{total}")
        context = context_list[idx]
        text = answer_dict["text"]
        answer_start = answer_dict["answer_start"]

        answer_starts_sizes = []  # (starts, size)
        context_broken = (
            []
        )  # we do not translate answers --> remove answers from context before translation
        context_start_idx = 0
        for i, each_answer in enumerate(text):
            # {"question": "Who calls Gramps and Pud from the other side?", "id": "6d849cc2e70742a1b7d4ad20b90bba61", "answers": [{"answer_start": 2875, "text": "Granny Nellie"}, {"answer_start": 2875, "text": "Granny Nellie"}]}
            context_broken.append(context[context_start_idx : answer_start[i]])
            context_start_idx = answer_start[i] + len(each_answer) + 1
            answer_starts_sizes.append((answer_start[i], len(each_answer)))
        context_broken.append(context[context_start_idx:])

        # operate back translation on every context
        aug_contexts = []
        back_translated = [[] for _ in args.languages]
        for context_part in context_broken:
            context_part = clean_line(context_part)
            if (
                len(context_part.split()) <= 5
            ):  # incomplete phrases do not get translated -> get errors from google trans
                for i, language in enumerate(args.languages):
                    back_translated[i].append(clean_line(context_part))
            else:
                # Google trans
                # using chinese as media is not stable, sometimes translation fail on long and weird texts
                # -> error happens within site-packages/googletrans/client.py, hard to fix
                # -> change to french
                # google trans performs the best on spanish
                # added sleeping=1 so not to get "429" from ['translate.google.com']
                for i, language in enumerate(args.languages):
                    back_translated[i].append(
                        clean_line(
                            trans.translate(
                                context_part, src="en", tmp=language, sleeping=0.5
                            ).result_text
                        )
                    )
                    # NMT
                    # back_translated[i].append(clean_line(back_translate([context_part], source_lang="en", target_lang=language)))

        for i, language in enumerate(args.languages):
            aug_contexts.append(back_translated[i])

        # print("")
        # print("text", text)
        # print("original:", context)
        for idx_context, aug_context in enumerate(aug_contexts):

            new_answer_dict = {"answer_start": [], "text": []}

            aug_context_string = ""
            for i, each_answer in enumerate(text):
                new_each_answer = clean_line(each_answer)
                aug_context_string += aug_context[i] + " "
                new_answer_dict["answer_start"].append(len(aug_context_string))
                aug_context_string += new_each_answer + " "
                new_answer_dict["text"].append(new_each_answer)
            aug_context_string += aug_context[-1]

            # print("")
            # print("aug_context_string:", aug_context_string)

            new_data_dict_collapsed["question"].append(clean_line(question_list[idx]))
            new_data_dict_collapsed["context"].append(aug_context_string)
            new_data_dict_collapsed["answer"].append(new_answer_dict)
            new_data_dict_collapsed["id"].append(
                str(idx_context) + "translate" + id_list[idx]
            )

    # Save augmented data to JSON file
    save_json_file = open("datasets/back_translate_" + dataset_name + ".json", "w+")
    save_json_file.write(json.dumps(new_data_dict_collapsed))
    save_json_file.close()

    return shuffle_augment_data(new_data_dict_collapsed)


def perform_back_translate(args, path, dataset_name):
    data_dict_collapsed = read_squad(path)
    new_data_dict_collapsed = data_augmentation(args, dataset_name, data_dict_collapsed)

    print("=" * 20)
    print("Data augmentation(back translation) is finished for file ", path)
    print("Number of original samples: ", len(data_dict_collapsed["question"]))

    # merge with original unaugmented
    new_data_dict_collapsed = util.merge(data_dict_collapsed, new_data_dict_collapsed)

    print(
        "Total number of samples after augmentation: ",
        len(new_data_dict_collapsed["question"]),
    )
    print("=" * 20 + "\t")
    return new_data_dict_collapsed
