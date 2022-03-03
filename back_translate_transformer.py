# -*- coding: utf-8 -*-
"""back_translate_transformer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nV1n03wI2AkxXkNC931qz4aT6KwgUmRq
"""

# !pip install transformers
# !pip install sentencepiece

from transformers import MarianMTModel, MarianTokenizer

fr_model_name = 'Helsinki-NLP/opus-mt-en-fr'
fr_tokenizer = MarianTokenizer.from_pretrained(fr_model_name)
fr_model = MarianMTModel.from_pretrained(fr_model_name)

en_model_name = 'Helsinki-NLP/opus-mt-fr-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)

def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    #print(template)
    src_texts = [template(text) for text in texts]
    #print(src_texts)

    
    # Generate translation using model
    translated = model.generate(**tokenizer(src_texts, return_tensors="pt", padding=True))

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts

def back_translate(texts, source_lang="en", target_lang="fr"):
    # Translate from source to target language (fr)
    fr_texts = translate(texts, fr_model, fr_tokenizer, 
                         language=target_lang)

    # Translate from target language back to source language (en)
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer, 
                                      language=source_lang)
    
    return fr_texts, back_translated_texts

en_texts = ['This is so cool', 'I hated the food', 'They were very helpful']
aug_texts = back_translate(en_texts, source_lang="en", target_lang="fr")

print(aug_texts)