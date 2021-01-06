# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:53:44 2018

@author: kuruba
"""
import spacy
import re
from spacy.lang.en import English
from config.paths import input_path
from gensim.models import Word2Vec
from DBConnection.DBActions.get_ar_data import get_ar_data
nlp = spacy.load('en')
tokenizer = English().Defaults.create_tokenizer(nlp)


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def create_W2VModel(sentences_list, op_file_name):
    model = Word2Vec(sentences_list, size=200, window=3, min_count=1, workers=20)
    print('Training model!')
    model.train(sentences_list, total_examples=len(sentences_list), epochs=50)
    print('Model trained!')
    model.save(op_file_name)


if __name__ == '__main__':
    ar_df, df_actions, df_results = get_ar_data()
    sentences = df_actions.nltk_tokenized_raw_actions.to_list()
    sentences += df_results.nltk_tokenized_raw_results.to_list()

    print(len(sentences))
    print('Created sentences!')

    file_name = input_path + '/word_embeddings/Word2Vec_new_ar.bin'
    create_W2VModel(sentences, file_name)
