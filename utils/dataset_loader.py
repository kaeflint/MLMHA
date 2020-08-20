import collections
import math

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import random
from .data_helper_functions import *
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from tensor2tensor.data_generators import text_encoder

def LoadorCreateVocabulary(vocab_file,dataset,vocab_size):
    try:
        subtokenizer = SubwordTextEncoder(vocab_file)
        print('Loaded existing vocabulary')
    except:
        print('Building vocabulary')
        subtokenizer = SubwordTextEncoder.build_from_generator(dataset,vocab_size)
        subtokenizer.store_to_file(vocab_file)
        print('Vocab File path: ',vocab_file)
    return subtokenizer

def loadTrainDataset(params):
    train_ds = train_input_fn(params)
    train_ds = train_ds.map(map_data_for_transformer_fn, num_parallel_calls=params["num_parallel_calls"])
    return train_ds

def loadTestDataset(params):
    test_ds = eval_input_fn(params)
    test_ds = test_ds.map(
          map_data_for_transformer_fn, num_parallel_calls=params["num_parallel_calls"])
    return test_ds

def getVectorizer(tokenizer):
    tokenizer_src =tokenizer
    vectorizer = lambda x: [tokenizer_src.encode(s) +[text_encoder.EOS_ID] for s in x] if type(x) is list else tokenizer_src.encode(x) +[text_encoder.EOS_ID]
    return vectorizer

def getSentGenerator(tokenizer):
    tokenizer_trg = tokenizer
    start_token= 0
    end_token = text_encoder.EOS_ID
    sent_funct = lambda x: tokenizer_trg.decode([i for i in x 
                                            if i not in [start_token,end_token,0]]) if type(x) is list else tokenizer_trg.decode([i for i in list(x)  if i not in [start_token,end_token,0]])
    return sent_funct
