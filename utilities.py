import tensorflow_datasets as tfds
import tensorflow as tf
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import time

import os
import tempfile
import subprocess
from os import system
from TransMetrics import *

import beamsV22 as beam_search
from tensorflow.python.util import nest
import beamsearchv1 as bs1

import tensorflow_datasets as tfds
import tensorflow as tf
from mosestokenizer import *
from sumeval.metrics.bleu import BLEUCalculator
import unicodedata
import re
import numpy as np
import os
import time
def to_float(x):
    """Cast x to float; created because tf.to_float is deprecated."""
    return tf.cast(x, tf.float32)
# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    return w

cleanPrep= lambda x: preprocess_sentence(x)
def computeSacreBleu(translation_path,reference_path,lang,detokenize_trans=True,detokenize_ref=False):
    bleu = BLEUCalculator(lang=lang)
    trans_raw = trans = readSentences(translation_path)
    reference_raw=reference= readSentences(reference_path)
    if detokenize_trans or detokenize_ref:
        detok = MosesDetokenizer(lang)
        
        if detokenize_trans:
            trans=[detok([d]) for d in trans_raw]
        if detokenize_ref:
            reference = [detok([d]) for d in reference_raw]
    bleu_score = bleu.bleu( summary=trans, references=[reference],score_only=True)
    print(bleu_score)
    return bleu_score
def writeToFile(content,filename):
    fil = filename+'.txt'
    if os.path.exists(fil):
        os.remove(fil)
    with open(fil,'x') as fwrite:
        fwrite.writelines("%s\n" % s for s in content)
    print('Done')
    return

def readSentences(file,lower=False):
    with open(file,'r', encoding="utf-8") as o_file:
        sentennces = []
        for s in o_file.readlines():
            ss = s.strip() #.lower() if  lower else s.strip()
            sentennces.append(ss)
    return sentennces

def getSubwordTokenizer(train_sentences,vocab_filename,vocab_size,re_generate=False):
    
    try:
        if not re_generate:
            subword_tokenizer= tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)
            print('Loading existing subword tokenizer')
        else:
            subword_tokenizer= tfds.features.text.SubwordTextEncoder.build_from_corpus(train_sentences, target_vocab_size=vocab_size)
            subword_tokenizer.save_to_file(vocab_filename)
    except:
        print('Generating the subword text tokenizer ')
        subword_tokenizer= tfds.features.text.SubwordTextEncoder.build_from_corpus(train_sentences, target_vocab_size=vocab_size)
        subword_tokenizer.save_to_file(vocab_filename)
    return subword_tokenizer

def encode(lang1, lang2,tokenizer_src,tokenizer_trg):
  
  lang1 = [tokenizer_src.vocab_size] + tokenizer_src.encode(
      lang1) + [tokenizer_src.vocab_size+1]

  lang2 = [tokenizer_trg.vocab_size] + tokenizer_trg.encode(
      lang2) + [tokenizer_trg.vocab_size+1]
  
  return lang1, lang2

def getVectorizer(tokenizer_src,MAX_LENGTH,lower_sent=True):
    pad_mm = lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                 maxlen=MAX_LENGTH ,
                                                                 padding='post')
    if lower_sent:
        
        return lambda x: pad_mm([[tokenizer_src.vocab_size] + tokenizer_src.encode(s.lower()) +[tokenizer_src.vocab_size + 1] for s in x]) if type(x) is list else pad_mm([[tokenizer_src.vocab_size] + tokenizer_src.encode(x.lower()) +[tokenizer_src.vocab_size + 1]])
    else:
        return lambda x: pad_mm([[tokenizer_src.vocab_size] + tokenizer_src.encode(s) +[tokenizer_src.vocab_size + 1] for s in x]) if type(x) is list else pad_mm([[tokenizer_src.vocab_size] + tokenizer_src.encode(x) +[tokenizer_src.vocab_size + 1]])
    
def getSentFunct(tokenizer_trg):
    start_token = tokenizer_trg.vocab_size
    end_token = tokenizer_trg.vocab_size + 1
    sent_funct = lambda x: tokenizer_trg.decode([i for i in x 
                                            if i not in [start_token,end_token,0]]) if type(x) is list else tokenizer_trg.decode([i for i in list(x) 
                                            if i not in [start_token,end_token,0]])
    return sent_funct


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)      