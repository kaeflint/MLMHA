from __future__ import absolute_import, division, print_function
import tensorflow as tf
# Import TensorFlow >= 1.10 and enable eager execution

#tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import time
import sys
import warnings
import six
import collections
import math

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

class UnicodeRegex(object):
  """Ad-hoc hack to recognize all punctuation and symbols."""
  def __init__(self):

    punctuation = self.property_chars("P")

    self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")

    self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")

    self.symbol_re = re.compile("([" + self.property_chars("S") + "])")



  def property_chars(self, prefix):

    return "".join(six.unichr(x) for x in range(sys.maxunicode)

                   if unicodedata.category(six.unichr(x)).startswith(prefix))





uregex = UnicodeRegex()
def bleu_tokenize(string):
    r"""Tokenize a string following the official BLEU implementation.
      See https://github.com/moses-smt/mosesdecoder/'

               'blob/master/scripts/generic/mteval-v14.pl#L954-L983

      In our case, the input string is expected to be just one line

      and no HTML entities de-escaping is needed.

      So we just tokenize on punctuation and symbols,

      except when a punctuation is preceded and followed by a digit

      (e.g. a comma/dot as a thousand/decimal separator).



      Note that a numer (e.g. a year) followed by a dot at the end of sentence

      is NOT tokenized,

      i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`

      does not match this case (unless we add a space after each sentence).

      However, this error is already in the original mteval-v14.pl

      and we want to be consistent with it.



      Args:

        string: the input string



      Returns:

        a list of tokens

      """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()

def _get_ngrams_with_counter(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in xrange(1, max_order + 1):
    for i in xrange(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts
def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
    translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

    overlap = dict((ngram,
                    min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
          ngram]

  precisions = [0] * max_order
  smooth = 1.0

  for i in xrange(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
            i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    ratio = translation_length / reference_length
    bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
  bleu = geo_mean * bp
  return np.float32(bleu)



def native_to_unicode(s):

  """Convert string to unicode (required in Python 2)."""

  try:               # Python 2

    return s if isinstance(s, unicode) else s.decode("utf-8")

  except NameError:  # Python 3

    return s





def _unicode_to_native(s):

  """Convert string from unicode to native format (required in Python 2)."""

  try:               # Python 2

    return s.encode("utf-8") if isinstance(s, unicode) else s

  except NameError:  # Python 3

    return s

def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
    ref_lines = native_to_unicode(
      tf.io.gfile.GFile(ref_filename).read()).strip().splitlines()
    hyp_lines = native_to_unicode(
      tf.io.gfile.GFile(hyp_filename).read()).strip().splitlines()
    if len(ref_lines) != len(hyp_lines):
        raise ValueError("Reference and translation files have different number of "

                     "lines. If training only a few steps (100-200), the "

                     "translation may be empty.")
    if not case_sensitive:
        ref_lines = [x.lower() for x in ref_lines]
        hyp_lines = [x.lower() for x in hyp_lines]
    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
    return compute_bleu(ref_tokens, hyp_tokens) * 100

def computeAllBleu(ref_filename, hyp_filename):
    ref_lines = native_to_unicode(
      tf.io.gfile.GFile(ref_filename).read()).strip().splitlines()
    hyp_lines = native_to_unicode(
      tf.io.gfile.GFile(hyp_filename).read()).strip().splitlines()
    if len(ref_lines) != len(hyp_lines):
        raise ValueError("Reference and translation files have different number of "

                     "lines. If training only a few steps (100-200), the "

                     "translation may be empty.")
    tokenize_sents = lambda sent:  [bleu_tokenize(x) for x in sent]
    ref_lines_uncased = tokenize_sents([x.lower() for x in ref_lines])
    hyp_lines_uncased = tokenize_sents([x.lower() for x in hyp_lines])
    
    ref_lines_cased = tokenize_sents([x for x in ref_lines])
    hyp_lines_cased = tokenize_sents([x for x in hyp_lines])
    
    return {'uncased': round(compute_bleu(ref_lines_uncased, hyp_lines_uncased) * 100,2),
            'cased': round(compute_bleu(ref_lines_cased, hyp_lines_cased) * 100,2)}