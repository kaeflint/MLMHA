from tensorflow.python.distribute import values
import collections
import math

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import random
from data_helper_functions import *
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from tensor2tensor.data_generators import text_encoder
end_token = text_encoder.EOS_ID
tokenizer = text_encoder
def _get_sorted_inputs(filename):
    """Read and sort lines from the file sorted by decreasing length.

      Args:
        filename: String name of file to read inputs from.
      Returns:
        Sorted list of inputs, and dictionary mapping original index->sorted index
        of each element.
    """
    with tf.io.gfile.GFile(filename) as f:
        records = f.read().split("\n")
        inputs = [record.strip() for record in records]
        if not inputs[-1]:
            inputs.pop()

    input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
    sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

    sorted_inputs = [None] * len(sorted_input_lens)
    sorted_keys = [0] * len(sorted_input_lens)
    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs[i] = inputs[index]
        sorted_keys[index] = i
    return sorted_inputs, sorted_keys


def _encode_and_add_eos(line, subtokenizer):
  """Encode line with subtokenizer, and add EOS id to the end."""
  return subtokenizer.encode(line) + [text_encoder.EOS_ID]


def _trim_and_decode(ids, subtokenizer):
  """Trim EOS and PAD tokens from ids, and decode to return a string."""
  try:
    index = list(ids).index(tokenizer.EOS_ID)
    return subtokenizer.decode(ids[:index])
  except ValueError:  # No EOS found in sequence
    #print('ids: ', ids)
    return subtokenizer.decode(ids)

import time

def translate_file(model,
                   params,
                   subtokenizer,
                   input_file,
                   output_file=None,
                   print_all_translations=True,
                   distribution_strategy=None):
    """Translate lines in file, and save to output file if specified.

      Args:
        model: A Keras model, used to generate the translations.
        params: A dictionary, containing the translation related parameters.
        subtokenizer: A subtokenizer object, used for encoding and decoding source
          and translated lines.
        input_file: A file containing lines to translate.
        output_file: A file that stores the generated translations.
        print_all_translations: A bool. If true, all translations are printed to
          stdout.
        distribution_strategy: A distribution strategy, used to perform inference
          directly with tf.function instead of Keras model.predict().

      Raises:
        ValueError: if output file is invalid.
    """
    batch_size = params["decode_batch_size"]
    
    # Read and sort inputs by length. Keep dictionary (original index-->new index
    # in sorted list) to write translations in the original order.
    sorted_inputs, sorted_keys = _get_sorted_inputs(input_file)
    total_samples = len(sorted_inputs)
    num_decode_batches = (total_samples - 1) // batch_size + 1
    
    def input_generator():
        """Yield encoded strings from sorted_inputs."""
        for i in range(num_decode_batches):
            lines = [
              sorted_inputs[j + i * batch_size]
              for j in range(batch_size)
              if j + i * batch_size < total_samples
              ]
            lines = [_encode_and_add_eos(l, subtokenizer) for l in lines]
            if distribution_strategy:
                for j in range(batch_size - len(lines)):
                    lines.append([tokenizer.EOS_ID])
            batch = tf.keras.preprocessing.sequence.pad_sequences(
                  lines,
                  maxlen=params["decode_max_length"],
                  dtype="int32",
                  padding="post")
            tf.compat.v1.logging.info("Decoding batch %d out of %d.", i,
                                num_decode_batches)
            yield batch
    @tf.function
    def predict_step(inputs):
        """Decoding step function for TPU runs."""

        def _step_fn(inputs):
            """Per replica step function."""
            tag = inputs[0]
            val_inputs = inputs[1]
            val_outputs, _ = model([val_inputs], training=False)
            return tag, val_outputs
        return distribution_strategy.experimental_run_v2(_step_fn, args=(inputs,))
    translations = []
    if distribution_strategy:
        num_replicas = distribution_strategy.num_replicas_in_sync
        local_batch_size = params["decode_batch_size"] // num_replicas
    deco_times=[]
    
   #print(geneOPath+'newOutput@'+str(infID)+args.filen_pre+'n.txt')




    for i, text in enumerate(input_generator()):
        start=time.perf_counter()
        val_outputs, _ = model(text)
        val_outputs = val_outputs.numpy()
        elapsed= time.perf_counter()-start
        deco_times.append(elapsed)
        length = len(val_outputs)
        for j in range(length):
            if j + i * batch_size < total_samples:
                translation = _trim_and_decode(val_outputs[j], subtokenizer)
                translations.append(translation)
                if print_all_translations:
                    tf.compat.v1.logging.info(
                        "Translating:\n\tInput: %s\n\tOutput: %s" %
                        (sorted_inputs[j + i * batch_size], translation))
    print('Took --Secs: ',np.mean(deco_times))
    # Write translations in the order they appeared in the original file.
    if output_file is not None:
        if tf.io.gfile.isdir(output_file):
            raise ValueError("File output is a directory, will not save outputs to "
                       "file.")
        tf.compat.v1.logging.info("Writing to file %s" % output_file)
        with tf.compat.v1.gfile.Open(output_file, "w") as f:
            for i in sorted_keys:
                f.write("%s\n" % translations[i])   
                
                
                
def translate_from_text(model, subtokenizer, txt):
    encoded_txt = np.array([_encode_and_add_eos(txt, subtokenizer)])
    try:
        result = model.predict([encoded_txt])[0][0]
    except:
        result = model([encoded_txt])[0][0].numpy()
    outputs = result#["outputs"]
    tf.compat.v1.logging.info("Original: \"%s\"" % txt)
    print("Original: \"%s\"" % txt)
    print()
    translate_from_input(outputs, subtokenizer)


def translate_from_input(outputs, subtokenizer):
    translation = _trim_and_decode(outputs, subtokenizer)
    tf.compat.v1.logging.info("Translation: \"%s\"" % translation)
    print("Translation: \"%s\"" % translation)