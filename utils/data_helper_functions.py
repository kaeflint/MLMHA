import six
import tensorflow as tf
import random
try:
    tf.logging= tf.compat.v1.logging
    tf.gfile = tf.compat.v1.gfile
    tf.python_io = tf.compat.v1.python_io
except:
    pass

_PREFIX = "iwslt_2014"
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from tensor2tensor.data_generators import text_encoder

def shard_filename(path, tag, shard_num, total_shards):
    """Create filename for data shard."""
    return os.path.join(
      path, "%s-%s-%.5d-of-%.5d" % (_PREFIX, tag, shard_num, total_shards))

def shard_filename(path, tag, shard_num, total_shards):
  """Create filename for data shard."""
  return os.path.join(
      path, "%s-%s-%.5d-of-%.5d" % (_PREFIX, tag, shard_num, total_shards))


def shuffle_records(fname):
  """Shuffle records in a single file."""
  tf.logging.info("Shuffling records in file %s" % fname)

  # Rename file prior to shuffling
  tmp_fname = fname + ".unshuffled"
  tf.gfile.Rename(fname, tmp_fname)

  reader = tf.compat.v1.io.tf_record_iterator(tmp_fname)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      tf.logging.info("\tRead: %d", len(records))

  random.shuffle(records)

  # Write shuffled records to original file name
  with tf.python_io.TFRecordWriter(fname) as w:
    for count, record in enumerate(records):
      w.write(record)
      if count > 0 and count % 100000 == 0:
        tf.logging.info("\tWriting record: %d" % count)

  tf.gfile.Remove(tmp_fname)


def dict_to_example(dictionary):
  """Converts a dictionary of string->int to a tf.Example."""
  features = {}
  for k, v in six.iteritems(dictionary):
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


def all_exist(filepaths):
  """Returns true if all files in the list exist."""
  for fname in filepaths:
    if not tf.gfile.Exists(fname):
      return False
  return True


def make_dir(path):
  if not tf.gfile.Exists(path):
    tf.logging.info("Creating directory %s" % path)
    tf.gfile.MakeDirs(path)

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.io.gfile.GFile(path) as f:
    for line in f:
      yield line.strip()


def encode_and_save_files(dataset_seqs, data_dir,  tag, total_shards):
    """Save data from files as encoded Examples in TFrecord format.
  Args:
    dataset_seqs: [train_src,train_trg].
    data_dir: The directory in which to write the examples
    
    tag: String that will be added onto the file names.
    total_shards: Number of files to divide the data into.
  Returns:
    List of all files produced.
    """
    try:
        os.makedirs(data_dir)
    except:
        pass
    # Create a file for each shard.
    filepaths = [shard_filename(data_dir, tag, n + 1, total_shards)
               for n in range(total_shards)]
    
    if all_exist(filepaths):
        tf.logging.info("Files with tag %s already exist." % tag)
        return filepaths
    tmp_filepaths = [fname + ".incomplete" for fname in filepaths]
    writers = [tf.compat.v1.python_io.TFRecordWriter(fname) for fname in tmp_filepaths]
    counter, shard = 0, 0
    input_file,target_file= dataset_seqs
    
    for counter, (input_line, target_line) in enumerate(zip(input_file, target_file)):
        if counter > 0 and counter % 100000 == 0:
            tf.logging.info("\tSaving case %d." % counter)
        example = dict_to_example(
        {"inputs": input_line,
         "targets":target_line,})
        writers[shard].write(example.SerializeToString())
        shard = (shard + 1) % total_shards
    for writer in writers:
        writer.close()
    for tmp_name, final_name in zip(tmp_filepaths, filepaths):
        tf.gfile.Rename(tmp_name, final_name)
    tf.logging.info("Saved %d Examples", counter + 1)
    return filepaths

def encode_and_save_file_with_max_len(
    subtokenizer, data_dir, raw_files, tag, total_shards,max_sub_word_seqs=10000000):
  """Save data from files as encoded Examples in TFrecord format.
  Args:
    subtokenizer: Subtokenizer object that will be used to encode the strings.
    data_dir: The directory in which to write the examples
    raw_files: A tuple of (input, target) data files. Each line in the input and
      the corresponding line in target file will be saved in a tf.Example.
    tag: String that will be added onto the file names.
    total_shards: Number of files to divide the data into.
  Returns:
    List of all files produced.
  """
  # Create a file for each shard.
  filepaths = [shard_filename(data_dir, tag, n + 1, total_shards)
               for n in range(total_shards)]

  if all_exist(filepaths):
    tf.logging.info("Files with tag %s already exist." % tag)
    return filepaths

  tf.logging.info("Saving files with tag %s." % tag)
  input_file = raw_files[0]
  target_file = raw_files[1]

  # Write examples to each shard in round robin order.
  tmp_filepaths = [fname + ".incomplete" for fname in filepaths]
  writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filepaths]
  counter, shard = 0, 0
  for counter, (input_line, target_line) in enumerate(zip(
      txt_line_iterator(input_file), txt_line_iterator(target_file))):
    if counter > 0 and counter % 100000 == 0:
      tf.logging.info("\tSaving case %d." % counter)
    
    #Remove very long sequences
    
    source_toks = subtokenizer.encode(input_line)+[text_encoder.EOS_ID]
    target_toks = subtokenizer.encode(target_line)+[text_encoder.EOS_ID]
    if len(target_toks)<=max_sub_word_seqs and len(source_toks)<=max_sub_word_seqs:
        example = dict_to_example(
        {"inputs": source_toks,
         "targets": target_toks })
        writers[shard].write(example.SerializeToString())
        shard = (shard + 1) % total_shards
  for writer in writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filepaths, filepaths):
    tf.gfile.Rename(tmp_name, final_name)

  tf.logging.info("Saved %d Examples", counter + 1)
  return filepaths

#Loading the dataset from the TFRecords
import math
import os


# Buffer size for reading records from a TFRecord file. Each training file is
# 7.2 MB, so 8 MB allows an entire file to be kept in memory.
_READ_RECORD_BUFFER = 8 * 1000 * 1000

# Example grouping constants. Defines length boundaries for each group.
# These values are the defaults used in Tensor2Tensor.
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def _load_records(filename):
  """Read file and return a dataset of tf.Examples."""
  return tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER)


def _parse_example(serialized_example):
  """Return inputs and targets Tensors from a serialized tf.Example."""
  data_fields = {
      "inputs": tf.io.VarLenFeature(tf.int64),
      "targets": tf.io.VarLenFeature(tf.int64)
  }
  parsed = tf.io.parse_single_example(serialized_example, data_fields)
  inputs = tf.sparse.to_dense(parsed["inputs"])
  targets = tf.sparse.to_dense(parsed["targets"])
  return inputs, targets
def _parse_example_rev(serialized_example):
    """Return inputs and targets Tensors from a serialized tf.Example."""
    # Reverse the source and target inputs. 
    # Effect: source language becomes the new target language and the original target becomes becomes the source language
    data_fields = {
      "inputs": tf.io.VarLenFeature(tf.int64),
      "targets": tf.io.VarLenFeature(tf.int64)
      }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    inputs = tf.sparse.to_dense(parsed["inputs"])
    targets = tf.sparse.to_dense(parsed["targets"])
    return targets,inputs


def _filter_max_length(example, max_length=256):
  """Indicates whether the example's length is lower than the maximum length."""
  return tf.logical_and(tf.size(example[0]) <= max_length,
                        tf.size(example[1]) <= max_length)


def _get_example_length(example):
  """Returns the maximum length between the example inputs and targets."""
  length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
  return length


def _create_min_max_boundaries(
    max_length, min_boundary=_MIN_BOUNDARY, boundary_scale=_BOUNDARY_SCALE):
  """Create min and max boundary lists up to max_length.
  For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
  returned values will be:
    buckets_min = [0, 4, 8, 16, 24]
    buckets_max = [4, 8, 16, 24, 25]
  Args:
    max_length: The maximum length of example in dataset.
    min_boundary: Minimum length in boundary.
    boundary_scale: Amount to scale consecutive boundaries in the list.
  Returns:
    min and max boundary lists
  """
  # Create bucket boundaries list by scaling the previous boundary or adding 1
  # (to ensure increasing boundary sizes).
  bucket_boundaries = []
  x = min_boundary
  while x < max_length:
    bucket_boundaries.append(x)
    x = max(x + 1, int(x * boundary_scale))

  # Create min and max boundary lists from the initial list.
  buckets_min = [0] + bucket_boundaries
  buckets_max = bucket_boundaries + [max_length + 1]
  return buckets_min, buckets_max


def _batch_examples(dataset, batch_size, max_length):
  """Group examples by similar lengths, and return batched dataset.
  Each batch of similar-length examples are padded to the same length, and may
  have different number of elements in each batch, such that:
    group_batch_size * padded_length <= batch_size.
  This decreases the number of padding tokens per batch, which improves the
  training speed.
  Args:
    dataset: Dataset of unbatched examples.
    batch_size: Max number of tokens per batch of examples.
    max_length: Max number of tokens in an example input or target sequence.
  Returns:
    Dataset of batched examples with similar lengths.
  """
  # Get min and max boundary lists for each example. These are used to calculate
  # the `bucket_id`, which is the index at which:
  # buckets_min[bucket_id] <= len(example) < buckets_max[bucket_id]
  # Note that using both min and max lists improves the performance.
  buckets_min, buckets_max = _create_min_max_boundaries(max_length)

  # Create list of batch sizes for each bucket_id, so that
  # bucket_batch_size[bucket_id] * buckets_max[bucket_id] <= batch_size
  bucket_batch_sizes = [batch_size // x for x in buckets_max]
  # bucket_id will be a tensor, so convert this list to a tensor as well.
  bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

  def example_to_bucket_id(example_input, example_target):
    """Return int64 bucket id for this example, calculated based on length."""
    seq_length = _get_example_length((example_input, example_target))

    # TODO(xunkai): investigate if removing code branching improves performance.
    conditions_c = tf.logical_and(
        tf.less_equal(buckets_min, seq_length),
        tf.less(seq_length, buckets_max))
    bucket_id = tf.reduce_min(tf.where(conditions_c))
    return bucket_id

  def window_size_fn(bucket_id):
    """Return number of examples to be grouped when given a bucket id."""
    return bucket_batch_sizes[bucket_id]

  def batching_fn(bucket_id, grouped_dataset):
    """Batch and add padding to a dataset of elements with similar lengths."""
    bucket_batch_size = window_size_fn(bucket_id)

    # Batch the dataset and add padding so that all input sequences in the
    # examples have the same length, and all target sequences have the same
    # lengths as well. Resulting lengths of inputs and targets can differ.
    return grouped_dataset.padded_batch(bucket_batch_size, ([None], [None]))

  return dataset.apply(tf.data.experimental.group_by_window(
      key_func=example_to_bucket_id,
      reduce_func=batching_fn,
      window_size=None,
      window_size_func=window_size_fn))


def _read_and_batch_from_files(
    file_pattern, batch_size, max_length, num_parallel_calls, shuffle, repeat,
    static_batch=False, num_replicas=1,rev=False):
    """Create dataset where each item is a dict of "inputs" and "targets".
  Args:
    file_pattern: String used to match the input TFRecord files.
    batch_size: Maximum number of tokens per global batch of examples.
    max_length: Maximum number of tokens per example
    num_parallel_calls: Number of cpu cores for parallel input processing.
    shuffle: If true, randomizes order of elements.
    repeat: Number of times to repeat the dataset. If None, the dataset is
      repeated forever.
    static_batch: Whether the batches in the dataset should have static shapes.
      If True, the input is batched so that every batch has the
      shape [batch_size // max_length, max_length]. If False, the input is
      grouped by length, and batched so that batches may have different
      shapes [N, M], where:
        N * M <= batch_size
        M <= max_length
      In general, this setting should be False. Dynamic shapes allow the inputs
      to be grouped so that the number of padding tokens is minimized, and helps
      model training. In cases where the input shape must be static
      (e.g. running on TPU), this setting should be set to True.
    num_replicas: Number of GPUs or other workers. We will generate global
      batches, and each global batch is equally divisible by number of replicas.
      Currently it is only effective when static_batch==True. TODO: make it
      effective when static_batch=False.
  Returns:
    tf.data.Dataset object containing examples loaded from the files.
  """
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    # Read files and interleave results. When training, the order of the examples
    # will be non-deterministic.
    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.interleave(
      _load_records,
      cycle_length=num_parallel_calls,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)
    # Parse each tf.Example into a dictionary
    # TODO: Look into prefetch_input_elements for performance optimization.
    if not rev:
        dataset = dataset.map(_parse_example,
                        num_parallel_calls=num_parallel_calls)
    else:
        dataset = dataset.map(_parse_example_rev,
                        num_parallel_calls=num_parallel_calls)
    # Remove examples where the input or target length exceeds the maximum length,
    dataset = dataset.filter(lambda x, y: _filter_max_length((x, y), max_length))
    if static_batch:
        dataset = dataset.padded_batch(
            # First calculate batch size (token number) per worker, then divide it
            # into sentences, and finally expand to a global batch. It could prove
            # the global batch divisble for distribution strategy.
            ((batch_size // num_replicas) // max_length) * num_replicas,
            ([max_length], [max_length]), drop_remainder=True)
    else:
        # Group and batch such that each batch has examples of similar length.
        # TODO(xunkai): _batch_examples might need to do something special for
        # num_replicas.
        dataset = _batch_examples(dataset, batch_size, max_length)

    dataset = dataset.repeat(repeat)
    # Prefetch the next element to improve speed of input pipeline.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def _generate_synthetic_data(params):
  """Create synthetic data based on the parameter batch size."""
  batch = length = int(math.sqrt(params["batch_size"]))
  dataset = model_helpers.generate_synthetic_data(
      input_shape=tf.TensorShape([length]),
      input_value=1,
      input_dtype=tf.int64,
      label_shape=tf.TensorShape([length]),
      label_value=1,
      label_dtype=tf.int64,
  )
  return dataset.batch(batch, drop_remainder=True)


def train_input_fn(params):
    """Load and return dataset of batched examples for use during training."""
    file_pattern = os.path.join(params["data_dir"] or "", "*train*")
    if params["use_synthetic_data"]:
        return _generate_synthetic_data(params)
    rev = params.get('rev',False)
    return _read_and_batch_from_files(
      file_pattern, params["batch_size"], params["max_length"],
      params["num_parallel_calls"], shuffle=True,
      repeat=params["repeat_dataset"], static_batch=params["static_batch"],
      num_replicas=params["num_gpus"],rev=rev)

def train_input_fn1(params,file_p):
  """Load and return dataset of batched examples for use during training."""
  file_pattern = os.path.join(params["data_dir"] or "", file_p+"-train*")
  if params["use_synthetic_data"]:
    return _generate_synthetic_data(params)
  return _read_and_batch_from_files(
      file_pattern, params["batch_size"], params["max_length"],
      params["num_parallel_calls"], shuffle=True,
      repeat=params["repeat_dataset"], static_batch=params["static_batch"],
      num_replicas=params["num_gpus"])

def eval_input_fn(params):
    """Load and return dataset of batched examples for use during evaluation."""
    file_pattern = os.path.join(params["data_dir"] or "", "*dev*")
    if params["use_synthetic_data"]:
        return _generate_synthetic_data(params)
    rev = params.get('rev',False)
    return _read_and_batch_from_files(
      file_pattern, params["batch_size"], params["max_length"],
      params["num_parallel_calls"], shuffle=False, repeat=1,
      static_batch=params["static_batch"],
        num_replicas=params["num_gpus"],
    rev=rev,
    )


def map_data_for_transformer_fn(x, y,rev=False):
    """Maps data for training, and handles weried behaviors for different vers."""
    # Will transform input x and targets y into tuple(x, y) as new model inputs.
    return ((x, y),)
    if misc.is_v2():
        # For TF v2, the 2nd parameter is omitted to make Keras training work.
        return ((x, y),)
    else:
        # For TF v1, Keras requires a dummy placeholder as the 2nd parameter.
        return ((x, y), tf.constant(0.0))

    
    
from collections import defaultdict


BASE_PARAMS = defaultdict(
    lambda: None,  # Set default value to None.

    # Input params
    default_batch_size=2048,  # Maximum number of tokens per batch of examples.
    default_batch_size_tpu=32768,
    max_length=256,  # Maximum number of tokens per example.

    # Model params
    initializer_gain=1.0,  # Used in trainable variable initialization.
    vocab_size=33708,  # Number of tokens defined in the vocabulary file.
    hidden_size=512,  # Model dimension in the hidden layers.
    num_hidden_layers=6,  # Number of layers in the encoder and decoder stacks.
    num_heads=8,  # Number of heads to use in multi-headed attention.
    filter_size=2048,  # Inner layer dimension in the feedforward network.

    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    # Training params
    label_smoothing=0.1,
    learning_rate=2.0,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

    # Default prediction params
    extra_decode_length=50,
    beam_size=4,
    alpha=0.6,  # used to calculate length normalization in beam search

    # TPU specific parameters
    use_tpu=False,
    static_batch=False,
    allow_ffn_pad=True,
)

BIG_PARAMS = BASE_PARAMS.copy()
BIG_PARAMS.update(
    default_batch_size=4096,

    # default batch size is smaller than for BASE_PARAMS due to memory limits.
    default_batch_size_tpu=16384,

    hidden_size=1024,
    filter_size=4096,
    num_heads=16,
)

# Parameters for running the model in multi gpu. These should not change the
# params that modify the model shape (such as the hidden_size or num_heads).
BASE_MULTI_GPU_PARAMS = BASE_PARAMS.copy()
BASE_MULTI_GPU_PARAMS.update(
    learning_rate_warmup_steps=8000
)

BIG_MULTI_GPU_PARAMS = BIG_PARAMS.copy()
BIG_MULTI_GPU_PARAMS.update(
    layer_postprocess_dropout=0.3,
    learning_rate_warmup_steps=8000
)

# Parameters for testing the model
TINY_PARAMS = BASE_PARAMS.copy()
TINY_PARAMS.update(
    default_batch_size=1024,
    default_batch_size_tpu=1024,
    hidden_size=32,
    num_heads=4,
    filter_size=256,
)