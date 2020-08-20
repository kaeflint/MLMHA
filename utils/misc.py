"""Various utility functions to use throughout the project."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import sys
import inspect
import heapq
import os
import threading
import six

from six.moves import copyreg

import numpy as np

from tensorflow.python.training.tracking import graph_view



import time
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import profiler
from .LearningRateSchedule import *
#from utilities import *
import tempfile
class BatchTimestamp(object):
  """A structure to store batch time stamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp

  def __repr__(self):
    return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
        self.batch_index, self.timestamp)


class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
    """
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps
    self.global_steps = 0

    # Logs start of step 1 then end of each step based on log_steps interval.
    self.timestamp_log = []

    # Records the time each epoch takes to run from start to finish of epoch.
    self.epoch_runtime_log = []

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_start = time.time()

  def on_batch_begin(self, batch, logs=None):
    self.global_steps += 1
    if self.global_steps == 1:
      self.start_time = time.time()
      self.timestamp_log.append(BatchTimestamp(self.global_steps,
                                               self.start_time))

  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    if self.global_steps % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
      self.timestamp_log.append(BatchTimestamp(self.global_steps, timestamp))
      tf.compat.v1.logging.info(
          "BenchmarkMetric: {'global step':%d, 'time_taken': %f,"
          "'examples_per_second': %f}" %
          (self.global_steps, elapsed_time, examples_per_second))
      self.start_time = timestamp

  def on_epoch_end(self, epoch, logs=None):
    epoch_run_time = time.time() - self.epoch_start
    self.epoch_runtime_log.append(epoch_run_time)
    tf.compat.v1.logging.info(
        "BenchmarkMetric: {'epoch':%d, 'time_taken': %f}" %
        (epoch, epoch_run_time))


def get_profiler_callback(model_dir, profile_steps, enable_tensorboard):
  """Validate profile_steps flag value and return profiler callback."""
  profile_steps_error_message = (
      'profile_steps must be a comma separated pair of positive integers, '
      'specifying the first and last steps to be profiled.'
  )
  try:
    profile_steps = [int(i) for i in profile_steps.split(',')]
  except ValueError:
    raise ValueError(profile_steps_error_message)
  if len(profile_steps) != 2:
    raise ValueError(profile_steps_error_message)
  start_step, stop_step = profile_steps
  if start_step < 0 or start_step > stop_step:
    raise ValueError(profile_steps_error_message)
  if enable_tensorboard:
    tf.compat.v1.logging.warn(
        'Both TensorBoard and profiler callbacks are used. Note that the '
        'TensorBoard callback profiles the 2nd step (unless otherwise '
        'specified). Please make sure the steps profiled by the two callbacks '
        'do not overlap.')

  return ProfilerCallback(model_dir, start_step, stop_step)


class ProfilerCallback(tf.keras.callbacks.Callback):
  """Save profiles in specified step range to log directory."""

  def __init__(self, log_dir, start_step, stop_step):
    super(ProfilerCallback, self).__init__()
    self.log_dir = log_dir
    self.start_step = start_step
    self.stop_step = stop_step

  def on_batch_begin(self, batch, logs=None):
    if batch == self.start_step:
      profiler.start()
      tf.compat.v1.logging.info('Profiler started at Step %s', self.start_step)

  def on_batch_end(self, batch, logs=None):
    if batch == self.stop_step:
      results = profiler.stop()
      profiler.save(self.log_dir, results)
      tf.compat.v1.logging.info(
          'Profiler saved profiles for steps between %s and %s to %s',
          self.start_step, self.stop_step, self.log_dir)


def set_session_config(enable_eager=False,
                       enable_xla=False):
  """Sets the session config."""
  if is_v2_0():
    set_config_v2(enable_xla=enable_xla)
  else:
    config = get_config_proto_v1(enable_xla=enable_xla)
    if enable_eager:
      tf.compat.v1.enable_eager_execution(config=config)
    else:
      sess = tf.Session(config=config)
      tf.keras.backend.set_session(sess)


def get_config_proto_v1(enable_xla=False):
  """Return config proto according to flag settings, or None to use default."""
  config = None
  if enable_xla:
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_2)
    # Disable PinToHostOptimizer in grappler when enabling XLA because it causes
    # OOM and performance regression.
    config.graph_options.rewrite_options.pin_to_host_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)
  return config


def set_config_v2(enable_xla=False):
  """Config eager context according to flag values using TF 2.0 API."""
  if enable_xla:
    tf.config.optimizer.set_jit(True)
    # Disable PinToHostOptimizer in grappler when enabling XLA because it
    # causes OOM and performance regression.
    tf.config.optimizer.set_experimental_options(
        {'pin_to_host_optimization': False}
    )


def is_v2_0():
  """Returns true if using tf 2.0."""
  return tf2.enabled()







def build_stats(history, callbacks):
    """Normalizes and returns dictionary of stats.
      Args:
        history: Results of the training step.
        callbacks: a list of callbacks which might include a time history callback
          used during keras.fit.
      Returns:
        Dictionary of normalized results.
      """
      
    stats = {}
    if history and history.history:
        train_hist = history.history
        # Gets final loss from training.
        stats['loss'] = train_hist['loss'][-1].item()

    if not callbacks:
        return stats
    
    # Look for the time history callback which was used during keras.fit
    for callback in callbacks:
        if isinstance(callback, TimeHistory):
            timestamp_log = callback.timestamp_log
            stats['step_timestamp_log'] = timestamp_log
            stats['train_finish_time'] = callback.train_finish_time
            if len(timestamp_log) > 1:
                stats['avg_exp_per_second'] = (
                    callback.batch_size * callback.log_steps *
                    (len(callback.timestamp_log)-1) /
                    (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))
    return stats
def load_weights_if_available(model,weight_path=None):
    if weight_path:
        logging.info("Load weights: {}".format(weight_path))
        try:
            model.load_weights(weight_path)
            logging.info('Weights Loaded')
        except:
            print("Weights not loaded from path:{}".format(weight_path))
            pass
    return

def get_callbacks(params):
    callbacks=[]
    time_callback = TimeHistory(params['batch_size'], params['log_steps'])
    callbacks.append(time_callback)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=params['tfboard_Logs'])
    callbacks.append(tensorboard_callback)
    #profiler_callback = get_profiler_callback(params['model_dir'],params['profile_steps'],True)
    #callbacks.append(profiler_callback)
    return callbacks


def createOptimizer(params):
    lr_schedule = LearningRateSchedule(
        params["learning_rate"], params["hidden_size"],
        params["learning_rate_warmup_steps"])
    
    opt = tf.keras.optimizers.Adam(params["learning_rate"],
                                    params["optimizer_adam_beta1"],
                                    params["optimizer_adam_beta2"],
                                    epsilon=params["optimizer_adam_epsilon"],
                                    clipnorm=params.get('grad_norm',0.0),
                                    clipvalue=params.get('grad_val',0.0)
                                  )
    #grad_norm grad_val
    return opt

def createCallabacks_old(params,init_steps=0):
    cur_log_dir=params['model_dirs']
    sfunc = LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = LearningRateScheduler(sfunc, init_steps)
    callbacks = get_callbacks(params)
    callbacks.append(scheduler_callback)
    ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
                                                        save_weights_only=True))
    return callbacks


def createCallabacks_bk(params,checkpoint_path_output,init_steps=0,no_checkpoints=False):
    cur_log_dir=checkpoint_path_output
    callbacks=[]
    
    sfunc = LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = LearningRateScheduler(sfunc, init_steps)
    #callbacks = get_callbacks(params)
    callbacks.append(scheduler_callback)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=params['tfboard_Logs'])
    callbacks.append(tensorboard_callback)
    ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
    if not no_checkpoints:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,save_weights_only=True))
    return callbacks
def createCallabacks(params,checkpoint_path_output,init_steps=0,no_checkpoints=False):
    cur_log_dir=checkpoint_path_output
    callbacks=[]
    schedule=params['lr_schedule']
    if schedule=='cos':
        print('Using cosine annealing')
        sfunc = CosineAnnealingFn(eta_max=params["max_lr"],
                                  eta_min=params['min_lr'],
                                  max_training_steps=params['train_steps'],
                                  nb_cycles=params['nb_cycles'],
                                  warmup_steps=params["learning_rate_warmup_steps"])
    else:
        print('Using normal learning rate schedule')
        sfunc = LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = LearningRateScheduler(sfunc, init_steps)
    #callbacks = get_callbacks(params)
    callbacks.append(scheduler_callback)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=params['tfboard_Logs'])
    callbacks.append(tensorboard_callback)
    ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
    if not no_checkpoints:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,save_weights_only=True))
    return callbacks

def createCallabacksUpdated(params,checkpoint_path_output,init_steps=0):
    cur_log_dir=checkpoint_path_output
    callbacks=[]
    sfunc = LearningRateFnUpdated(params["warmup_max_lr"],
                                     params["warmup_init_lr"],
                                     params["learning_rate_warmup_steps"])
    print('Using new learning rate scheduler')
    scheduler_callback = LearningRateScheduler(sfunc, init_steps)
    #callbacks = get_callbacks(params)
    callbacks.append(scheduler_callback)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=params['tfboard_Logs'])
    callbacks.append(tensorboard_callback)
    ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,save_weights_only=True))
    return callbacks




def batchIterLoc(X,batch_size,vectorizer):
    xW= X
    num_samples = len(xW)
    num_batches_per_epoch = int((num_samples-1)/batch_size)+1
    for batch_num  in range(num_batches_per_epoch):
        startIndex= batch_num*batch_size
        endIndex = min((batch_num+1)*batch_size, num_samples)
        X_batch = np.array(vectorizer(xW[startIndex:endIndex]))
        #np.array(vectorizer(inp_sentences))
        yield X_batch
def computeMultiBleu(hypothesis,references):
    fd, path_h = tempfile.mkstemp()
    fc, path_r = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.writelines("%s\n" % s for s in hypothesis)
        with os.fdopen(fc, 'w') as tmp1:
            tmp1.writelines("%s\n" % s for s in references)
        res = subprocess.check_output(
                "perl ./multi-bleu.perl %s < %s" % (path_r, path_h), shell=True).decode("utf-8")
        msg = "Performance >> " + res.strip()
        print(msg)
        #getMetrics(hypothesis=path_h,references=path_r)
    finally:
        os.remove(path_h)
        os.remove(path_r)
    return res.strip()





def get_variable_name(variable, root, model_key="model"):
  """Gets the variable name in the object-based representation."""
  named_variables, _, _ = graph_view.ObjectGraphView(root).serialize_object_graph()
  for saveable_object in named_variables:
    if saveable_object.op.name == variable.name:
      return "%s/%s" % (model_key, saveable_object.name)
  return None

def print_bytes(str_as_bytes, stream=None):
  """Prints a string viewed as bytes.

  Args:
    str_as_bytes: The bytes to print.
    stream: The stream to print to (``sys.stdout`` if not set).
  """
  if stream is None:
    stream = sys.stdout
  write_buffer = stream.buffer if hasattr(stream, "buffer") else stream
  write_buffer.write(str_as_bytes)
  write_buffer.write(b"\n")
  stream.flush()

def format_translation_output(sentence,
                              score=None,
                              token_level_scores=None,
                              attention=None,
                              alignment_type=None):
  """Formats a translation output with possibly scores, alignments, etc., e.g:

  1.123214 ||| Hello world ||| 0.30907777 0.030488174 ||| 0-0 1-1

  Args:
    sentence: The translation to output.
    score: If set, attach the score.
    token_level_scores: If set, attach the token level scores.
    attention: The attention vector.
    alignment_type: The type of alignments to format (can be: "hard", "soft").
  """
  if score is not None:
    sentence = "%f ||| %s" % (score, sentence)
  if token_level_scores is not None:
    scores_str = " ".join("%f" % s for s in token_level_scores)
    sentence = "%s ||| %s" % (sentence, scores_str)
  if attention is not None and alignment_type is not None:
    if alignment_type == "hard":
      source_indices = np.argmax(attention, axis=-1)
      target_indices = range(attention.shape[0])
      pairs = ("%d-%d" % (src, tgt) for src, tgt in zip(source_indices, target_indices))
      sentence = "%s ||| %s" % (sentence, " ".join(pairs))
    elif alignment_type == "soft":
      vectors = []
      for vector in attention:
        vectors.append(" ".join("%.6f" % value for value in vector))
      sentence = "%s ||| %s" % (sentence, " ; ".join(vectors))
    else:
      raise ValueError("Invalid alignment type %s" % alignment_type)
  return sentence

def item_or_tuple(x):
  """Returns :obj:`x` as a tuple or its single element."""
  x = tuple(x)
  if len(x) == 1:
    return x[0]
  else:
    return x

def classes_in_module(module, public_only=False):
  """Returns a generator over the classes defined in :obj:`module`."""
  return (symbol for symbol in dir(module)
          if (inspect.isclass(getattr(module, symbol))
              and (not public_only or not symbol.startswith("_"))))

def function_args(fun):
  """Returns the name of :obj:`fun` arguments."""
  if hasattr(inspect, "getfullargspec"):
    return inspect.getfullargspec(fun).args
  return inspect.getargspec(fun).args  # pylint: disable=deprecated-method

def count_lines(filename):
  """Returns the number of lines of the file :obj:`filename`."""
  with tf.io.gfile.GFile(filename, mode="rb") as f:
    i = 0
    for i, _ in enumerate(f):
      pass
    return i + 1

def is_gzip_file(filename):
  """Returns ``True`` if :obj:`filename` is a GZIP file."""
  return filename.endswith(".gz")

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.shape.dims is None:
    return tf.shape(x)

  static = x.shape.as_list()
  shape = tf.shape(x)

  ret = []
  for i, _ in enumerate(static):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def index_structure(structure, path):
  """Follows :obj:`path` in a nested structure of objects, lists, and dicts."""
  for key in path.split("/"):
    if isinstance(structure, list):
      try:
        index = int(key)
        structure = structure[index] if index < len(structure) else None
      except ValueError:
        raise ValueError("Expected a list index, got %s instead" % key)
    elif isinstance(structure, dict):
      structure = structure.get(key)
    else:
      structure = getattr(structure, key, None)
    if structure is None:
      raise ValueError("Invalid path in structure: %s" % path)
  return structure

def clone_layer(layer):
  """Clones a layer."""
  # TODO: clean this up when this change is released:
  # https://github.com/tensorflow/tensorflow/commit/4fd10c487c7e287f99b9a1831316add453dcba04
  copyreg.pickle(threading.local, lambda _: (threading.local, []))
  return copy.deepcopy(layer)

def gather_all_layers(layer):
  """Returns all nested layer starting from :obj:`layer`."""
  layers = []
  if not isinstance(layer, tf.Module):
    return layers
  layers.append(layer)
  for value in six.itervalues(layer.__dict__):
    if isinstance(value, tf.Module):
      layers.extend(gather_all_layers(value))
    elif isinstance(value, list):
      for sub_layer in value:
        layers.extend(gather_all_layers(sub_layer))
  return layers

def set_dropout(root_layer, dropout):
  """Overrides all dropout values in :obj:`root_layer` and its descendants.

  Args:
    dropout: The dropout value to set.
  """
  for layer in gather_all_layers(root_layer):
    for attr, value in six.iteritems(layer.__dict__):
      if isinstance(value, tf.keras.layers.Dropout):
        value.rate = dropout
      elif "dropout" in attr:
        setattr(layer, attr, dropout)

def extract_batches(tensors):
  """Returns a generator to iterate on each batch of a Numpy array or dict of
  Numpy arrays."""
  if not isinstance(tensors, dict):
    for tensor in tensors:
      yield tensor
  else:
    batch_size = None
    for value in six.itervalues(tensors):
      batch_size = batch_size or value.shape[0]
    for b in range(batch_size):
      yield {
          key: value[b] for key, value in six.iteritems(tensors)
      }

def extract_prefixed_keys(dictionary, prefix):
  """Returns a dictionary with all keys from :obj:`dictionary` that are prefixed
  with :obj:`prefix`.
  """
  sub_dict = {}
  for key, value in six.iteritems(dictionary):
    if key.startswith(prefix):
      original_key = key[len(prefix):]
      sub_dict[original_key] = value
  return sub_dict

def extract_suffixed_keys(dictionary, suffix):
  """Returns a dictionary with all keys from :obj:`dictionary` that are suffixed
  with :obj:`suffix`.
  """
  sub_dict = {}
  for key, value in six.iteritems(dictionary):
    if key.endswith(suffix):
      original_key = key[:-len(suffix)]
      sub_dict[original_key] = value
  return sub_dict

def merge_dict(dict1, dict2):
  """Merges :obj:`dict2` into :obj:`dict1`.

  Args:
    dict1: The base dictionary.
    dict2: The dictionary to merge.

  Returns:
    The merged dictionary :obj:`dict1`.
  """
  for key, value in six.iteritems(dict2):
    if isinstance(value, dict):
      dict1[key] = merge_dict(dict1.get(key, {}), value)
    else:
      dict1[key] = value
  return dict1

def read_summaries(event_dir, event_file_pattern="events.out.tfevents.*"):
  """Reads summaries from TensorFlow event files.

  Args:
    event_dir: Directory containing event files.
    event_file_pattern: The pattern to look for event files.

  Returns:
    A list of tuple (step, dict of summaries), sorted by step.
  """
  if not tf.io.gfile.exists(event_dir):
    return []
  summaries = collections.defaultdict(dict)
  for event_file in tf.io.gfile.glob(os.path.join(event_dir, event_file_pattern)):
    for event in tf.compat.v1.train.summary_iterator(event_file):
      if not event.HasField("summary"):
        continue
      for value in event.summary.value:
        tensor_proto = value.tensor
        tensor = tf.io.parse_tensor(
            tensor_proto.SerializeToString(), tf.as_dtype(tensor_proto.dtype))
        summaries[event.step][value.tag] = tf.get_static_value(tensor)
  return [
      (step, values)
      for step, values in sorted(six.iteritems(summaries), key=lambda x: x[0])]

class OrderRestorer(object):
  """Helper class to restore out-of-order elements in order."""

  def __init__(self, index_fn, callback_fn):
    """Initializes this object.

    Args:
      index_fn: A callable mapping an element to a unique index.
      callback_fn: A callable taking an element that will be called in order.
    """
    self._index_fn = index_fn
    self._callback_fn = callback_fn
    self._next_index = 0
    self._elements = {}
    self._heap = []

  def _try_notify(self):
    while self._heap and self._heap[0] == self._next_index:
      index = heapq.heappop(self._heap)
      value = self._elements.pop(index)
      self._callback_fn(value)
      self._next_index += 1

  def push(self, x):
    """Push event :obj:`x`."""
    index = self._index_fn(x)
    if index < self._next_index:
      raise ValueError("Event index %d was already notified" % index)
    self._elements[index] = x
    heapq.heappush(self._heap, index)
    self._try_notify()