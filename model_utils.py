"""Helper functions for the Keras implementations of models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import profiler
from LearningRateSchedule import *
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

