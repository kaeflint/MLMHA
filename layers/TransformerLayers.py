import tensorflow as tf
def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)
    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret
class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout,return_attention=False):
    """Initialize Attention.
    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.return_attention = return_attention

  def build(self, input_shape):
    """Builds the layer."""
    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.keras.layers.Dense(
        self.hidden_size, use_bias=False, name="q")
    self.k_dense_layer = tf.keras.layers.Dense(
        self.hidden_size, use_bias=False, name="k")
    self.v_dense_layer = tf.keras.layers.Dense(
        self.hidden_size, use_bias=False, name="v")
    self.output_dense_layer = tf.keras.layers.Dense(
        self.hidden_size, use_bias=False, name="output_transform")
    super(Attention, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
        'return_attention':self.return_attention
    }

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, hidden_size]
    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def call(self, x, y, bias, training, cache=None,save_weights_to=None,return_attention=False):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([tf.cast(cache["k"], k.dtype), k], axis=1)
      v = tf.concat([tf.cast(cache["v"], k.dtype), v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True)
    #print(logits.shape)
    #print(bias.shape)
    if bias is not None:
        logits += bias
    #print(logits.shape)
    
    # Note that softmax internally performs math operations using float32
    # for numeric stability. When training with float16, we keep the input
    # and output in float16 for better performance.
    weights = tf.nn.softmax(logits, name="attention_weights")
    if training:
      weights = tf.nn.dropout(weights, rate=self.attention_dropout)
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    output=None
    if not return_attention:
        output=attention_output
    else:
        batch_size,nh,s1,s2=shape_list(weights)
        weights= tf.reshape(weights, [batch_size,s1,s2*nh])
        output=[attention_output,weights]
    return output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""
    def call(self, x, bias, training, cache=None,return_attention=False):
        
        return super(SelfAttention, self).call(x, x, bias, 
                                               training, 
                                               cache,
                                               return_attention=return_attention)


class FeedForwardNetwork(tf.keras.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout):
    """Initialize FeedForwardNetwork.
    Args:
      hidden_size: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
    super(FeedForwardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout

  def build(self, input_shape):
    self.filter_dense_layer = tf.keras.layers.Dense(
        self.filter_size,
        use_bias=True,
        activation=tf.nn.relu,
        name="filter_layer")
    self.output_dense_layer = tf.keras.layers.Dense(
        self.hidden_size, use_bias=True, name="output_layer")
    super(FeedForwardNetwork, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "filter_size": self.filter_size,
        "relu_dropout": self.relu_dropout,
    }

  def call(self, x, training):
    """Return outputs of the feedforward network.
    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      training: boolean, whether in training mode or not.
    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """

    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    output = self.filter_dense_layer(x)
    if training:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    output = self.output_dense_layer(output)
    return output

#class PrePostProcessingWrapper(tf.keras.layers.Layer):
class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params,eval_mode=False):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self.params = params
    self.postprocess_dropout = params["layer_postprocess_dropout"]

  def build(self, input_shape):
    # Create normalization layer
    self.layer_norm = LayerNormalization(self.params["hidden_size"])
    super(PrePostProcessingWrapper, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)
    other_outputs=[]
    if not training:
        if type(y)==list:
            #print(y,'  attention outptu')
            other_outputs=y[1:]
            y= y[0]
            
            #print('List output',self.layer,y)

    # Postprocessing: apply dropout and residual connection
    if training:
      y = tf.nn.dropout(y, rate=self.postprocess_dropout)
    combined= x +y
    output= combined
    if not training:
        if len(other_outputs)>0:
            #print([combined]+[other_outputs],' vv outputs')
            output=[combined]+other_outputs
        else:
            output=combined
    return output
        


class LayerNormalization(tf.keras.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size

  def build(self, input_shape):
    """Builds the layer."""
    # Passing experimental_autocast=False causes these variables to not be
    # automatically casted to fp16 when mixed precision is used. Since we use
    # float32 in call() for numeric stability, we do not want variables to be
    # casted to fp16.
    self.scale = self.add_weight(
        "layer_norm_scale",
        shape=[self.hidden_size],
        dtype="float32",
        initializer=tf.ones_initializer(),
        experimental_autocast=False)
    self.bias = self.add_weight(
        "layer_norm_bias",
        shape=[self.hidden_size],
        dtype="float32",
        initializer=tf.zeros_initializer(),
        experimental_autocast=False)
    super(LayerNormalization, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
    }

  def call(self, x, epsilon=1e-6):
    #print(x)    
    input_dtype = x.dtype
    if input_dtype == tf.float16:
      x = tf.cast(x, tf.float32)
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
    return tf.cast(norm_x * self.scale + self.bias, input_dtype)






class EmbeddingSharedWeights(tf.keras.layers.Layer):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size, dtype=None):
    """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
      dtype: The dtype of the layer: float16 or float32.
    """
    if dtype == tf.float16:
      # We cannot rely on the global policy of "infer_with_float32_vars", as
      # this layer is called on both int64 inputs and floating-point inputs.
      # If "infer_with_float32_vars" is used, the dtype will be inferred to be
      # int64, which means floating-point inputs would not be casted.
      # TODO(b/138859351): Remove this logic once we stop using the deprecated
      # "infer_with_float32_vars" policy
      dtype = tf.keras.mixed_precision.experimental.Policy(
          "float16_with_float32_vars")
    super(EmbeddingSharedWeights, self).__init__(dtype=dtype)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size

  def build(self, input_shape):
    """Build embedding layer."""
    with tf.name_scope("embedding_and_softmax"):
      # Create and initialize weights. The random normal initializer was chosen
      # arbitrarily, and works well.
      self.shared_weights = self.add_weight(
          "weights",
          shape=[self.vocab_size, self.hidden_size],
          dtype="float32",
          initializer=tf.random_normal_initializer(
              mean=0., stddev=self.hidden_size**-0.5))
    super(EmbeddingSharedWeights, self).build(input_shape)

  def get_config(self):
    return {
        "vocab_size": self.vocab_size,
        "hidden_size": self.hidden_size,
    }

  def call(self, inputs, mode="embedding"):
    """Get token embeddings of inputs.
    Args:
      inputs: An int64 tensor with shape [batch_size, length]
      mode: string, a valid value is one of "embedding" and "linear".
    Returns:
      outputs: (1) If mode == "embedding", output embedding tensor, float32 with
        shape [batch_size, length, embedding_size]; (2) mode == "linear", output
        linear tensor, float32 with shape [batch_size, length, vocab_size].
    Raises:
      ValueError: if mode is not valid.
    """
    if mode == "embedding":
      return self._embedding(inputs)
    elif mode == "linear":
      return self._linear(inputs)
    else:
      raise ValueError("mode {} is not valid.".format(mode))
      
  def _embedding(self, inputs):
    """Applies embedding based on inputs tensor."""
    with tf.name_scope("embedding"):
      # Create binary mask of size [batch_size, length]
      embeddings = tf.gather(self.shared_weights, inputs)
      mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
      embeddings *= tf.expand_dims(mask, -1)
      # Scale embedding by the sqrt of the hidden size
      embeddings *= self.hidden_size ** 0.5

      return embeddings

  def _linear(self, inputs):
    """Computes logits by running inputs through a linear layer.
    Args:
      inputs: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.name_scope("presoftmax_linear"):
      batch_size = tf.shape(inputs)[0]
      length = tf.shape(inputs)[1]

      x = tf.reshape(inputs, [-1, self.hidden_size])
      logits = tf.matmul(x, self.shared_weights, transpose_b=True)

      return tf.reshape(logits, [batch_size, length, self.vocab_size])


import numpy as np
import tensorflow as tf
import math
# Very low numbers to represent -infinity. We do not actually use -Inf, since we
# want to be able to multiply these values by zero to get zero. (-Inf * 0 = NaN)
_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = np.finfo(np.float16).min


def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
  """Return positional encoding.
  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.
  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position
  Returns:
    Tensor with shape [length, hidden_size]
  """
  # We compute the positional encoding in float32 even if the model uses
  # float16, as many of the ops used, like log and exp, are numerically unstable
  # in float16.
  position = tf.cast(tf.range(length), tf.float32)
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.cast(num_timescales, tf.float32) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  return signal


def get_decoder_self_attention_bias(length, dtype=tf.float32):
    """Calculate bias for decoder that maintains model's autoregressive property.
      Creates a tensor that masks out locations that correspond to illegal
      connections, so prediction at position i cannot draw information from future
      positions.
      Args:
        length: int length of sequences in batch.
        dtype: The dtype of the return value.
      Returns:
        float tensor of shape [1, 1, length, length]
     """
    neg_inf = _NEG_INF_FP16 if dtype == tf.float16 else _NEG_INF_FP32
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype),
                                         -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = neg_inf * (1.0 - valid_locs)
    return decoder_bias


def get_padding(x, padding_value=0, dtype=tf.float32):
  """Return float tensor representing the padding values in x.
  Args:
    x: int tensor with any shape
    padding_value: int value that
    dtype: The dtype of the return value.
  Returns:
    float tensor with same shape as x containing values 0 or 1.
      0 -> non-padding, 1 -> padding
  """
  with tf.name_scope("padding"):
    return tf.cast(tf.equal(x, padding_value), dtype)


def get_padding_bias(x):
  """Calculate bias tensor from padding values in tensor.
  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.
  Args:
    x: int tensor with shape [batch_size, length]
  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, length].
  """
  with tf.name_scope("attention_bias"):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF_FP32
    attention_bias = tf.expand_dims(
        tf.expand_dims(attention_bias, axis=1), axis=1)
  return attention_bias