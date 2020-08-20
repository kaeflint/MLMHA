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


## Joint Attention layer
import tensorflow as tf
class JASSAttentionLayer(tf.keras.layers.Layer):
    def maskJassLayer(self,layer,attention_weight):
        # Mask out the layer 
        mask = np.ones(shape=attention_weight.shape,dtype=np.float32)
        mask[:,layer,:,:,:]= np.zeros_like(mask[:,layer,:,:,:])
        layer_mask = tf.convert_to_tensor(mask)
        attention_weight*=layer_mask
        return attention_weight

    def __init__(self, 
                 hidden_size, 
                 num_heads, 
                 attention_dropout,
                 nb_links,
                 computation_mode=1,
                 share_weights=False,
                 share_query_weights=True,
                 return_attention=False,
                ):
        if hidden_size % num_heads:
              raise ValueError(
                  "Hidden size ({}) must be divisible by the number of heads ({})."
                  .format(hidden_size, num_heads))
        super(JASSAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.return_attention = return_attention
        self.computation_mode=computation_mode
        self.nb_links = nb_links
        self.share_weights = share_weights
        self.share_query_weights= share_query_weights
        self.is_hybrid = False
        
        print('Using normal JASS mode')
        if self.computation_mode  not in [1,2,3,4,]:
            raise ValueError(
                  "Invalid Attention computation mode: choose from [1,2,3,4]")
                  
    
    def build(self, input_shape):
        """Builds the layer."""
        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.keras.layers.Dense(
                self.hidden_size, use_bias=False, name="q") if self.share_query_weights else  [tf.keras.layers.Dense(
                self.hidden_size, use_bias=False, name="q_"+str(i)) for i in range(self.nb_links)]
        self.output_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=False, name="output_transform")
        
        if self.share_weights:
            self.k_dense_layer = tf.keras.layers.Dense(
                self.hidden_size, use_bias=False, name="k")
            self.v_dense_layer = tf.keras.layers.Dense(
                self.hidden_size, use_bias=False, name="v")
        else:
            self.k_dense_layer = [tf.keras.layers.Dense(
                self.hidden_size, use_bias=False, name="k_"+str(i)) for i in range(self.nb_links)]
            self.v_dense_layer = [tf.keras.layers.Dense(
                self.hidden_size, use_bias=False,name="v_"+str(i)) for i in range(self.nb_links)]
        super(JASSAttentionLayer, self).build(input_shape)
    def get_config(self):
        return {
            'nb_links':self.nb_links,
            'computation_mode':self.computation_mode,
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
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[3]
            x = tf.transpose(x, [0, 3, 1, 2,4])  # --> [batch, length,num_links, num_heads, depth]
            #if not self.is_hybrid:
            if self.computation_mode in [1,3]:
                    return tf.reshape(x, [batch_size, length, self.hidden_size])
            else:
                    vv= tf.reshape(x, [batch_size, length, self.hidden_size*self.nb_links]) 
                    return vv
            #else:
            #    if self.computation_mode in [1,3]:
            #        return tf.reshape(x, [batch_size, length, 2*self.hidden_size])
             #   else:
             #       vv= tf.reshape(x, [batch_size, length, 2*self.hidden_size*self.nb_links]) 
                    return vv
    
    def call(self, x, y, bias, training, cache=None,save_weights_to=None,return_attention=False):
        computation_mode = self.computation_mode
        k= y
        v= y
        if self.share_query_weights:
            q = tf.expand_dims(self.split_heads(self.q_dense_layer(x)),1) #(bs,1,nb_heads,seq_len,depth)
        else:
            q = [tf.expand_dims(self.split_heads(self.q_dense_layer[index](x)),1)  for index, vv in enumerate(k)]
            q = tf.concat(q,1) if len(q)>1 else q[-1]
        
        if self.share_weights:
            ks = [tf.expand_dims(self.split_heads(self.k_dense_layer(vv)),1)  for index, vv in enumerate(k)]
            vs = [tf.expand_dims(self.split_heads(self.v_dense_layer(vv)),1)  for index, vv in enumerate(v)]
        else:
            ks = [tf.expand_dims(self.split_heads(self.k_dense_layer[index](vv)),1)  for index, vv in enumerate(k)]
            vs = [tf.expand_dims(self.split_heads(self.v_dense_layer[index](vv)),1)  for index, vv in enumerate(v)]
        
        ks = tf.concat(ks,1) if len(ks)>1 else ks[-1] #(bs,nb_layers,nb_heads,seq_len,depth)
        vs = tf.concat(vs,1) if len(vs)>1 else vs[-1] #(bs,nb_layers,nb_heads,seq_len,depth)
        
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5
        
        # Calculate dot product attention
        logits = tf.matmul(q, ks, transpose_b=True)
        if bias is not None:
            modif_bias = tf.expand_dims(bias,1)
            modif_bias = tf.concat([modif_bias for _ in range(len(v))],1)
            logits += modif_bias
        
        if computation_mode in [1,2]:
                joint_logits = tf.expand_dims(tf.reduce_sum(logits,axis=1),1)
                joint_weight = tf.nn.softmax(joint_logits, name="j_attention_weights")
                if training:
                    joint_weight = tf.nn.dropout(joint_weight, rate=self.attention_dropout)
                weights= joint_weight
        else:
                weights = tf.nn.softmax(logits, name="attention_weights")
                if training:
                    weights = tf.nn.dropout(weights, rate=self.attention_dropout)
        attention_output = tf.matmul(weights, vs)

        if computation_mode in [1,3]:
                attention_output= tf.expand_dims(tf.reduce_sum(attention_output,1),1)

        attention_output = self.combine_heads(attention_output)
        #print(attention_output.shape)
        attention_output = self.output_dense_layer(attention_output)
        #print(attention_output.shape,'Not hybrid')
        output=None
        if not return_attention:
            output=attention_output
        else:
            batch_size,nb_links,nh,s1,s2=shape_list(logits)
            weights= tf.reshape(logits, [batch_size,s1,nh,nb_links,s2])
            output=[attention_output,weights]
        return output