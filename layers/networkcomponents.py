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

class SimpleIterativeAggregatorNode(tf.keras.layers.Layer):
    def __init__(self, params):
        super(SimpleIterativeAggregatorNode, self).__init__()
        self.params = params
        self.postprocess_dropout = params["layer_postprocess_dropout"]
        
    def build(self, input_shape):
        """Builds the IterativeAggregatorNode."""
        params   = self.params
        self.ffn = FeedForwardNetwork(params["hidden_size"],params["filter_size"],params["relu_dropout"])
        self.layer_norm = LayerNormalization(self.params["hidden_size"])
        #self.layer_norm_y = LayerNormalization(self.params["hidden_size"])
        super(SimpleIterativeAggregatorNode,self).build(input_shape)
    def get_config(self):
        return {
            "params": self.params,
        }
    def call(self,inputs,training):
        if len(inputs)==1:
            return inputs[-1]
        else:
            x,y= inputs
        #x= self.layer_norm_x(x)
        #y= self.layer_norm_y(y)
        c = tf.concat([x,y],axis=-1)
        fout= self.ffn(c,training=training)  
        if training:
            fout = tf.nn.dropout(fout, rate=self.postprocess_dropout)
        output = fout + x + y
        return self.layer_norm(output)
class SimpleIterativeAggregator(tf.keras.layers.Layer):
    def __init__(self, params):
        super(SimpleIterativeAggregator, self).__init__()
        self.params = params
        self.nb_nodes = params["num_hidden_layers"]
        print('Iterative Combination')
        if params.get('top_n',None) is None:
            print('Will pick the source features from all layers')
            self.nb_nodes = params["num_hidden_layers"] 
        else:
            self.nb_nodes = params.get('top_n')-1
    def build(self, input_shape):
        """Builds the IterativeAggregatorNode."""
        params   = self.params
        self.iterative_nodes =[SimpleIterativeAggregatorNode(params) for _ in range(self.nb_nodes)]
        super(SimpleIterativeAggregator,self).build(input_shape)
    def get_config(self):
        return {
            "params": self.params,
        }
    def call(self,inputs,training):
        output = inputs[0]
        for index,vector in enumerate(inputs[1:]):
            y = vector
            node = self.iterative_nodes[index]
            output = node([output,vector],training)
        return output
class LinearAggregation(tf.keras.layers.Layer):
    def __init__(self, params):
        super(LinearAggregation, self).__init__()
        self.params = params
        print('Linear Combination')
        if params.get('top_n',None) is None:
            self.nb_links = params["num_hidden_layers"] +1
        else:
            self.nb_links = params.get('top_n')
        self.postprocess_dropout = params["layer_postprocess_dropout"]
    def build(self, input_shape):
        """Builds the IterativeAggregatorNode."""
        params   = self.params
        self.Wi= [tf.keras.layers.Dense(params["hidden_size"],use_bias=False) 
                                    for _ in range(self.nb_links)]
        super(LinearAggregation,self).build(input_shape)
    def get_config(self):
        return {
            "params": self.params,
        }
    def call(self,inputs,training):
        if len(inputs)==1:
            return self.Wi[-1](inputs[-1])
        fouts = tf.keras.layers.add([self.Wi[index](v) for index,v in enumerate(inputs)])
        return fouts