
#from utilities import *
#from data_helper_functions import *
#from Seq2seqMetrics import *
#from model_utils import *
#from  beamsV22 import *
'''
Instead of performing the source-target attention across the top-level encoder layer, 
We propose a Transformer model with Multi-layer Multi-head Attention (MLMHA)
MLMAH employs different joint attention strategies to leverage the multiple source representations
Approach:
    The encoder returns the source representations from the top_n layer.
    These are then passed to the decoding subnetwork where the source-target attention operations 
    are jointly performed across the representations from encoder subnetwork. The goal is to provide the
    decoder a more direct access to the encoding layers which improves the flow of gradient information.
'''
from layers.attention import *
from layers.networkcomponents import *
from utils.beamsV22 import *
from utils.Seq2seqMetrics import *
from utils.misc import *
import copy
def new_expand_dims(a,axes):
    if type(axes) == int:
        return tf.expand_dims(a, axes)
    else:
        for ax in sorted(axes):
            a = tf.expand_dims(a, ax)
        return a
class EncoderStack(tf.keras.layers.Layer):
    """Transformer encoder stack.
      The encoder stack is made up of N identical layers. Each layer is composed
      of the sublayers:
        1. Self-attention layer
        2. Feedforward network (which is 2 fully-connected layers)
      """

    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params
        self.layers = []
        #The number of source representations to return to the decoder subnetwork
        self.last_n = params.get('top_n',1)
        if params['top_n'] is None:
            print('Will pick the source features from all layers')
            self.last_n = params["num_hidden_layers"]
        #self.aux_layer_indices = aux_layer_indices

        num_layers = params["num_hidden_layers"]
        self.attention_layers_weights={}
        self.multi_attention_mode = params.get('jass_mode',1)

    def build(self, input_shape):
        """Builds the encoder stack."""
        params = self.params
        #Create the final normalization layer for each vector within the range determined by self.last_n
        self.output_normalizations=[LayerNormalization(params["hidden_size"]) for _ in range(self.last_n)]
        for _ in range(params["num_hidden_layers"]):
            # Create sublayers for each layer.
            self_attention_layer = SelfAttention(
              params["hidden_size"], params["num_heads"],
              params["attention_dropout"])
            feed_forward_network = FeedForwardNetwork(
              params["hidden_size"], params["filter_size"], params["relu_dropout"])
            self.layers.append([
              PrePostProcessingWrapper(self_attention_layer, params),
              PrePostProcessingWrapper(feed_forward_network, params)])
        super(EncoderStack, self).build(input_shape)
        
    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, encoder_inputs, attention_bias, inputs_padding, training,return_attention=False):
        """Return the output of the encoder layer stacks.
        Args:
          encoder_inputs: tensor with shape [batch_size, input_seq_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
            1, input_seq_length]
          inputs_padding: tensor with shape [batch_size, input_seq_lengthh], inputs with
            zero paddings.
          training: boolean, whether in training mode or not.
        Returns:
          Output of encoder layer stack.
          top_n float32 tensors each  with shape [batch_size, input_seq_length, hidden_size]
        """
        outputs = [encoder_inputs]
        ##Initialize the self attention dictionary
        num_layers =self.params["num_hidden_layers"]
        #encoder_inputs
        att_batch_size, enc_seq_length = shape_list(encoder_inputs)[0:2]
        self.attention_layers_weights={}
        params = self.params
        if return_attention:
            for layer in range(num_layers):
                self.attention_layers_weights["layer_%d" % layer] = tf.zeros(
                    [att_batch_size, self.params["num_heads"], enc_seq_length, enc_seq_length])
        
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]
            layer_name = "layer_%d" % n
            with tf.name_scope("layer_%d" % n):
                with tf.name_scope("self_attention"):
                    if not return_attention:
                        encoder_inputs = self_attention_layer(
                            encoder_inputs, attention_bias, training=training,return_attention=False)
                    else:
                        layer_out = self_attention_layer(
                            encoder_inputs, attention_bias, training=training,return_attention=True)
                        encoder_inputs=layer_out[0]
                        atten_weights= layer_out[-1]
                        batch_size,s1,s2=shape_list(atten_weights)
                        weights= tf.reshape(atten_weights, [batch_size,params["num_heads"],s1,s2//params["num_heads"]])
                        self.attention_layers_weights[layer_name] = weights
                with tf.name_scope("ffn"):
                    encoder_inputs = feed_forward_network(encoder_inputs, training=training)
                
                #if  n in self.choices:
                outputs.append(encoder_inputs)
        #take the top-n encoder layer output
        outputs = outputs[-1*self.last_n:]
        output_vectors =[self.output_normalizations[i](e) for i,e in enumerate(outputs)]
        return output_vectors



class DecoderStack(tf.keras.layers.Layer):
    """Transformer decoder stack.
      Like the encoder stack, the decoder stack is made up of N identical layers.
      Each layer is composed of the sublayers:
        1. Self-attention layer
        2. Multi-headed attention layer combining encoder outputs with results from
           the previous self-attention layer.
        3. Feedforward network (2 fully-connected layers)
      """
    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params
        self.layers = []
        self.last_n = params['top_n']
       
        self.multi_attention_mode = params.get('jass_mode',3)
        if self.multi_attention_mode in [5,6]:
            #Will use thesame computation mechanism for the JASS mode 3
            self.multi_attention_mode = 3
            params['top_n'] = 1
        if params['top_n'] is None:
            self.last_n = params["num_hidden_layers"]
        cc= self.last_n 
        print('Number of source Features: ',cc)
        
    def build(self, input_shape):
        """Builds the decoder stack."""
        params = self.params
        
        for _ in range(params["num_hidden_layers"]):
            self_attention_layer = SelfAttention(
              params["hidden_size"], params["num_heads"],
              params["attention_dropout"])
            #Since multiple source representations are returned by the encoder
            enc_dec_attention_layer = JASSAttentionLayer(
              params["hidden_size"], 
                params["num_heads"],
              params["attention_dropout"],
                nb_links=self.last_n ,
                computation_mode= self.multi_attention_mode,
                share_weights = params.get('share_weights',False),
                share_query_weights=  params.get('share_query_weights',False),
                is_hybrid= params.get('is_hybrid',False),)
            feed_forward_network = FeedForwardNetwork(
              params["hidden_size"], params["filter_size"], params["relu_dropout"])
            self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(enc_dec_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
                  ])
        self.output_normalization = LayerNormalization(params["hidden_size"])
        super(DecoderStack, self).build(input_shape)
    def get_config(self):
        return {
            "params": self.params,
        }
    def call(self,
           decoder_inputs,
           encoder_outputs,
           decoder_self_attention_bias,
           attention_bias,
           training,
           cache=None,
            return_attention=False,
            ):
        """Return the output of the decoder layer stacks.
        Args:
          decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
          encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
          decoder_self_attention_bias: bias for decoder self-attention layer. [1, 1,
            target_len, target_length]
          attention_bias: bias for encoder-decoder attention layer. [batch_size, 1,
            1, input_length]
          training: boolean, whether in training mode or not.
          cache: (Used for fast decoding) A nested dictionary storing previous
            decoder self-attention values. The items are:
              {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                         "v": tensor with shape [batch_size, i, value_channels]},
                           ...}
        Returns:
          Output of decoder layer stack.
          float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        params = self.params
        attention_weights= None
        if cache is not None and return_attention:
            try:
                attention_weights= cache["attention_history"]
            except:
                attention_weights= None
            cur_attention = []
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]
            
            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.name_scope(layer_name):
                with tf.name_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                                      decoder_inputs,
                                      decoder_self_attention_bias,
                                      training=training,
                                      cache=layer_cache)
                with tf.name_scope("encdec_attention"):
                    if attention_weights is not None and return_attention:
                        layer_out = enc_dec_attention_layer(
                                          decoder_inputs,
                                          encoder_outputs,
                                          attention_bias,
                                          training=training,
                            return_attention=return_attention)
                        #print(layer_out)
                        decoder_inputs=layer_out[0]
                        atten_weights= layer_out[-1]
                        batch_size,s1,nh,nb_links,s2=shape_list(atten_weights)
                        #print(batch_size,s1,s2)
                        weights= tf.reshape(atten_weights, [batch_size,nb_links,params["num_heads"],s1,-1])
                        if attention_weights is not None:
                            cache["attention_history"][layer_name] = tf.concat([tf.cast(cache["attention_history"][layer_name]
                                                                                        ,weights.dtype),weights], axis=3)
                    else:
                        #print(decoder_inputs.shape)
                        decoder_inputs = enc_dec_attention_layer(
                      decoder_inputs,
                      encoder_outputs,
                      attention_bias,
                      training=training)
                        
                with tf.name_scope("ffn"):
                    decoder_inputs = feed_forward_network(
                        decoder_inputs, training=training)
        return self.output_normalization(decoder_inputs)


class JASSTransformer(tf.keras.layers.Layer):
    def __init__(self,params,
                 name=None):
        super(JASSTransformer, self).__init__(name=name)
        self.params = params
        self.return_all = True
        self.curModel_index = -1
        #Attention weights will be saved in this dictionary
        
        num_layers = params["num_hidden_layers"]
        self.embedding_softmax_layer = EmbeddingSharedWeights(params["vocab_size"], 
                                                              params['d_model'], dtype=params['dtype'])
        
        #Define the encoding subnetwork
        self.encoder_stack = EncoderStack(params) 
        self.params["hidden_size"] = params['d_model']
        
        
        self.attention_weights = {} #Keeps track of the attention weights for visualization during the target generation
        #Define the decoder subnetwork
        decoder_params = copy.deepcopy(params)
        self.decoder_stack = DecoderStack(decoder_params)
    
  
    def get_config(self):
        #print(self.aux_layer_indices)
        return {"params": self.params,
               }
    def build(self, input_shape):
        """Builds the decoder stack."""
        params = self.params
        self.built=True
        super(JASSTransformer, self).build(input_shape)
    def getEncoderOutputs(self,inputs,training=False):
        with tf.name_scope("Jass_Transformer_encoder"):
            attention_bias = get_padding_bias(inputs)
            encoder_outputs = self.encode(inputs, attention_bias, training,return_attention=True)
            return encoder_outputs
    
    def encode(self, inputs, attention_bias, training,return_attention=False):
        with tf.name_scope("jass_encode"):
            embedded_inputs = self.embedding_softmax_layer(inputs)
            embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])
            inputs_padding = get_padding(inputs)
            attention_bias = tf.cast(attention_bias, self.params["dtype"])
            
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = get_position_encoding( length, self.params["hidden_size"])
                pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
                encoder_inputs = embedded_inputs + pos_encoding
            if training:
                encoder_inputs = tf.nn.dropout(encoder_inputs, rate=self.params["layer_postprocess_dropout"])
            
            return self.encoder_stack(encoder_inputs,
                                      attention_bias, 
                                      inputs_padding, 
                                      training=training,
                                      return_attention=return_attention)
    def decode(self, targets, encoder_outputs, attention_bias, training):
        with tf.name_scope("jass_decode"):
            decoder_inputs = self.embedding_softmax_layer(targets)
            decoder_inputs = tf.cast(decoder_inputs, self.params['dtype'])
            attention_bias = tf.cast(attention_bias, self.params["dtype"])
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(decoder_inputs,[[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                pos_encoding = get_position_encoding(
                    length, self.params["hidden_size"])
                pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
                decoder_inputs += pos_encoding
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self.params["layer_postprocess_dropout"])
            
            # Run values
            decoder_self_attention_bias = get_decoder_self_attention_bias(length, dtype=self.params['dtype'])
            #print(self.decoder_stacks,len(encoder_outputs))
            decoderstack = self.decoder_stack
            outputs = decoderstack(decoder_inputs,encoder_outputs,
                                     decoder_self_attention_bias,
                                   attention_bias,
                                    training=training,
                                  return_attention=False,
                                   cache=None,
                                  ) 
            logits = tf.cast(self.embedding_softmax_layer(outputs, mode="linear"), tf.float32)
            #if self.params['use_layer_diversity'] and divLoss is  not None:
            #    return logits,divLoss
            #else:
            return logits
    def forcedDecode(self,targets,encoder_outputs,attention_bias,training=False):
        batch_size = tf.shape(encoder_outputs[0])[0]
        input_length = tf.shape(encoder_outputs[0])[1]
        max_decode_length = input_length + self.params["extra_decode_length"]
        att_cache={"attention_history": {}}
        num_layers =self.params["num_hidden_layers"]
        att_batch_size, enc_seq_length = shape_list(encoder_outputs[0])[0:2]
        for layer in range(num_layers):
            att_cache["attention_history"]["layer_%d" % layer] = tf.zeros(
            [att_batch_size,len(encoder_outputs), self.params["num_heads"], 0, enc_seq_length])
            self.attention_weights["layer_%d" % layer] = tf.zeros(
            [att_batch_size,len(encoder_outputs), self.params["num_heads"], 0, enc_seq_length])
        cache = {
        "layer_%d" % layer: {
            "k": tf.zeros([batch_size, 0, self.params["hidden_size"]],
                          dtype=self.params["dtype"]),
            "v": tf.zeros([batch_size, 0, self.params["hidden_size"]],
                          dtype=self.params["dtype"])
        } for layer in range(self.params["num_hidden_layers"])}
        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = attention_bias
        cache["attention_history"]=att_cache["attention_history"]
        decoderstack= self.decoder_stack
        
        with tf.name_scope("collaborative_force_decode"):
            decoder_inputs = self.embedding_softmax_layer(targets)
            decoder_inputs = tf.cast(decoder_inputs, self.params['dtype'])
            attention_bias = tf.cast(attention_bias, self.params["dtype"])
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(decoder_inputs,[[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                pos_encoding = get_position_encoding(
                    length, self.params["hidden_size"])
                pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
                decoder_inputs += pos_encoding
            decoder_self_attention_bias = self_attention_bias = get_decoder_self_attention_bias(length, dtype=self.params['dtype'])
            
            decoder_outputs = decoderstack(decoder_inputs,cache.get("encoder_outputs"),
                                                               self_attention_bias,
              cache.get("encoder_decoder_attention_bias"),
              training=False,
              cache=cache,return_attention=True,)
            logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
            logits = logits#tf.squeeze(logits, axis=[1])
            try:
                for name,value in self.attention_weights.items():
                    #print(cache["attention_history"][name].shape)
                    self.attention_weights[name]=cache["attention_history"][name]
            
            #try:
            #    update_decoder_attention_history(cache)
            except:
                print('Error saving the attention weights')
            return logits#, cache
    
    def call(self, inputs, 
             training,
             beam_search_dict={"beam_size":4,
                               "alpha":0.6,},
             reverse=False,
             force_decode=False):
        # if reverse, then the translation will be done with the output as the source and inputs as the target
        # Simply swap the 2
        if len(inputs) == 2:
            
            inputs, targets = inputs[0], inputs[1]
            
        else:
            inputs, targets = inputs[0], None
        with tf.name_scope("JASS_Transformer"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = get_padding_bias(inputs)
            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs= self.encode(inputs, attention_bias, training)
            # Generate output sequence if targets is None, or return logits if target is known
            if force_decode:
                #To check the quality of the attention layer
                encoder_output_from_i= encoder_outputs
                #forcedDecode(self,targets,encoder_outputs,attention_bias,training=False,model_index=-1):
                logits = self.forcedDecode(targets, encoder_outputs, attention_bias, 
                                           training=False,)
                return logits
            else:
                if targets is None:
                    encoder_output_from_i= encoder_outputs
                    return self.predict(encoder_outputs, attention_bias, training,beam_search_dict)
                else:
                    logits = self.decode(targets, encoder_outputs, attention_bias, training)
                    return logits
    
    #Set up the beam search modes
    def _get_symbols_to_logits_fn(self,max_decode_length, training=False):
        """Returns a decoding function that calculates logits of the next tokens."""
        timing_signal = get_position_encoding(
        max_decode_length + 1, self.params["hidden_size"])
        
        timing_signal = tf.cast(timing_signal, self.params["dtype"])
        decoder_self_attention_bias = get_decoder_self_attention_bias(
            max_decode_length, dtype=self.params["dtype"])
        
        def symbols_to_logits_fn(ids, i, cache):
            decoder_input = ids[:, -1:]
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]
            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoderstack= self.decoder_stack
            #print(model_index, decoderstack,' ',cache.get("encoder_outputs"))
            decoder_outputs = decoderstack(decoder_input,cache.get("encoder_outputs"),
                                                               self_attention_bias,
              cache.get("encoder_decoder_attention_bias"),
              training=training,
              cache=cache,return_attention=True,)
            logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
            logits = tf.squeeze(logits, axis=[1])
            try:
                for name,value in self.attention_weights.items():
                    #print(cache["attention_history"][name].shape)
                    self.attention_weights[name]=cache["attention_history"][name]
            
            #try:
            #    update_decoder_attention_history(cache)
            except:
                print('Error saving the attention weights')
            return logits, cache
        return symbols_to_logits_fn
    
    def predict(self, encoder_outputs, 
                encoder_decoder_attention_bias, 
                training=False,
                beam_search_dict={"beam_size":4,"alpha":0.6}):
        #(encoder_output_from_i, attention_bias, training,model_index)
        #Perform evaluation across the decoder subModel @ model_index
        batch_size = tf.shape(encoder_outputs[0])[0]
        input_length = tf.shape(encoder_outputs[0])[1]
        max_decode_length = input_length + self.params["extra_decode_length"]
        encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                             self.params["dtype"])
        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length, training)
        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)
        # Create cache storing decoder attention values for each layer.
        # pylint: disable=g-complex-comprehension
        att_cache={"attention_history": {}}
        num_layers =self.params["num_hidden_layers"]
        att_batch_size, enc_seq_length = shape_list(encoder_outputs[0])[0:2]
        for layer in range(num_layers):
            att_cache["attention_history"]["layer_%d" % layer] = tf.zeros(
            [att_batch_size,len(encoder_outputs), self.params["num_heads"], 0, enc_seq_length])
            self.attention_weights["layer_%d" % layer] = tf.zeros(
            [att_batch_size,len(encoder_outputs), self.params["num_heads"], 0, enc_seq_length])
        cache = {
        "layer_%d" % layer: {
            "k": tf.zeros([batch_size, 0, self.params["hidden_size"]],
                          dtype=self.params["dtype"]),
            "v": tf.zeros([batch_size, 0, self.params["hidden_size"]],
                          dtype=self.params["dtype"])
        } for layer in range(self.params["num_hidden_layers"])}
        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
        cache["attention_history"]=att_cache["attention_history"]
        try:
            eos_id= self.params["EOS_ID"]
        except:
            eos_id = self.params["vocab_size"]
        # Use beam search to find the top beam_size sequences and scores.
        
        decoded_ids, scores =  sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params["vocab_size"],
            beam_size=beam_search_dict["beam_size"],
            alpha=beam_search_dict["alpha"],
            max_decode_length=max_decode_length,
            eos_id=eos_id ,)
        
        #print(cache.keys())

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}
    
    
def buildBasicJASSModelLayer(params,
                                   checkpoint_path_output,
                             reverse=False,
                               no_checkpoints=False):
    vocab_size = params["vocab_size"]
    label_smoothing = params["label_smoothing"]
    import os
    try:
        os.makedirs(params['tfboard_Logs'])
        os.makedirs(params['tfcheckpoint_dir'])

    except:
        pass
    print('TensorBoard Path ',params['tfboard_Logs'])
    # we set params['use_layer_diversity'] = False to disable the computation of the diversity regularization loss
    params['use_layer_diversity'] = False
    params['use_cL']=False
    #Build the Training Model
    inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
    targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
    
    
    internal_model = JASSTransformer(params=params, 
                                               name="transformer")
    logits = [internal_model([inputs, targets], training=True,reverse=reverse)]
    #Gather Metric evaluation across the decoding sub-models
    louts=[]
    print(len(logits))
    for ii,logit in enumerate(logits):
        logit = MetricLayer(vocab_size,name='M_'+str(ii))([logit, targets]) 
        logit = tf.keras.layers.Lambda(lambda x: x,name='out_conv_'+str(ii), )(logit)
        louts.append(logit)
    #Set up the training Model
    model = tf.keras.Model([inputs,targets], louts)
    for logit in louts:
        loss = transformer_loss(logit, targets, label_smoothing, vocab_size) 
        model.add_loss(loss)
    #Add Optimizer
    optimizer=createOptimizer(params)
    model.compile(optimizer)
    
    #Create the callbacks
    
    callbacks = createCallabacks(params,checkpoint_path_output+'basic_',no_checkpoints=no_checkpoints)
    
    print('checkpoint path ',checkpoint_path_output+'basic_')
    return model,internal_model,callbacks

def buildInferenceModel(internal_model,beam_search_dict={"beam_size":4,"alpha":0.6}):
    inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs_inf")
    ret = internal_model([inputs], training=False,beam_search_dict=beam_search_dict)
    outputs, scores = ret["outputs"], ret["scores"]
    inf_model = tf.keras.Model(inputs, [outputs, scores])
    return inf_model
