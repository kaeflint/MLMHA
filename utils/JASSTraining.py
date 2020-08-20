from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import time
os.environ['PYTHONHASHSEED'] = '0'
import pickle as pk
import numpy as np
import os
import numpy as np
import random
import tensorflow as tf
np.random.seed(1348)
random.seed(1348)
tf.compat.v1.set_random_seed(89)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import subprocess as sp
import os
import tensorflow_datasets as tfds
import tensorflow as tf
from keras import backend as K
import tensorflow_datasets as tfds
import tensorflow as tf
from utilities import *
import collections
import math

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from utils.Checkpointmanager import *
from utils.dataset_loader import *

from utils.BleuMetric import *
from utils.newtranslate import *
import argparse


parser = argparse.ArgumentParser(description='JASS Model Parameters')

# Set up the dataset information
parser.add_argument('-tn', '--task_name', type=str, required=True)
parser.add_argument('-is_baseline', '--is_baseline', action='store_true',help='Flag to switch to baseline model i.e without auxiliary decoders')
parser.add_argument('-rev','--rev', action='store_true',help='Flag to swap the source and target languages')
parser.add_argument('-mf','--model_form',type=str,default='basic')
parser.add_argument('-data_path', '--dp', type=str,required=True,
                    help='Folder containing the training  and validation TDF files, the test source and target files, as well as the vocabulary')
parser.add_argument('-vocab',
                    '--vocab_file', 
                    type=str, required=True, 
                    help='vocabulary file')
parser.add_argument('-test_src', '--test_src', default='', type=str, help='test source file')
parser.add_argument('-test_trg', '--test_trg', default='', type=str, help='test target file')
parser.add_argument('-max_length', "--max_length", default=200, type=int,
                    help="Maximum length for an input sequence")

# Set up the model parameters

#with_layer_position_embedding
#parser.add_argument('-agg_fn', '--agg_fn', default='lc', type=str, help='Function to compute the global sentence representation')
#parser.add_argument('-layerweight', '--layerweight', action='store_true',help='Flag to employ the attention scores to choose the best contexts ')
#parser.add_argument('-is_hybrid', '--is_hybrid', action='store_true',help='Flag to combine context extracted via the joint attention and that from layer-wise attention')
parser.add_argument('-num_hidden_layers', "--num_hidden_layers", default=6, type=int,
                    help="Number of layers")
parser.add_argument('-top_n', "--top_n", default=6, type=int,
                    help="Number of encoder layers to consider")
parser.add_argument('-share_query_weights','--share_query_weights', action='store_true',help='Flag to share the weights for the query multi-head attention ')
parser.add_argument('-share_weights','--share_weights', action='store_true',help='Flag to share the weights for the query-key-value multi-head attention ')
parser.add_argument('-jass_mode', "--jass_mode", default=1, type=int,
                    help='''
                    JASS computation mode: 1=> Joint_attention + Sum, 2=> joint_attention + Concat, 3=> indv_attention + Sum and 4=> indv_attention + Concat
                    mode: 5=> Naive-sum Mode: 6=> Naive Weighted-Concat
                    '''#parser.add_argument('-share_weights', '--share_weights', action='store_true',help='Flag to indicate attention weight sharing ')
                   )
parser.add_argument('-num_heads', "--num_heads", default=8, type=int,
                    help="Number of attention heads")
parser.add_argument('-d_model', "--d_model", default=512, type=int, help="Output dimension of each layer")
parser.add_argument('-filter_size', "--filter_size", default=2048, type=int,
                    help="Dimension of the Filter for the FFN sub-layer")

# Set up the dropout rates
parser.add_argument('-gate_dropout', "--gate_dropout", default=0.1,
                    type=float,
                    help="dropout rate for the Gated Information Fusion")
parser.add_argument('-attention_dropout', "--attention_dropout", default=0.1,
                    type=float,
                    help="dropout rate for the Attention weights")
parser.add_argument('-relu_dropout', "--relu_dropout", default=0.1,
                    type=float,
                    help="dropout rate for the RELU of FFN")
parser.add_argument('-layer_postprocess_dropout', "--layer_postprocess_dropout", default=0.1,
                    type=float,
                    help="dropout rate for the Residual and Layer Normalization Unit")
# Set up model training parameters
parser.add_argument('-batch_size', "--batch_size", 
                    #default=4096, 
                    default=4960,
                    type=int,
                    help="Maximum tokens in Each batch")
parser.add_argument('-train_steps', "--train_steps", default=160000, type=int,
                    help="Number of Training steps")

parser.add_argument('-steps_between_evals', "--steps_between_evals", default=10000, type=int,
                    help="Number of Training steps before each evaluation step")

parser.add_argument('-label_smoothing', "--label_smoothing", default=0.1,
                    type=float,
                    help="Label Smoothing weight")

# Set up the learning rate and optimizer parameters
parser.add_argument('-optimizer_adam_beta1', "--optimizer_adam_beta1", default=0.9,
                    type=float,
                    help="optimizer_adam_beta1")
parser.add_argument('-optimizer_adam_beta2', "--optimizer_adam_beta2", default=0.98,
                    type=float,
                    help="optimizer_adam_beta2")
parser.add_argument('-optimizer_adam_epsilon', "--optimizer_adam_epsilon", default=1e-9,
                    type=float,
                    help="optimizer_adam_epsilon")
parser.add_argument('-learning_rate', "--learning_rate", default=2.0,
                    type=float,
                    help="Learning rate")
parser.add_argument('-learning_rate_warmup_steps', "--learning_rate_warmup_steps", default=16000,
                    type=float,
                    help="Learning rate warmup steps")
parser.add_argument('-max_lr', "--max_lr", default=1e-3,
                    type=float,
                    help="Maximum learning rate")
parser.add_argument('-min_lr', "--min_lr", default=1e-6,
                    type=float,
                    help="Minimum learning rate")
parser.add_argument('-lr_schedule', '--lr_schedule', default='cos',
                    type=str, help='Learning rate Scheduling algorithm')
parser.add_argument('-nb_cycles', "--nb_cycles", default=1, type=int,
                    help="Number of Cosine Cycles")

# Set up the parameters for inference
parser.add_argument('-beam_size', "--beam_size", default=4, type=int,
                    help="Beam Size")
parser.add_argument('-alpha', "--alpha", default=0.6,
                    type=float,
                    help="Beam Search Alpha")
parser.add_argument('-extra_decode_length', "--extra_decode_length", default=20, type=int,
                    help="extra decode sequence length")
parser.add_argument('-decode_batch_size', "--decode_batch_size", default=20, type=int,
                    help="Beam search batch size")
parser.add_argument('-decode_max_length', "--decode_max_length", default=250, type=int,
                    help="decode max length")


args = parser.parse_args()
if args.rev:
   print('Reverse Translation')
auxLayers = []
# Build the Dictionary
params_dict = vars(parser.parse_args())
#del params_dict['auxLayers']

_TARGET_VOCAB_SIZE =32000
t2t_path = args.dp
eval_test_path =[t2t_path+args.test_src,
                 t2t_path+args.test_trg]
print('learning rates: ', str(args.min_lr),' ',str(args.max_lr),' Warmup: ',str(args.learning_rate_warmup_steps))
print(eval_test_path)

vocab_file = os.path.join(t2t_path, args.vocab_file)
print(vocab_file)

# Load the Vocabulary
subwordTokenizer = LoadorCreateVocabulary(vocab_file,[],_TARGET_VOCAB_SIZE)
vectorizer = getVectorizer(subwordTokenizer)
sentFunction = sent_funct = getSentGenerator(subwordTokenizer)
target_vocab_size = subwordTokenizer.vocab_size
end_token = text_encoder.EOS_ID

_PREFIX = "en_de_exp_"
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"
_TRAIN_SHARDS = 100

task_name=args.task_name
model_form=args.model_form
path_str=model_form+'-Model-Mode'+str(args.jass_mode)+'_top_n'+str(args.top_n) 
print(path_str)
# Some extra prefixed keys-values
constant_params = {
    'num_parallel_calls': True,
    'static_batch': False,
    "enable_metrics_in_training": False,
    'num_gpus':1,
    "use_synthetic_data":False,
    "use_tpu":False,
    'profile_steps':'2,2',
    'enable_metrics_in_training':False,
    'log_steps':100,
    'use_cL':False,
    'hidden_size':params_dict['d_model'],
    "vocab_size":target_vocab_size,
    "EOS_ID":end_token,
    "num_parallel_calls":tf.data.experimental.AUTOTUNE,
    'dtype':tf.float32,
   'tfboard_Logs':task_name+'_tensorboard/tfLogs-'+path_str+'/scalars/nxx',
   'data_dir':t2t_path,
   "repeat_dataset":None,
'aux_layers_@': [],
'share_nodes':args.share_weights
}
#params_dict['aux_layers_@'] = auxLayers

params_dict.update(constant_params)
params=params_dict

try:
  #Load the training set
  train_ds = train_input_fn(params)
  train_ds = train_ds.map(map_data_for_transformer_fn, num_parallel_calls=params["num_parallel_calls"])
  #Load The test Dataset
  test_ds = eval_input_fn(params)
  test_ds = test_ds.map(map_data_for_transformer_fn, num_parallel_calls=params["num_parallel_calls"])
  print('Data Loaded')
except:
  print('Error Loading dataset')

#Build Model
checkpoint_eval= 'ExperimentationOutput/Trans_'+task_name+'/'+path_str+'/'
checkpoint_path_output = checkpoint_eval+'chck_outputs/'
try:
    os.makedirs(checkpoint_path_output )
except:
    pass
geneOPath=checkpoint_path_output+'/basic/outpreds/'
try:
    os.makedirs(geneOPath)
except:
    print(geneOPath+'  already exists')
    pass
model_params= copy.deepcopy(params)
pk.dump(model_params,open(checkpoint_path_output+'/params.config','wb'))
check_point_path = check_point_path = checkpoint_path_output + '/basic_/'
print('Building JASS model')
from JASSModel import *
model,internal_model,callbacks = buildBasicJASSModelLayer(model_params,
                                                                    checkpoint_path_output=checkpoint_path_output,
                                                                    no_checkpoints=True)
# save the Model weights
modelchkp = Checkpoint(model, model.optimizer, model_dir=check_point_path, keep_checkpoint_max=10)
tmp_model = modelchkp.model
inference_models = buildInferenceModel(internal_model=internal_model,
                                     beam_search_dict={"beam_size": args.beam_size, "alpha": args.alpha})
inference_models.summary()

# Final Clean up before Training
eparams = copy.deepcopy(params)
iterations = params['train_steps'] // params['steps_between_evals']
print('Training Epochs: ',iterations)

#Time to train the Model
losses=[]
bleus=[]
chck_paths=[]

for i in range(1, iterations + 1):
    history = model.fit(
        train_ds,
        initial_epoch=i - 1,
        epochs=i,
        steps_per_epoch=params['steps_between_evals'],
        callbacks=callbacks, verbose=2)
    try:
        losses.append(history.history)
    except:
        try:
            losses.append(history.history())
        except:
            pass
        pass
    try:
        llp = modelchkp.save(i)
        chck_paths.append(llp)
    except:
        pass
try:
    pk.dump(losses, open(geneOPath + '/trainStats.dat', 'wb'))
except:
    pass
print('Completed')