#!/usr/bin/bash

#Script to run

#Script for training JASS  model
nb_layers=4
nb_heads=4
task_name=iwslt_en_vi
data_path=datasets/iwslt_en_vi/
vocab_file=vocab.translate_envi_iwslt32k.32768.subwords
test_src=tst2013.en
test_trg=tst2013.vi

#The JASS_MODE tells the MLMHA layers how to compute the source-target attention across the source representations from the $top_n encoding layers
JASS_MODE=2
top_n=4

#Train the model
python3.7 JASSTraining.py -top_n $top_n -jass_mode $JASS_MODE  -batch_size 2048  --model_form MLMHA -tn $task_name --dp $data_path -vocab $vocab_file --num_hidden_layers 4 -train_steps 200000 --max_length 150 -batch_size 2048 -steps_between_evals 4000 -learning_rate_warmup_steps 4000 -filter_size 1024 -d_model 256  -num_heads $nb_heads   
#for epoch in {41..50}
#do
#	python3.7 MLMHA_model_Composer.py -tn $task_name -decode_batch_size 32 -is_EvalMode -top_n $top_n  -jass_mode $JASS_MODE --model_form MLMHA  -beam_size 6 -alpha 1.1 -infID $epoch -filter_size 1024 -d_model 256  -num_heads $nb_heads  --num_hidden_layers $nb_layers --decode_max_length 150 --dp $data_path -vocab $vocab_file --test_src $test_src --test_trg $test_trg	
#done

