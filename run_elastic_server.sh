#!/bin/bash

export KUNGFU_CONFIG_LOG_LEVEL=INFO # DEBUG | INFO | WARN | ERROR

KUNGFU_RUN=$HOME/KungFu/bin/kungfu-run
BERT_BASE_DIR=$HOME/bert/uncased_L-12_H-768_A-12
SQUAD_DIR=$HOME/dataset/squad2
OUTPUT_DIR=tmp

$HOME/KungFu/bin/kungfu-config-server-example \
    -init workers.json &

$KUNGFU_RUN \
    -w \
    -H 127.0.0.1:4 \
    -config-server http://127.0.0.1:9100/get \
    -np 4 \
    -logfile kungfu-run.log \
    -logdir $OUTPUT_DIR \
    python3 run_squad_elastic_server.py \
		--vocab_file=$BERT_BASE_DIR/vocab.txt \
		--bert_config_file=$BERT_BASE_DIR/bert_config.json \
		--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
		--do_train=True \
		--train_file=$SQUAD_DIR/train-v2.0.json \
		--do_predict=False \
		--predict_file=$SQUAD_DIR/dev-v2.0.json \
		--train_batch_size=8 \
		--learning_rate=3e-5 \
		--max_seq_length=384 \
		--doc_stride=128 \
		--output_dir=$OUTPUT_DIR \
		--version_2_with_negative=True \
                --num_train_epochs=0.5

pkill -P $$

