#!/usr/bin/env bash

PROJECT_DIR_PATH=`realpath $(dirname $0)/../`

python3 ${PROJECT_DIR_PATH}/linguistic_style_transfer_model/main.py \
--vocab-size 10000 \
--training-epochs 50 \
--train-model \
--text-file-path '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/reviews-train.txt' \
--label-file-path '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/sentiment-train.txt' \
--validation-text-file-path '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/reviews-val.txt' \
--validation-label-file-path '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/sentiment-val.txt' \
--training-embeddings-file-path '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/w2v_models/w2v_train.txt' \
--validation-embeddings-file-path "/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/w2v_models/w2v_valid.txt" \
--dump-embeddings False --logging-level DEBUG \
--classifier-saved-model-path '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/linguistic_style_transfer_model/saved-models-classifier/20180905124158/'

#python3 /home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/linguistic_style_transfer_model/main.py \
#--generate-novel-text \
#--evaluation-text-file-path '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/reviews-test.txt' \
#--saved-model-path '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/scripts/saved-models/20180905142637' \
#--evaluation-label-file-path '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/sentiment-test.txt' \
#--logging-level="DEBUG"