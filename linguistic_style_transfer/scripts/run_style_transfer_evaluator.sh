#!/usr/bin/env bash

PROJECT_DIR_PATH=`realpath $(dirname $0)/../`

PYTHONPATH=${PROJECT_DIR_PATH} \
python3  ${PROJECT_DIR_PATH}/linguistic_style_transfer_model/evaluators/style_transfer.py --classifier-saved-model-path /home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/linguistic_style_transfer_model/saved-models-classifier/20180905124158/ --text-file-path /home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/sentiment-test.txt --label-index 0
