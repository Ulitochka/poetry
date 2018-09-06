import argparse
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from linguistic_style_transfer.linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer.linguistic_style_transfer_model.utils import log_initializer

logger = None


def train_word2vec_model(text_file_path, model_file_path):
    # define training data
    # train model
    logger.info("Loading input file and training mode ...")
    model = Word2Vec(sentences=LineSentence(text_file_path), min_count=1, size=global_config.embedding_size)
    # summarize the loaded model
    logger.info("Model Details: {}".format(model))
    # save model
    model.wv.save_word2vec_format(model_file_path, binary=False)
    logger.info("Model saved")


def main(argv):
    text_file_path = '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/'
    model_file_path = '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/w2v_models/'
    logging_level="DEBUG"

    global logger

    for files in [(text_file_path + 'reviews-val.txt', model_file_path + 'w2v_valid.txt'),
                  (text_file_path + 'reviews-test.txt',  model_file_path + 'w2v_test.txt'),
                  (text_file_path + 'reviews-train.txt', model_file_path + 'w2v_train.txt')]:
      logger = log_initializer.setup_custom_logger(global_config.logger_name, logging_level)
      train_word2vec_model(files[0], files[1])
      logger.info("Training Complete!")

if __name__ == "__main__":
    main(sys.argv[1:])
