import sys

import argparse
import json
import numpy as np
import os
import pickle
import tensorflow as tf

from config import global_config
from config.model_config import mconf
from config.options import Options
from models import adversarial_autoencoder
from utils import bleu_scorer, data_processor, log_initializer, word_embedder, tf_session_helper


def get_word_embeddings(embedding_model_path, word_index):
    encoder_embedding_matrix = np.random.uniform(
        size=(global_config.vocab_size, global_config.embedding_size),
        low=-0.05, high=0.05).astype(dtype=np.float32)
    logger.debug("encoder_embedding_matrix: {}".format(encoder_embedding_matrix.shape))

    decoder_embedding_matrix = np.random.uniform(
        size=(global_config.vocab_size, global_config.embedding_size),
        low=-0.05, high=0.05).astype(dtype=np.float32)
    logger.debug("decoder_embedding_matrix: {}".format(decoder_embedding_matrix.shape))

    if embedding_model_path:
        logger.info("Loading pretrained embeddings")
        encoder_embedding_matrix, decoder_embedding_matrix = \
            word_embedder.add_word_vectors_to_embeddings(
                word_index, encoder_embedding_matrix, decoder_embedding_matrix,
                embedding_model_path)

    return encoder_embedding_matrix, decoder_embedding_matrix


def execute_post_inference_operations(
        actual_word_lists, generated_sequences, final_sequence_lengths, inverse_word_index,
        timestamped_file_suffix, label):
    logger.debug("Minimum generated sentence length: {}".format(min(final_sequence_lengths)))

    # first trims the generates sentences down to the length the decoder returns
    # then trim any <eos> token
    trimmed_generated_sequences = \
        [[index for index in sequence
          if index != global_config.predefined_word_index[global_config.eos_token]]
         for sequence in [x[:(y - 1)] for (x, y) in zip(generated_sequences, final_sequence_lengths)]]

    generated_word_lists = \
        [data_processor.generate_words_from_indices(x, inverse_word_index)
         for x in trimmed_generated_sequences]

    # Evaluate model scores
    bleu_scores = bleu_scorer.get_corpus_bleu_scores(
        [[x] for x in actual_word_lists], generated_word_lists)
    logger.info("bleu_scores: {}".format(bleu_scores))

    generated_sentences = [" ".join(x) for x in generated_word_lists]
    output_file_path = "output/{}-inference/generated_sentences_{}.txt".format(
        timestamped_file_suffix, label)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as output_file:
        for sentence in generated_sentences:
            output_file.write(sentence + "\n")

    actual_sentences = [" ".join(x) for x in actual_word_lists]
    output_file_path = "output/{}-inference/actual_sentences_{}.txt".format(
        timestamped_file_suffix, label)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as output_file:
        for sentence in actual_sentences:
            output_file.write(sentence + "\n")




def main(argv):
    options = Options()

    parser = argparse.ArgumentParser()
    parser.add_argument("--logging-level", type=str, default="INFO")
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument("--train-model", action="store_true", default=False)
    run_mode.add_argument("--generate-novel-text", action="store_true", default=False)

    parser.parse_known_args(args=argv, namespace=options)
    parser.add_argument("--saved-model-path", type=str, required=True)
    parser.add_argument("--evaluation-text-file-path", type=str, required=True)
    parser.parse_known_args(args=argv, namespace=options)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, options.logging_level)

    if not (options.train_model or options.generate_novel_text):
        logger.info("Nothing to do. Exiting ...")
        sys.exit(0)

    global_config.training_epochs = options.training_epochs
    logger.info("experiment_timestamp: {}".format(global_config.experiment_timestamp))

    logger.info("Generating novel text ...")

    with open(os.path.join(options.saved_model_path, global_config.model_config_file), 'r') as json_file:
        model_config_dict = json.load(json_file)
        mconf.init_from_dict(model_config_dict)
        logger.info("Restored model config from saved JSON")

    with open(os.path.join(options.saved_model_path, global_config.vocab_save_file), 'r') as json_file:
        word_index = json.load(json_file)
    with open(os.path.join(options.saved_model_path, global_config.index_to_label_dict_file), 'r') as json_file:
        index_to_label_map = json.load(json_file)
    with open(os.path.join(options.saved_model_path, global_config.average_label_embeddings_file), 'rb') as pickle_file:
        average_label_embeddings = pickle.load(pickle_file)

    global_config.vocab_size = len(word_index)

    num_labels = len(index_to_label_map)
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=global_config.vocab_size,
        filters=global_config.tokenizer_filters)
    text_tokenizer.word_index = word_index

    inverse_word_index = {v: k for k, v in word_index.items()}
    [actual_sequences, _, padded_sequences, text_sequence_lengths] = \
        data_processor.get_test_sequences(options.evaluation_text_file_path,
                                          text_tokenizer,
                                          word_index,
                                          inverse_word_index)

    # TODO CREATE FAKE LABELS, options = that style you use: {0 - general, 1 - new_style}
    [label_sequences, _] = data_processor.get_test_labels(options.evaluation_label_file_path,
                                                          options.saved_model_path)

    logger.info("Building model architecture ...")
    network = adversarial_autoencoder.AdversarialAutoencoder()
    encoder_embedding_matrix, decoder_embedding_matrix = get_word_embeddings(None, word_index)
    network.build_model(word_index, encoder_embedding_matrix, decoder_embedding_matrix, num_labels)
    sess = tf_session_helper.get_tensorflow_session()

    total_nll = 0
    for i in range(num_labels):
        logger.info("Style chosen: {}".format(i))

        filtered_actual_sequences = list()
        filtered_padded_sequences = list()
        filtered_text_sequence_lengths = list()
        for k in range(len(actual_sequences)):
            if label_sequences[k] != i:
                filtered_actual_sequences.append(actual_sequences[k])
                filtered_padded_sequences.append(padded_sequences[k])
                filtered_text_sequence_lengths.append(text_sequence_lengths[k])

        style_embedding = np.asarray(average_label_embeddings[i])
        [generated_sequences, final_sequence_lengths, _, _, _, cross_entropy_scores] = \
            network.generate_novel_sentences(
                sess, filtered_padded_sequences, filtered_text_sequence_lengths, style_embedding,
                num_labels, os.path.join(options.saved_model_path, global_config.model_save_file))
        nll = -np.mean(a=cross_entropy_scores, axis=0)
        total_nll += nll
        logger.info("NLL: {}".format(nll))

        actual_word_lists = [
            data_processor.generate_words_from_indices(x, inverse_word_index) for x in filtered_actual_sequences]

        execute_post_inference_operations(
            actual_word_lists,
            generated_sequences,
            final_sequence_lengths,
            inverse_word_index,
            global_config.experiment_timestamp,
            i)

        logger.info("Generation complete for label {}".format(i))

    logger.info("Mean NLL: {}".format(total_nll / num_labels))



