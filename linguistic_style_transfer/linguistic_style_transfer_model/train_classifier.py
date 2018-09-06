import sys
import random
random.seed(1024)

import argparse
import datetime
import json
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import f1_score

from linguistic_style_transfer.linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer.linguistic_style_transfer_model.config.model_config import mconf
from linguistic_style_transfer.linguistic_style_transfer_model.models.text_classifier import TextCNN
from linguistic_style_transfer.linguistic_style_transfer_model.utils import data_processor, log_initializer, tf_session_helper

logger = None
patience = 30


def train_classifier_model(options):
    # Load data
    logger.info("Loading data...")

    [word_index, x, _, _, _] = \
        data_processor.get_text_sequences(
            options['text_file_path'],
            options['vocab_size'],
            global_config.classifier_vocab_save_path)

    x = np.asarray(x)

    [y, _] = data_processor.get_labels(options['label_file_path'], False)

    # print(len(y), len(x))
    # assert 0

    shuffle_indices = np.random.permutation(np.arange(len(x)))
    # shuffle_indices = [i for i in shuffle_indices if i != max(shuffle_indices)]

    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(0.01 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    logger.info("Vocabulary Size: {:d}".format(options['vocab_size']))
    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # Training
    sess = tf_session_helper.get_tensorflow_session()
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=options['vocab_size'],
            embedding_size=128,
            filter_sizes=list(map(int, [3, 4, 5])),
            num_filters=128,
            l2_reg_lambda=0.0)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        out_dir = global_config.classifier_save_directory
        logger.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        # Write vocabulary
        with open(global_config.classifier_vocab_save_path, 'w') as json_file:
            json.dump(word_index, json_file, ensure_ascii=False)
            logger.info("Saved vocabulary")

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            # logger.info("step {}: loss {:g}, acc {:g}".format(step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
            }
            step, summaries, loss, accuracy, pred = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if writer:
                writer.add_summary(summaries, step)
            return loss, pred, step, accuracy

        # Generate batches
        batches = data_processor.batch_iter(
            list(zip(x_train, y_train)), 256, options['training_epochs'])
        # Training loop. For each batch...
        best_f1 = 0
        non_performances = 0
        for i, batch in enumerate(batches):
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if non_performances < patience:
                if current_step % 100 == 0:
                   logger.info("\nEvaluation ...")
                   dev_loss, dev_pred, dev_step_, dev_acc = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                   curr_f1 = f1_score(np.argmax(y_dev, axis=1), dev_pred)
                   if curr_f1 > best_f1:
                       non_performances = 0
                       best_f1 = curr_f1
                       logger.info("step {}, loss {:g}, acc {:g}, f1 {:g}".format(dev_step_, dev_loss, dev_acc, curr_f1))
                       path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                       logger.info("Saved model checkpoint to {}\n".format(path))
                   else:
                       non_performances += 1
            else:
                break


def main():
    logging_level="INFO"

    options = {
        "text_file_path": '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/reviews-train.txt',
        "label_file_path": '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_yelp/sentiment-train.txt',
        "vocab_size": 10000,
        "training_epochs": 100
    }

    global logger

    logger = log_initializer.setup_custom_logger(global_config.logger_name, logging_level)
    os.makedirs(global_config.classifier_save_directory)
    train_classifier_model(options)
    logger.info("Training Complete!")


if __name__ == "__main__":
    main()
