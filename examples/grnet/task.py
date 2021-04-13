#!/usr/bin/env python3
import argparse
import os
import tensorflow as tf

import model
from shared import utils

DATA_URL = "https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/GrData/AFEW_Gr_data.zip"
DATA_FOLDER = "grface_400_inter_histeq"
AFEW_CLASSES = 7


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir', type=str, required=True, help='checkpoint dir'
    )
    parser.add_argument('--data-dir', type=str, required=True, help='data dir')
    parser.add_argument(
        '--num-epochs',
        type=float,
        default=50,
        help='number of training epochs (default 50)',
    )
    parser.add_argument(
        '--batch-size',
        default=30,
        type=int,
        help='number of examples per batch (default 30)',
    )
    parser.add_argument(
        '--shuffle-buffer',
        default=100,
        type=int,
        help='shuffle buffer size (default 100)',
    )
    parser.add_argument(
        '--learning-rate',
        default=0.01,
        type=float,
        help='learning rate (default .01)',
    )
    return parser.parse_args()


def train_and_evaluate(args):
    utils.download_data(args.data_dir, DATA_URL, unpack=True)
    train = utils.load_matlab_data("Y1", args.data_dir, DATA_FOLDER, "train")
    val = utils.load_matlab_data("Y1", args.data_dir, DATA_FOLDER, "val")
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train)
        .repeat(args.num_epochs)
        .shuffle(args.shuffle_buffer)
        .batch(args.batch_size, drop_remainder=True)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(val).batch(
        args.batch_size, drop_remainder=True
    )

    grnet = model.create_model(args.learning_rate, num_classes=AFEW_CLASSES)

    os.makedirs(args.job_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.job_dir, "afew-grnet.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )
    log_dir = os.path.join(args.job_dir, "logs")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    grnet.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=val_dataset,
        callbacks=[cp_callback, tb_callback],
    )
    _, acc = grnet.evaluate(val_dataset, verbose=2)
    print("Final accuracy: {}%".format(acc * 100))


if __name__ == "__main__":
    tf.get_logger().setLevel("INFO")
    train_and_evaluate(get_args())
