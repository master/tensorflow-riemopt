#!/usr/bin/env python3
import argparse
import os
import tensorflow as tf
import numpy as np

import model
from shared import utils

DATA_URL = "https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/LieData/G3D_Lie_data.zip"
DATA_FOLDER = "lie20_half_inter1"
G3D_CLASSES = 20
VAL_SPLIT = 0.2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir', type=str, required=True, help='checkpoint dir'
    )
    parser.add_argument('--data-dir', type=str, required=True, help='data dir')
    parser.add_argument(
        '--num-epochs',
        type=float,
        default=100,
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


def prepare_data(args):
    features, labels = utils.load_matlab_data("fea", args.data_dir, DATA_FOLDER)
    features = np.array([np.stack(example) for example in features.squeeze()])
    # reshape to [batch_size, spatial_dim, temp_dim, num_rows, num_cols]
    features = np.transpose(features, axes=[0, 1, 4, 2, 3])
    indices = np.random.permutation(len(features))
    features, labels = features[indices], labels[indices]
    val_len = int(len(features) * VAL_SPLIT)
    X_train, X_val = features[-val_len:, ...], features[:-val_len, ...]
    y_train, y_val = labels[-val_len:, ...], labels[:-val_len, ...]
    return (X_train, y_train), (X_val, y_val)


def train_and_evaluate(args):
    utils.download_data(args.data_dir, DATA_URL, unpack=True)
    train, val = prepare_data(args)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train)
        .repeat(args.num_epochs)
        .shuffle(args.shuffle_buffer)
        .batch(args.batch_size, drop_remainder=True)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(val).batch(
        args.batch_size, drop_remainder=True
    )

    lienet = model.create_model(args.learning_rate, num_classes=G3D_CLASSES)

    os.makedirs(args.job_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.job_dir, "g3d-lienet.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )
    log_dir = os.path.join(args.job_dir, "logs")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    lienet.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=val_dataset,
        callbacks=[cp_callback, tb_callback],
    )
    _, acc = lienet.evaluate(val_dataset, verbose=2)
    print("Final accuracy: {}%".format(acc * 100))


if __name__ == "__main__":
    tf.get_logger().setLevel("INFO")
    train_and_evaluate(get_args())
