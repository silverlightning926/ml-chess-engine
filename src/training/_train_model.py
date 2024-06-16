from src.training._load_dataset import BATCH_SIZE

import tensorflow as tf
from keras.api.models import Model
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime as dt

LOG_DIR = 'logs/' + dt.now().strftime("%Y%m%d-%H%M%S")


def fit_model(model: Model, train_data: tf.data.Dataset, epochs=10, batch_size=BATCH_SIZE, verbose=1):
    early_stopping = EarlyStopping(
        monitor='loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.2, patience=3, min_lr=0.0001)
    tensorboard = TensorBoard(log_dir=LOG_DIR, histogram_freq=1,
                              write_graph=True, write_images=True, update_freq='batch')

    model.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            early_stopping,
            reduce_lr,
            tensorboard
        ]
    )

    return model


def save_model(model: Model, path: str):
    model.save(path)
