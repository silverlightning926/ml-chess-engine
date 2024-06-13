import tensorflow as tf
from keras.api.models import Sequential
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau


def fit_model(model, train_data: tf.data.Dataset, epochs=10, batch_size=32, verbose=1):
    early_stopping = EarlyStopping(
        monitor='loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.2, patience=3, min_lr=0.0001)

    model.fit(train_data, epochs=epochs,
              batch_size=batch_size, verbose=verbose,
              callbacks=[early_stopping, reduce_lr])

    return model


def save_model(model: Sequential, path: str):
    model.save(path)
