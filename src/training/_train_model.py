import tensorflow as tf
from keras.api.models import Model
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau


def fit_model(model: Model, train_data: tf.data.Dataset, epochs=10, batch_size=32, verbose=1):
    early_stopping = EarlyStopping(
        monitor='loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.2, patience=3, min_lr=0.0001)

    boards, move_counts, to_move, castling_rights, material, winners = train_data

    model.fit(
        x=[boards, move_counts, to_move, castling_rights, material],
        y=winners,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[early_stopping, reduce_lr],
        validation_split=0.1
    )

    return model


def save_model(model: Model, path: str):
    model.save(path)
