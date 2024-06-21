import os
from keras.api.models import Sequential, load_model
from keras.api.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, Dense, GlobalAveragePooling1D, BatchNormalization
from src.utils.path_utils import find_project_directory

from src.training._load_dataset import MAX_MOVES, BATCH_SIZE


def build_model():
    project_dir = find_project_directory()

    if os.path.exists(os.path.join(project_dir, 'models/model.keras')):
        print('Model already exists. Loading model...')
        model = load_model('models/model.keras')
    else:
        print('Model not found. Building model...')

        model = Sequential([
            Input(shape=(MAX_MOVES, 8, 8, 12),
                  batch_size=BATCH_SIZE, name='input'),

            TimeDistributed(
                Conv2D(64, (3, 3), activation='relu', padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(MaxPooling2D((2, 2))),

            TimeDistributed(
                Conv2D(128, (3, 3), activation='relu', padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(MaxPooling2D((2, 2))),

            TimeDistributed(
                Conv2D(256, (3, 3), activation='relu', padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(MaxPooling2D((2, 2))),

            TimeDistributed(Flatten()),

            LSTM(256, return_sequences=True),
            Dropout(0.5),

            TimeDistributed(Dense(512, activation='relu')),
            TimeDistributed(Dropout(0.5)),

            TimeDistributed(Dense(256, activation='relu')),
            TimeDistributed(Dropout(0.5)),

            TimeDistributed(Dense(128, activation='relu')),
            TimeDistributed(Dropout(0.5)),

            GlobalAveragePooling1D(),

            Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    return model
