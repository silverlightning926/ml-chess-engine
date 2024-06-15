import os
from keras.api.models import Sequential, load_model
from keras.api.layers import Dense, Conv2D, Dropout, Input, BatchNormalization, \
    GlobalAveragePooling2D, LeakyReLU, SpatialDropout2D, Concatenate, SeparableConv2D, Add, TimeDistributed, LSTM

from src.utils.path_utils import find_project_directory


def build_model():
    project_dir = find_project_directory()

    if os.path.exists(os.path.join(project_dir, 'models/model.keras')):
        print('Model already exists. Loading model...')
        model = load_model('models/model.keras')

    else:
        print('Model not found. Building model...')
        model = Sequential(layers=[
            Input(shape=(None, 8, 8, 12)),
            TimeDistributed(SeparableConv2D(128, (3, 3), padding='same',
                            activation='relu')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(LeakyReLU()),

            TimeDistributed(SeparableConv2D(128, (3, 3), padding='same',
                            activation='relu')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(LeakyReLU()),

            TimeDistributed(SeparableConv2D(128, (3, 3), padding='same',
                            activation='relu')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(LeakyReLU()),

            TimeDistributed(GlobalAveragePooling2D()),

            LSTM(256, return_sequences=True),
            LSTM(256, return_sequences=True),

            Dense(128, activation='relu'),
            Dropout(0.5),

            Dense(128, activation='relu'),
            Dropout(0.5),

            Dense(1, activation='tanh')
        ])

        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy']
        )

    return model
