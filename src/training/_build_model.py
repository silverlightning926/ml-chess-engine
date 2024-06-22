import os
from keras.api.models import Sequential, load_model
from keras.api.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, Dense, GlobalAveragePooling1D, BatchNormalization
from keras.api.optimizers import Adam
from src.utils.path_utils import find_project_directory

from src.training._load_dataset import BATCH_SIZE


def build_model():
    project_dir = find_project_directory()

    if os.path.exists(os.path.join(project_dir, 'models/model.keras')):
        print('Model already exists. Loading model...')
        model = load_model('models/model.keras')
    else:
        print('Model not found. Building model...')

        model = Sequential([
            Input(shape=(8, 8, 12),
                  batch_size=BATCH_SIZE, name='input'),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),


            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),


            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Flatten(),

            Dense(512, activation='relu'),
            Dropout(0.5),

            Dense(256, activation='relu'),
            Dropout(0.5),

            Dense(128, activation='relu'),
            Dropout(0.5),

            Dense(64, activation='relu'),
            Dropout(0.5),

            Dense(3, activation='linear', name='value'),
        ])

        model.compile(
            optimizer=Adam(),
            loss='mean_squared_error',
            metrics=['accuracy']
        )

    return model
