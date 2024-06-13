from keras.api.models import Sequential, load_model
from keras.api.layers import Dense, Flatten, Conv2D, Dropout, Input, BatchNormalization, GlobalAveragePooling2D
import os


def build_model():
    if os.path.exists('model.h5'):
        print('Model already exists. Loading model...')
        model = load_model('model.keras')

    else:
        model = Sequential([
            Input(shape=(8, 8, 12)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
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
