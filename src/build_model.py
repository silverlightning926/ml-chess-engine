from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input


def build_model():
    model = Sequential([
        Input(shape=(12, 8, 8)),
        Conv2D(64, (2, 2), activation='relu'),
        Conv2D(64, (2, 2), activation='relu'),
        Conv2D(64, (2, 2), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='tanh')
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error', metrics=['accuracy'])

    return model
