from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input


def build_model():
    model = Sequential([
        Input(shape=(8, 8, 12)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((3, 3)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
