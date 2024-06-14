import os

from src.utils import find_project_directory
from keras.api.models import Model, load_model
from keras.api.layers import Dense, Conv2D, Dropout, Input, BatchNormalization, \
    GlobalAveragePooling2D, LeakyReLU, SpatialDropout2D, Concatenate, SeparableConv2D, Add


def build_model():
    project_dir = find_project_directory()

    if os.path.exists(os.path.join(project_dir, 'models/model.keras')):
        print('Model already exists. Loading model...')
        model = load_model('models/model.keras')

    else:
        print('Model not found. Building model...')
        input_board = Input(shape=(8, 8, 12), name='board')
        input_move_count = Input(shape=(1,), name='move_count')
        input_to_move = Input(shape=(2,), name='to_move')
        input_castling_rights = Input(shape=(4,), name='castling_rights')
        input_has_castled = Input(shape=(4,), name='has_castled')
        input_material = Input(shape=(10,), name='material')
        input_features = [input_board, input_move_count, input_to_move, input_castling_rights, input_has_castled,
                          input_material]

        conv1 = Conv2D(128, (3, 3), padding='same')(input_board)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU()(conv1)

        conv2 = SeparableConv2D(128, (3, 3), padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU()(conv2)

        conv3 = SeparableConv2D(128, (3, 3), padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU()(conv3)

        conv4 = SeparableConv2D(128, (3, 3), padding='same')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU()(conv4)

        residual = Add()([conv1, conv4])

        spatial_dropout = SpatialDropout2D(0.3)(residual)
        global_pool = GlobalAveragePooling2D()(spatial_dropout)

        combined_features = Concatenate()([global_pool, input_move_count, input_to_move, input_castling_rights,
                                           input_has_castled, input_material])

        dense1 = Dense(256)(combined_features)
        dense1 = BatchNormalization()(dense1)
        dense1 = LeakyReLU()(dense1)
        dense1 = Dropout(0.5)(dense1)

        dense2 = Dense(256)(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = LeakyReLU()(dense2)
        dense2 = Dropout(0.5)(dense2)

        dense3 = Dense(256)(dense2)
        dense3 = BatchNormalization()(dense3)
        dense3 = LeakyReLU()(dense3)

        output = Dense(1, activation='tanh')(dense3)

        model = Model(inputs=input_features, outputs=output)

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    return model
