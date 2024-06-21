import os
from keras.api.models import Model, load_model
from keras.api.layers import Input, Conv2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Dense, Dropout, TimeDistributed, GRU, Add, LayerNormalization, MultiHeadAttention
from keras.api.optimizers import Adam
from keras.api.regularizers import l2
from keras.api.metrics import Accuracy, BinaryCrossentropy, AUC, Precision, Recall, MeanSquaredError
from src.utils.path_utils import find_project_directory

from src.training._load_dataset import MAX_MOVES, BATCH_SIZE


def time_distributed_residual_block(x, filters, kernel_size):
    shortcut = x
    x = TimeDistributed(Conv2D(filters, kernel_size,
                        padding='same', kernel_regularizer=l2(1e-4)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU())(x)
    x = TimeDistributed(Conv2D(filters, kernel_size,
                        padding='same', kernel_regularizer=l2(1e-4)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Add()([shortcut, x])
    x = TimeDistributed(LeakyReLU())(x)
    return x


def build_model():
    project_dir = find_project_directory()

    if os.path.exists(os.path.join(project_dir, 'models/model.keras')):
        print('Model already exists. Loading model...')
        model = load_model('models/model.keras')
    else:
        print('Model not found. Building model...')

        input_layer = Input(shape=(MAX_MOVES, 8, 8, 12),
                            batch_size=BATCH_SIZE, name='input')

        x = TimeDistributed(Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=l2(1e-4)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(LeakyReLU())(x)

        x = time_distributed_residual_block(x, 64, (3, 3))
        x = time_distributed_residual_block(x, 64, (3, 3))

        x = TimeDistributed(GlobalAveragePooling2D())(x)

        x = GRU(256, return_sequences=True,
                kernel_regularizer=l2(1e-4), stateful=False)(x)
        x = GRU(256, return_sequences=True,
                kernel_regularizer=l2(1e-4), stateful=False)(x)

        attention_output = MultiHeadAttention(num_heads=4, key_dim=256)(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)

        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.5)(x)

        output_layer = Dense(2, activation='softmax',
                             kernel_regularizer=l2(1e-4))(x)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=[Accuracy(), BinaryCrossentropy(), AUC(), Precision(),
                     Recall(), MeanSquaredError()]
        )

    return model
