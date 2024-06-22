import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import chess
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sys import getsizeof

from src.utils.encoding_utils import encode_board, encode_winner

DATASET_PATH = 'data/games.csv'
PREPROCESSED_DATA_PATH = 'data/preprocessed_data.npz'

BATCH_SIZE = 16

VALIDATION_SPLIT = 0.2

api = KaggleApi()


def _authenticate():
    api.authenticate()


def _download_data(dataset: str, path: str = 'data'):
    api.dataset_download_files(dataset, path=path, unzip=True)


def _fetch_data():
    if os.path.exists('data'):
        print('Data already downloaded. Skipping download.')
        return

    _authenticate()
    _download_data('datasnaek/chess')
    print('Data downloaded successfully.')


def _read_data():
    print('Reading CSV...')
    df = pd.read_csv(DATASET_PATH)
    return df


def _generate_game_sequences():
    print('Generating boards...')

    if os.path.exists(PREPROCESSED_DATA_PATH):
        print('Preprocessed data found. Loading...')
        data = np.load(PREPROCESSED_DATA_PATH)
        return data['positions'], data['winners']

    print('Creating DataFrame...')
    df = _read_data()
    print(df.head())

    positions = []
    winners = []

    tqdm.pandas(desc='Processing Data')

    def process_row(row):
        board = chess.Board()

        winner = encode_winner(row['winner'])

        for move in row['moves'].split():

            board.push_san(move)
            positions.append(encode_board(board))
            winners.append(winner)

    df.progress_apply(process_row, axis=1)

    print(f'Size of positions: {getsizeof(positions)} bytes')
    print(f'Size of winners: {getsizeof(winners)} bytes')

    print('Saving preprocessed data...')
    np.savez_compressed(PREPROCESSED_DATA_PATH, positions=positions,
                        winners=winners)
    print('Preprocessed data saved.')

    print('Creating numpy arrays...')
    positions = np.array(positions, dtype=np.float32)
    winners = np.array(winners, dtype=np.float32)
    print('Numpy arrays created.')

    return positions, winners


def _generate_dataset(games, winners, ):
    print('Generating dataset...')

    split = int(len(games) * (1 - VALIDATION_SPLIT))

    print(
        f'Splitting data into {split} training samples and {len(games) - split} validation samples.')

    print('Bulding Training Data...')
    train_data = tf.data.Dataset.from_tensor_slices(
        (games[:split], winners[:split])
    ).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    print('Building Validation Data...')
    val_data = tf.data.Dataset.from_tensor_slices(
        (games[split:], winners[split:])
    ).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    return train_data, val_data


def get_data():
    _fetch_data()

    games, winners = _generate_game_sequences()

    return _generate_dataset(games, winners)
