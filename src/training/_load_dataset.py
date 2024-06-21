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

MAX_MOVES = 75

BATCH_SIZE = 16

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
        return data['games'], data['winners']

    print('Creating DataFrame...')
    df = _read_data()
    print(df.head())

    if MAX_MOVES is not None:
        print('Removing games that are too long...')
        df = df[df['moves'].apply(lambda x: len(x.split())) <= MAX_MOVES]

    games = []
    winners = []

    tqdm.pandas(desc='Processing Data')

    def process_row(row):
        moves = []

        board = chess.Board()

        for move in row['moves'].split():

            board.push_san(move)
            moves.append(encode_board(board))

        for _ in range(MAX_MOVES - len(row['moves'].split())):
            moves.append(np.zeros((8, 8, 12), dtype=np.float32))

        games.append(np.array(moves, dtype=np.float32))
        winners.append(encode_winner(row['winner']))

    df.progress_apply(process_row, axis=1)

    print(f'Size of games: {getsizeof(games)} bytes')
    print(f'Size of winners: {getsizeof(winners)} bytes')

    print('Saving preprocessed data...')
    np.savez_compressed(PREPROCESSED_DATA_PATH, games=games,
                        winners=winners)
    print('Preprocessed data saved.')

    print('Creating numpy arrays...')
    games = np.array(games, dtype=np.float32)
    winners = np.array(winners, dtype=np.float32)
    print('Numpy arrays created.')

    return games, winners


def _generate_dataset(games, winners, ):
    print('Generating dataset...')
    return tf.data.Dataset.from_tensor_slices((games, winners)).shuffle(
        10000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


def get_data():
    _fetch_data()

    games, winners = _generate_game_sequences()

    return _generate_dataset(games, winners)
