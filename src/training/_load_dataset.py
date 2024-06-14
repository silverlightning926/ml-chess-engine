import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import chess
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.utils import encode_board

DATASET_PATH = 'data/games.csv'
PREPROCESSED_DATA_PATH = 'data/preprocessed_data.npz'

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
    print('Reading data...')
    df = pd.read_csv(DATASET_PATH)
    return df


def _generate_boards():
    print('Generating boards...')

    if os.path.exists(PREPROCESSED_DATA_PATH):
        print('Preprocessed data found. Loading...')
        data = np.load(PREPROCESSED_DATA_PATH)
        return data['boards'], data['winners'], data['move_counts']

    df = _read_data()

    boards = []
    winners = []
    move_counts = []

    tqdm.pandas(desc='Processing Data')

    def process_row(row):
        winner = row['winner']
        if winner == 'white':
            winner = 1
        elif winner == 'black':
            winner = -1
        else:
            winner = 0

        board = chess.Board()
        for move in row['moves'].split():
            board.push_san(move)
            encoded_board = encode_board(board)

            boards.append(encoded_board)
            winners.append(winner)

            move_counts.append(board.fullmove_number)  # pylint: disable=no-member

    df.progress_apply(process_row, axis=1)

    move_counts = np.array(move_counts)
    min_moves = np.min(move_counts)
    max_moves = np.max(move_counts)
    normalized_move_counts = (move_counts - min_moves) / \
                             (max_moves - min_moves)

    np.savez_compressed(PREPROCESSED_DATA_PATH, boards=boards,
                        winners=winners, move_counts=move_counts)

    return boards, winners, normalized_move_counts


def _generate_dataset(boards, winners, move_counts):
    print('Generating dataset...')

    boards = np.array(boards)
    winners = np.array(winners)

    dataset = tf.data.Dataset.from_tensor_slices(
        (boards, winners, move_counts))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)

    dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

    return dataset


def get_data():
    _fetch_data()
    boards, winners, move_counts = _generate_boards()
    dataset = _generate_dataset(boards, winners, move_counts)
    return dataset
