import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import chess
from utils.encoding_utils import encodeBoard
import numpy as np
import tensorflow as tf
from tqdm import tqdm

DATASET_PATH = 'data/games.csv'
PREPROCESSED_DATA_PATH = 'data/preprocessed_data.npz'


api = KaggleApi()


def _authenticate():
    api.authenticate()


def _download_data(dataset: str, path: str = 'data'):
    api.dataset_download_files(dataset, path=path, unzip=True)


def _fetchData():
    if os.path.exists('data'):
        print('Data already downloaded. Skipping download.')
        return

    _authenticate()
    _download_data('datasnaek/chess')
    print('Data downloaded successfully.')


def _readData():
    print('Reading data...')
    df = pd.read_csv(DATASET_PATH)
    return df


def _generateBoards():
    print('Generating boards...')

    if os.path.exists(PREPROCESSED_DATA_PATH):
        print('Preprocessed data found. Loading...')
        data = np.load(PREPROCESSED_DATA_PATH)
        return data['boards'], data['winners'], data['move_counts']

    df = _readData()

    boards = []
    winners = []
    move_counts = []

    tqdm.pandas(desc='Processing Data')

    def processRow(row):
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
            encodedBoard = encode_board(board)

            boards.append(encodedBoard)
            winners.append(winner)

            move_counts.append(board.fullmove_number)

    df.progress_apply(processRow, axis=1)

    move_counts = np.array(move_counts)
    min_moves = np.min(move_counts)
    max_moves = np.max(move_counts)
    normalized_move_counts = (move_counts - min_moves) / \
        (max_moves - min_moves)

    np.savez_compressed(PREPROCESSED_DATA_PATH, boards=boards,
                        winners=winners, move_counts=move_counts)

    return boards, winners, normalized_move_counts


def _generateDataset(boards, winners, move_counts):
    print('Generating dataset...')

    boards = np.array(boards)
    winners = np.array(winners)

    dataset = tf.data.Dataset.from_tensor_slices(
        (boards, winners, move_counts))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)

    dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

    return dataset


def getData():
    _fetchData()
    boards, winners, move_counts = _generateBoards()
    dataset = _generateDataset(boards, winners, move_counts)
    return dataset
