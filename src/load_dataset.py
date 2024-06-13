import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import chess
import numpy as np
import tensorflow as tf
from tqdm import tqdm

DATASET_PATH = 'data/games.csv'
PIECE_TO_INDEX = {
    'p': 0,
    'r': 1,
    'n': 2,
    'b': 3,
    'q': 4,
    'k': 5,
    'P': 6,
    'R': 7,
    'N': 8,
    'B': 9,
    'Q': 10,
    'K': 11
}

INDEX_TO_PIECE = {v: k for k, v in PIECE_TO_INDEX.items()}

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
    df = _readData()

    games = []
    winners = []

    max_moves = df['moves'].apply(lambda x: len(x.split())).max()

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

        moves = []

        for move in row['moves'].split():
            board.push_san(move)
            encodedBoard = _encodeBoard(board)

            moves.append(encodedBoard)

        while len(moves) < max_moves:
            moves.append(np.zeros(shape=(8, 8, 12), dtype=np.int8))

        moves = np.array(moves, dtype=np.int8)
        games.append(moves)
        winners.append(winner)

    df.progress_apply(processRow, axis=1)

    return games, winners


def _encodeBoard(board: chess.Board):
    encodedBoard = np.zeros(shape=(8, 8, 12), dtype=np.int8)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            encodedBoard[chess.square_rank(square)][chess.square_file(
                square)][PIECE_TO_INDEX[piece.symbol()]] = 1

    return encodedBoard


def _generateDataset(games, winners):
    print('Generating dataset...')

    games = np.array(games, dtype=np.int8)
    winners = np.array(winners, dtype=np.int8)

    dataset = tf.data.Dataset.from_tensor_slices((games, winners))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(32)

    return dataset


def getData():
    _fetchData()
    boards, winners = _generateBoards()
    dataset = _generateDataset(boards, winners)
    return dataset
