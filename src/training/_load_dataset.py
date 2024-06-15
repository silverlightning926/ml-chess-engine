import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import chess
import numpy as np
from tqdm import tqdm

from src.utils.encoding_utils import encode_board, encode_castling_rights, encode_to_move, encode_material_advantage, encode_winner, encode_move_count, encode_is_checked

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


def _generate_game_sequences():
    print('Generating boards...')

    if os.path.exists(PREPROCESSED_DATA_PATH):
        print('Preprocessed data found. Loading...')
        data = np.load(PREPROCESSED_DATA_PATH)
        return data['boards'], data['winners'], data['move_counts'], data['to_move'], data['castling_rights'], data['material'], data['is_checked']

    df = _read_data()

    boards = []
    winners = []
    move_counts = []
    to_move = []
    castling_rights = []
    material = []
    is_checked = []

    tqdm.pandas(desc='Processing Data')

    def process_row(row):
        board = chess.Board()

        for move in row['moves'].split():
            board.push_san(move)

            encoded_board = encode_board(board)
            encoded_castling = encode_castling_rights(board)
            encoded_to_move = encode_to_move(board)
            encoded_material = encode_material_advantage(board)
            encoded_move_count = encode_move_count(board)
            encoded_is_checked = encode_is_checked(board)

            boards.append(encoded_board)
            winners.append(encode_winner(row['winner']))
            move_counts.append(encoded_move_count)
            to_move.append(encoded_to_move)
            castling_rights.append(encoded_castling)
            material.append(encoded_material)
            is_checked.append(encoded_is_checked)

    df.progress_apply(process_row, axis=1)

    move_counts = np.array(move_counts)
    min_moves = np.min(move_counts)
    max_moves = np.max(move_counts)
    move_counts = (move_counts - min_moves) / \
        (max_moves - min_moves)

    np.savez_compressed(PREPROCESSED_DATA_PATH, boards=boards,
                        winners=winners, move_counts=move_counts, to_move=to_move, castling_rights=castling_rights, material=material, is_checked=is_checked)

    return np.array(boards, dtype=np.float32), np.array(winners, dtype=np.float32), np.array(move_counts, dtype=np.float32), np.array(to_move, dtype=np.float32), np.array(castling_rights, dtype=np.float32), np.array(material, dtype=np.float32), np.array(is_checked, dtype=np.float32)


def get_data():
    _fetch_data()

    return _generate_game_sequences()
