import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import chess
import numpy as np
from tqdm import tqdm

from src.utils.encoding_utils import encode_board, encode_castling_rights, encode_has_castled, encode_to_move, encode_material, encode_winner

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
        return data['boards'], data['winners'], data['move_counts'], data['to_move'], data['castling_rights'], data[
            'has_castled'], data['material']

    df = _read_data()

    boards = []
    winners = []
    move_counts = []
    to_move = []
    castling_rights = []
    has_castled = []
    material = []

    tqdm.pandas(desc='Processing Data')

    def process_row(row):
        winner = encode_winner(row['winner'])

        board = chess.Board()

        for move in row['moves'].split():
            board.push_san(move)
            encoded_board = encode_board(board)

            boards.append(encoded_board)
            winners.append(winner)

            to_move.append(
                encode_to_move(board)
            )

            castling_rights.append(
                encode_castling_rights(board)
            )

            has_castled.append(
                encode_has_castled(board)
            )

            move_counts.append(board.fullmove_number)

            material.append(
                encode_material(board)
            )

    df.progress_apply(process_row, axis=1)

    move_counts = np.array(move_counts)
    min_moves = np.min(move_counts)
    max_moves = np.max(move_counts)
    normalized_move_counts = (move_counts - min_moves) / \
                             (max_moves - min_moves)

    np.savez_compressed(PREPROCESSED_DATA_PATH, boards=boards,
                        winners=winners, move_counts=move_counts, to_move=to_move, castling_rights=castling_rights,
                        has_castled=has_castled, material=material)

    return boards, winners, normalized_move_counts, to_move, castling_rights, has_castled, material


def get_data():
    _fetch_data()

    boards, winners, move_counts, to_move, castling_rights, has_castled, material = _generate_boards()

    boards = np.array(boards)
    winners = np.array(winners)
    move_counts = np.array(move_counts)
    to_move = np.array(to_move)
    castling_rights = np.array(castling_rights)
    has_castled = np.array(has_castled)
    material = np.array(material)

    return boards, move_counts, to_move, castling_rights, has_castled, material, winners
