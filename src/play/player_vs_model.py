from keras.api.models import load_model, Model
import chess
import numpy as np
from tqdm import tqdm

from src.utils.encoding_utils import encode_board, encode_castling_rights, encode_to_move, encode_material, encode_move_count

model: Model = load_model('models/model.keras')

# Cache for legal moves
legal_moves_cache = {}


def get_legal_moves(board: chess.Board):
    board_key = board.fen()
    if board_key in legal_moves_cache:
        return legal_moves_cache[board_key]

    legal_moves = list(board.legal_moves)
    legal_moves_cache[board_key] = legal_moves
    return legal_moves


def evaluate_board(board: chess.Board):
    encoded_board = encode_board(board)
    encoded_board = np.reshape(encoded_board, (1, 8, 8, 12))

    encoded_move_count = encode_move_count(board)
    encoded_move_count = np.reshape(encoded_move_count, (1, 1))

    encoded_castling_rights = encode_castling_rights(board)
    encoded_castling_rights = np.reshape(encoded_castling_rights, (1, 4))

    encoded_to_move = encode_to_move(board)
    encoded_to_move = np.reshape(encoded_to_move, (1, 1))

    encoded_material = encode_material(board)
    encoded_material = np.reshape(encoded_material, (1, 10))

    prediction = model.predict(
        [
            encoded_board,
            encoded_move_count,
            encoded_to_move,
            encoded_castling_rights,
            encoded_material
        ], verbose=0, batch_size=1)
    prediction = prediction[0][0]
    prediction = np.clip(prediction, -1, 1)
    return prediction


def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = get_legal_moves(board)

    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            evaluation = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval

    min_eval = float('inf')
    for move in legal_moves:
        board.push(move)
        evaluation = minimax(board, depth - 1, alpha, beta, True)
        board.pop()
        min_eval = min(min_eval, evaluation)
        beta = min(beta, evaluation)
        if beta <= alpha:
            break
    return min_eval


def minimax_root(board: chess.Board, depth: int, maximizing_player: bool):
    best_move = None
    best_eval = float('-inf') if maximizing_player else float('inf')
    legal_moves = get_legal_moves(board)
    for move in tqdm(legal_moves, desc="Finding best move", ascii=True, leave=True):
        board.push(move)
        evaluation = minimax(board, depth - 1, float('-inf'),
                             float('inf'), not maximizing_player)
        board.pop()
        if maximizing_player and evaluation > best_eval:
            best_eval = evaluation
            best_move = move
        elif not maximizing_player and evaluation < best_eval:
            best_eval = evaluation
            best_move = move
    return best_move


def minimax_move(board: chess.Board, depth: int):
    return minimax_root(board, depth, board.turn)


def main():
    board = chess.Board()

    while not board.is_game_over():
        if board.turn:
            move = minimax_move(board, 2)
        else:
            move = input('Enter your move: ')
            while chess.Move.from_uci(move) not in get_legal_moves(board):
                move = input('Enter a legal move: ')
            move = chess.Move.from_uci(move)

        board.push(move)
        print(board)
        print("===")


if __name__ == '__main__':
    main()
