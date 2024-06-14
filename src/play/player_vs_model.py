from keras.api.models import load_model, Sequential
import chess
import numpy as np
from tqdm import tqdm

from src.utils import encode_board

model: Sequential = load_model('../../models/model.keras')

# Cache for legal moves
legal_moves_cache = {}


def get_legal_moves(board: chess.Board):
    board_key = board.fen()
    if board_key in legal_moves_cache:
        return legal_moves_cache[board_key]
    else:
        legal_moves = list(board.legal_moves)
        legal_moves_cache[board_key] = legal_moves
        return legal_moves


def evaluate_board(board: chess.Board):
    encoded_board = encode_board(board)
    encoded_board = np.reshape(encoded_board, (1, 8, 8, 12))
    prediction = model.predict(encoded_board, verbose=0, batch_size=1)
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
    else:
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
