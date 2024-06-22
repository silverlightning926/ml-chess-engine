from keras.api.models import load_model, Model
import chess
import numpy as np
from tqdm import tqdm

from src.utils.encoding_utils import encode_board

model: Model = load_model('models/model.keras')

legal_moves_cache = {}

transposition_table = {}


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

    prediction = model.predict(encoded_board, verbose=0, batch_size=1)
    return prediction[0][0]


def minimax(board: chess.Board, depth: int, alpha: float, beta, maximizing_player: bool):
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
        move = minimax_move(board, 2)
        if move is None:
            break  # No valid moves, must be a game over state
        board.push(move)
        print(board)
        print("===")

    print(board.result())
    print("===")
    print(f"fen: {board.fen()}")
    print("===")
    print(board)
    print("===")
    for move in board.move_stack:
        print(move)


if __name__ == '__main__':
    main()
