from keras.api.models import load_model, Sequential
import chess
from utils.encoding_utils import encodeBoard
import numpy as np
from tqdm import tqdm

model: Sequential = load_model('model.keras')

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


def evaluateBoard(board: chess.Board):
    encodedBoard = encodeBoard(board)
    encodedBoard = np.reshape(encodedBoard, (1, 8, 8, 12))
    prediction = model.predict(encodedBoard, verbose=0, batch_size=1)
    prediction = prediction[0][0]
    prediction = np.clip(prediction, -1, 1)
    return prediction


def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizingPlayer: bool):
    if depth == 0 or board.is_game_over():
        return evaluateBoard(board)

    legal_moves = get_legal_moves(board)

    if maximizingPlayer:
        maxEval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval


def minimaxRoot(board: chess.Board, depth: int, maximizingPlayer: bool):
    bestMove = None
    bestEval = float('-inf') if maximizingPlayer else float('inf')
    legal_moves = get_legal_moves(board)
    for move in tqdm(legal_moves, desc="Finding best move", ascii=True, leave=True):
        board.push(move)
        eval = minimax(board, depth - 1, float('-inf'),
                       float('inf'), not maximizingPlayer)
        board.pop()
        if maximizingPlayer and eval > bestEval:
            bestEval = eval
            bestMove = move
        elif not maximizingPlayer and eval < bestEval:
            bestEval = eval
            bestMove = move
    return bestMove


def minimaxMove(board: chess.Board, depth: int):
    return minimaxRoot(board, depth, board.turn)


def main():
    board = chess.Board()

    while not board.is_game_over():
        if board.turn:
            move = minimaxMove(board, 2)
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
