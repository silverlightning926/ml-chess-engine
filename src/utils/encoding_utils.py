import numpy as np
import chess

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


def encode_board(board: chess.Board):
    encodedBoard = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank, file = divmod(square, 8)
            encodedBoard[rank, file, PIECE_TO_INDEX[piece.symbol()]] = 1

    return encodedBoard


def decode_board(encodedBoard):
    board = chess.Board()
    for rank in range(8):
        for file in range(8):
            for piece_index in range(12):
                if encodedBoard[rank, file, piece_index] == 1:
                    piece = INDEX_TO_PIECE[piece_index]
                    board.set_piece_at(chess.square(
                        file, rank), chess.Piece.from_symbol(piece))
    return board
