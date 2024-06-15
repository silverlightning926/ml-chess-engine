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
    encoded_board = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank, file = divmod(square, 8)
            encoded_board[rank, file, PIECE_TO_INDEX[piece.symbol()]] = 1

    return encoded_board


def decode_board(encoded_board):
    board = chess.Board()
    for rank in range(8):
        for file in range(8):
            for piece_index in range(12):
                if encoded_board[rank, file, piece_index] == 1:
                    piece = INDEX_TO_PIECE[piece_index]
                    board.set_piece_at(chess.square(
                        file, rank), chess.Piece.from_symbol(piece))
    return board


def encode_winner(winner: str):
    if winner == 'white':
        return 1
    elif winner == 'black':
        return -1
    else:
        return 0


def encode_castling_rights(board: chess.Board):
    return (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    )


def encode_to_move(board: chess.Board):
    return (1 if board.turn else 0, 1 if not board.turn else 0)


def encode_move_count(board: chess.Board):
    return board.fullmove_number


def encode_material(board: chess.Board):
    return (
        len(board.pieces(chess.PAWN, chess.WHITE)),
        len(board.pieces(chess.KNIGHT, chess.WHITE)),
        len(board.pieces(chess.BISHOP, chess.WHITE)),
        len(board.pieces(chess.ROOK, chess.WHITE)),
        len(board.pieces(chess.QUEEN, chess.WHITE)),
        len(board.pieces(chess.PAWN, chess.BLACK)),
        len(board.pieces(chess.KNIGHT, chess.BLACK)),
        len(board.pieces(chess.BISHOP, chess.BLACK)),
        len(board.pieces(chess.ROOK, chess.BLACK)),
        len(board.pieces(chess.QUEEN, chess.BLACK))
    )
