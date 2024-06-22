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

PIECE_TO_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

MAX_PIECE_COUNTS = {
    chess.PAWN: 8,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 2,
    chess.QUEEN: 1,
}


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
    return np.array([
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK))
    ], dtype=np.float32)


def encode_to_move(board: chess.Board):
    return int(board.turn)


def encode_move_count(board: chess.Board):
    return board.fullmove_number


def encode_material_advantage(board: chess.Board):

    material_count = np.zeros(10, dtype=np.float32)

    material_count[0] = len(board.pieces(
        chess.PAWN, chess.WHITE)) * PIECE_TO_VALUE[chess.PAWN] / MAX_PIECE_COUNTS[chess.PAWN]
    material_count[1] = len(board.pieces(
        chess.KNIGHT, chess.WHITE)) * PIECE_TO_VALUE[chess.KNIGHT] / MAX_PIECE_COUNTS[chess.KNIGHT]
    material_count[2] = len(board.pieces(
        chess.BISHOP, chess.WHITE)) * PIECE_TO_VALUE[chess.BISHOP] / MAX_PIECE_COUNTS[chess.BISHOP]
    material_count[3] = len(board.pieces(
        chess.ROOK, chess.WHITE)) * PIECE_TO_VALUE[chess.ROOK] / MAX_PIECE_COUNTS[chess.ROOK]
    material_count[4] = len(board.pieces(
        chess.QUEEN, chess.WHITE)) * PIECE_TO_VALUE[chess.QUEEN] / MAX_PIECE_COUNTS[chess.QUEEN]

    material_count[5] = len(board.pieces(
        chess.PAWN, chess.BLACK)) * PIECE_TO_VALUE[chess.PAWN] / MAX_PIECE_COUNTS[chess.PAWN]
    material_count[6] = len(board.pieces(
        chess.KNIGHT, chess.BLACK)) * PIECE_TO_VALUE[chess.KNIGHT] / MAX_PIECE_COUNTS[chess.KNIGHT]
    material_count[7] = len(board.pieces(
        chess.BISHOP, chess.BLACK)) * PIECE_TO_VALUE[chess.BISHOP] / MAX_PIECE_COUNTS[chess.BISHOP]
    material_count[8] = len(board.pieces(
        chess.ROOK, chess.BLACK)) * PIECE_TO_VALUE[chess.ROOK] / MAX_PIECE_COUNTS[chess.ROOK]
    material_count[9] = len(board.pieces(
        chess.QUEEN, chess.BLACK)) * PIECE_TO_VALUE[chess.QUEEN] / MAX_PIECE_COUNTS[chess.QUEEN]

    return material_count


def encode_is_checked(board: chess.Board):
    return np.array([board.is_check(), board.turn == chess.WHITE and board.is_check(), board.turn == chess.BLACK and board.is_check()], dtype=np.float32)
