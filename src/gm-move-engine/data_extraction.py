import os
from torch.utils.data import DataLoader 
from chess import pgn
import chess

import chess
import torch

WHITE_PAWN = 0
WHITE_KNIGHT = 1
WHITE_BISHOP = 2
WHITE_ROOK = 3
WHITE_QUEEN = 4
WHITE_KING = 5

BLACK_PAWN = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_ROOK = 9
BLACK_QUEEN = 10
BLACK_KING = 11

WHITE_KINGSIDE = 12
WHITE_QUEENSIDE = 13
BLACK_KINGSIDE = 14
BLACK_QUEENSIDE = 15

SIDE_TO_MOVE = 16
EN_PASSANT = 17

NUM_PLANES = 18

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): WHITE_PAWN,
    (chess.KNIGHT, chess.WHITE): WHITE_KNIGHT,
    (chess.BISHOP, chess.WHITE): WHITE_BISHOP,
    (chess.ROOK, chess.WHITE): WHITE_ROOK,
    (chess.QUEEN, chess.WHITE): WHITE_QUEEN,
    (chess.KING, chess.WHITE): WHITE_KING,

    (chess.PAWN, chess.BLACK): BLACK_PAWN,
    (chess.KNIGHT, chess.BLACK): BLACK_KNIGHT,
    (chess.BISHOP, chess.BLACK): BLACK_BISHOP,
    (chess.ROOK, chess.BLACK): BLACK_ROOK,
    (chess.QUEEN, chess.BLACK): BLACK_QUEEN,
    (chess.KING, chess.BLACK): BLACK_KING,
}

def board_to_tensor(board):
    tensor = torch.zeros(
        (NUM_PLANES,8,8),
        dtype=torch.float32
    )

    for square in chess.SQUARES:
        piece=board.piece_at(square) # piece_at gives the type of piece which is on that square
        
        if piece is None:
            continue
        
        plane = PIECE_TO_PLANE[(piece.piece_type,piece.color)]


        file = chess.square_file(square)
        rank = chess.square_rank(square)

        row = rank
        col = file
        
        tensor[plane][row][col] = 1.0

    # Castle Planes

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[WHITE_KINGSIDE] =1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[WHITE_QUEENSIDE] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[BLACK_KINGSIDE] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[BLACK_QUEENSIDE] = 1.0

    # Turn Planes

    if board.turn == chess.WHITE:
        tensor[SIDE_TO_MOVE] =1.0
    
    # En Passant Plane

    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        ep_rank = chess.square_rank(board.ep_square)
        tensor[EN_PASSANT][ep_rank][ep_file] = 1.0

    return tensor

