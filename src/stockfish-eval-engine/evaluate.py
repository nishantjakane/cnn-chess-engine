import torch
import chess
import chess.pgn
from encoder import board_to_tensor
from model import ChessCNN

device = "cuda"

model = ChessCNN().to(device)

model_data = torch.load(r"D:\Coding\CNN Chess Engine\data\chess_models\best_model.pth")

model.load_state_dict(model_data["model_state_dict"])

model.eval()

def evaluate_position(board):
    tensor = board_to_tensor(board)
    tensor = tensor.unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        score = model(tensor)

    return score.item()

def find_best_move(board,depth):

    if board.turn == chess.WHITE:
        best_score = float("-inf")

    else:
        best_score = float("inf")


    best_move = None

    for move in board.legal_moves:
        board.push(move)

        score = minimax(board,depth-1)

        board.pop()

        if board.turn == chess.WHITE:
            if score > best_score :
                best_score = score
                best_move = move
        else:
            if score < best_score :
                best_score = score
                best_move = move

    return best_move


def minimax(board,depth):
    if depth == 0:
        return evaluate_position(board)

    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -10000
        else:
            return 10000

    if board.is_stalemate():
        return 0

    if board.turn == chess.WHITE:

        best_score = float("-inf")

        for move in board.legal_moves:
            board.push(move)

            score = minimax(board,depth-1)

            board.pop()

            best_score = max(score,best_score)

        return best_score

    else:
        best_score = float("inf")

        for move in board.legal_moves:
            board.push(move)

            score = minimax(board,depth-1)

            board.pop()

            best_score = min(score,best_score)

        return best_score



board = chess.Board()

# for move in board.legal_moves:

#     board.push(move)

#     score = minimax(board, 2)

#     board.pop()

#     print(move, score)

game = chess.pgn.Game()

node = game

while not board.is_game_over():

    move = find_best_move(board, depth=2)

    board.push(move)

    node = node.add_variation(move)

print(game)