import torch
import chess
import chess.pgn

from model import ChessCNN
from data_extraction import board_to_tensor

# ==========================
# LOAD MODEL
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"

move_to_idx = torch.load(
    r"D:\Coding\CNN Chess Engine\data\gmdata\move_to_idx.pt"
)

idx_to_move = torch.load(
    r"D:\Coding\CNN Chess Engine\data\gmdata\idx_to_move.pt"
)

num_moves = len(move_to_idx)

model = ChessCNN(num_moves).to(device)

checkpoint = torch.load(
    r"D:\Coding\CNN Chess Engine\data\gmdata\model_data\best_model.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# ==========================
# EVALUATION
# ==========================

def evaluate_position(board):

    tensor = board_to_tensor(board)
    tensor = tensor.unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor)[0]

    legal_scores = []

    for move in board.legal_moves:

        uci = move.uci()

        if uci in move_to_idx:
            idx = move_to_idx[uci]
            legal_scores.append(logits[idx].item())

    if len(legal_scores) == 0:
        return 0

    return max(legal_scores)


# ==========================
# MINIMAX
# ==========================

def minimax(board, depth):

    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -100000
        else:
            return 100000

    if board.is_stalemate():
        return 0

    if depth == 0:
        return evaluate_position(board)

    if board.turn == chess.WHITE:

        best_score = float("-inf")

        for move in board.legal_moves:

            board.push(move)

            score = minimax(board, depth - 1)

            board.pop()

            best_score = max(best_score, score)

        return best_score

    else:

        best_score = float("inf")

        for move in board.legal_moves:

            board.push(move)

            score = minimax(board, depth - 1)

            board.pop()

            best_score = min(best_score, score)

        return best_score


# ==========================
# FIND BEST MOVE
# ==========================

def find_best_move(board, depth):

    best_move = None

    if board.turn == chess.WHITE:
        best_score = float("-inf")
    else:
        best_score = float("inf")

    for move in board.legal_moves:

        board.push(move)

        score = minimax(board, depth - 1)

        board.pop()

        if board.turn == chess.WHITE:

            if score > best_score:
                best_score = score
                best_move = move

        else:

            if score < best_score:
                best_score = score
                best_move = move

    return best_move


# ==========================
# SELF PLAY
# ==========================

DEPTH = 2
MAX_MOVES = 200

board = chess.Board()

game = chess.pgn.Game()
game.headers["White"] = "PolicyNet"
game.headers["Black"] = "PolicyNet"

node = game

move_count = 0

while not board.is_game_over() and move_count < MAX_MOVES:

    move = find_best_move(board, DEPTH)

    if move is None:
        break

    board.push(move)

    node = node.add_variation(move)

    move_count += 1

    print(move_count, move)

game.headers["Result"] = board.result()

# ==========================
# SAVE PGN
# ==========================

output_file = r"D:\Coding\CNN Chess Engine\data\gmdata\selfplay.pgn"

with open(output_file, "w", encoding="utf-8") as f:
    print(game, file=f)

print("\nGame saved to:")
print(output_file)

print("\nPGN:")
print(game)