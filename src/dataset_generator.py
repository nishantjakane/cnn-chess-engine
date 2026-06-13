import chess.pgn

def extract_positions(game):
    positions = []

    board = game.board()

    ply_number =0

    for move in game.mainline_moves():
        board.push(move)
        ply_number +=1

        positions.append((board.fen(),ply_number))

    return positions

with open("src/test.pgn") as pgn:
    game = chess.pgn.read_game(pgn)

positions = extract_positions(game)

print(len(positions))
print(positions[0])