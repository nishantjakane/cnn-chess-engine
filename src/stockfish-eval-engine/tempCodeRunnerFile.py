game = chess.pgn.Game()

node = game

while not board.is_game_over():

    move = find_best_move(board, depth=2)

    board.push(move)

    node = node.add_variation(move)

print(game)