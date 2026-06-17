import chess.pgn
from pathlib import Path

games =0
positions=0
unique_moves = set()

pgn_dir = Path(r"D:\Coding\CNN Chess Engine\data\gmdata")

for pgn_dir in pgn_dir.glob("*.pgn"):
    
    with open(pgn_dir,encoding="utf-8") as pgn:
        print(pgn_dir)
        while True:

            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            games+=1

            for move in game.mainline_moves():
                positions+=1
                unique_moves.add(move.uci())
        
        print(f"Games: {games:,}")
        print(f"Positions: {positions:,}")
        print(f"Unique Moves: {len(unique_moves):,}")
print(f"Games: {games:,}")
print(f"Positions: {positions:,}")
print(f"Unique Moves: {len(unique_moves):,}")