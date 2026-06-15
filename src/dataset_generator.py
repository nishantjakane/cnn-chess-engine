import chess.pgn
import chess.engine
import chess
import random
import csv
import pandas as pd

#Not in use as I will be using the data from https://www.kaggle.com/datasets/lichess/chess-evaluations


STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
STOCKFISH_DEPTH = 12

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def extract_positions(game):
    positions = []

    board = game.board()

    ply_number =0

    for move in game.mainline_moves():
        board.push(move)
        ply_number +=1

        positions.append((board.fen(),ply_number))

    return positions

def sample_positions(positions):
    total = len(positions)
    if total ==0:
        return []

    opening_end = total//4 # first 25% of moves we are assuming to be opening
    middlegame_end = (3*total)//4 # next 50% of moves we are assuming to be middle game
    
    opening=positions[:opening_end]
    middlegame=positions[opening_end:middlegame_end]
    endgame= positions[middlegame_end:]

    sampled=[]

    sampled.extend(
        random.sample(
            opening,min(2,len(opening))
        )
    )

    sampled.extend(
        random.sample(
            middlegame,min(2,len(middlegame))
        )
    )

    sampled.extend(
        random.sample(
            endgame,min(1,len(endgame))
        )
    )
    # 2 opening pos
    # 2 middle game pos
    # 1 endgame pos
    # 5 samples from a game

    return sampled



def evaluate_position(fen,engine,depth=STOCKFISH_DEPTH):
    board = chess.Board(fen)
    
    info = engine.analyse(board,chess.engine.Limit(depth=depth))

    score = info["score"].white()
    mate_distance = score.mate()
    if score.is_mate(): 
        sign =1
        if mate_distance > 0:
            sign =1
        else:
            sign =-1
    
        return sign*(1000-abs(mate_distance)) # If stockfish says there is a mate in 2 for white then this will return +998
    
    return score.cp


def generate_dataset_rows(game,engine,depth=STOCKFISH_DEPTH):
    positions = extract_positions(game)

    samples = sample_positions(positions)
    
    eval_data = []

    for position in samples:
        ply_number = position[1]
        score = evaluate_position(position[0],engine,depth)
        eval_data.append((position[0],score,depth,ply_number))

    return eval_data

def save_rows_to_csv(rows,filepath):
    df = pd.DataFrame(rows,columns=("fen","eval_cp","depth","ply"))
    df.to_csv(filepath,index=False)

rows = generate_dataset_rows(game,engine)
save_rows_to_csv(rows,"data/hi.csv")

for row in rows:
    print(row)


engine.quit()