import chess
from encoder import board_to_tensor
import torch
from torch.utils.data import Dataset
import pandas as pd



class ChessDataset(Dataset):
    def __init__(self,df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        fen = row["fen"]
        cp = row["cp"]
        mate = row["mate"]
        
        board = chess.Board(fen)
        
        tensor = board_to_tensor(board)

        target = torch.tensor(evaluation_to_target(cp,mate),dtype=torch.float32)

        return tensor , target

def evaluation_to_target(cp,mate):
    sign =1
    eval_cp =0


    if mate is not None:
        if mate < 0 :
            sign =-1
        eval_cp = sign * (1000 - abs(mate))
    else:
        eval_cp =cp
    
    eval_cp = max(-1000,min(1000,eval_cp))
    target = eval_cp/1000.0
    return target

import pandas as pd

df = pd.DataFrame([
    {
        "fen": chess.STARTING_FEN,
        "cp": 47,
        "mate": None
    }
])

dataset = ChessDataset(df)

tensor, target = dataset[0]

print(tensor.shape)
print(target)
print(type(target))