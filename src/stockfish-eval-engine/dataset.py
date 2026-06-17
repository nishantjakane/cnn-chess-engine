import chess
from encoder import board_to_tensor
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        fen = row["FEN"]
        evaluation = row["Evaluation"]

        board = chess.Board(fen)

        tensor = board_to_tensor(board)

        target = torch.tensor(
            evaluation_to_target(evaluation),
            dtype=torch.float32
        )

        return tensor, target


def evaluation_to_target(eval_str):
    eval_str = str(eval_str)

    if eval_str.startswith("#"):
        mate_distance = int(eval_str[1:])

        sign = 1
        if mate_distance < 0:
            sign = -1

        eval_cp = sign * (1000 - abs(mate_distance))

    else:
        eval_cp = int(eval_str)

    eval_cp = max(-1000, min(1000, eval_cp))

    return eval_cp / 1000.0