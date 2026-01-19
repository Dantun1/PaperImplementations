import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from Seq2Seq.data.vocab import Vocabulary

class EnglishFrenchDataset(Dataset):
    """
    Dataset for English to French machine translation.
    """
    def __init__(self, clean_df: pd.DataFrame,  src_vocab: Vocabulary, target_vocab: Vocabulary, src_col: str = "English", target_col: str = "French") -> None:
        """

        Args:
            clean_df (pd.DataFrame): Dataframe containing the source and target columns
            vocabulary (Vocabulary):
        """
        self.df = clean_df
        self.src_vocab = src_vocab
        self.src = src_col
        self.target_vocab = target_vocab
        self.target = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get English + French text sample
        english_sentence = self.df.iloc[idx][self.src]
        french_sentence = self.df.iloc[idx][self.target]

        # Get Special character tokens

        eos_idx = self.src_vocab.stoi["<EOS>"]
        sos_idx = self.target_vocab.stoi["<SOS>"]



        # Get the raw indices
        x = self.src_vocab.get_indices(english_sentence)
        y = self.target_vocab.get_indices(french_sentence)

        # Add EOS + Reverse the input
        x.append(eos_idx)
        x = list(reversed(x))

        # Get the teacher force input (sos and shifted right)
        x_fr = [sos_idx] + y

        # Add eos to output
        y.append(eos_idx)

        # Return as tensors
        return torch.tensor(x),torch.tensor(x_fr), torch.tensor(y)


class CollateFunctor:
    """Callable to be passed to DataLoader to apply padding"""
    def __init__(self, pad_idx: int = 0) -> None:
        self.pad_idx = pad_idx
    def __call__(self, batch):

        x_en = [b[0] for b in batch]
        x_fr = [b[1] for b in batch]
        y = [b[2] for b in batch]
        # Apply necessary padding, left for input and right for output/teacher force input.
        x_en_padded = pad_sequence(x_en, batch_first=True, padding_value=self.pad_idx, padding_side="left")
        x_fr_padded = pad_sequence(x_fr, batch_first=True, padding_value=self.pad_idx, padding_side="right")
        y_padded = pad_sequence(y, batch_first=True, padding_value=self.pad_idx, padding_side="right")

        return x_en_padded, x_fr_padded, y_padded











