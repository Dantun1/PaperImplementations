import torch
from torch.utils.data import Dataset
import pandas as pd

from Seq2Seq.data.vocab import Vocabulary


class EnglishFrenchDataset(Dataset):
    """
    Dataset for English to French machine translation.
    """
    def __init__(self, clean_df: pd.DataFrame,  vocabulary: Vocabulary, src_col: str = "English", target_col: str = "French") -> None:
        """

        Args:
            clean_df (pd.DataFrame): Dataframe containing the source and target columns
            vocabulary (Vocabulary):
        """
        self.df = clean_df
        self.vocab = vocabulary

    def __len__(self):
        return len(self.df)