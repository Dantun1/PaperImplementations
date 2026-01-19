import pandas as pd
from typing import Self, Mapping, Sequence


class Vocabulary:
    """Vocabulary class responsible for mapping storage and translation."""
    def __init__(self, stoi: Mapping[str,int], itos: Mapping[str,int]) -> None:
        self.stoi = stoi
        self.itos = itos

    def __len__(self) -> int:
        return len(self.stoi)

    def get_indices(self, text: str) -> list[int]:
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(f"Character '{e.args[0]}' not found in vocabulary.") from e

    def get_string(self, indices: Sequence[int]) -> str:
        try:
            return "".join([self.itos[i] for i in indices])
        except KeyError as e:
            raise ValueError(f"Index '{e.args[0]}' not found in vocabulary. Expected indices from 0 to {len(self)}.") from e

    @classmethod
    def from_series(cls, string_series: pd.Series) -> Self:
        """
        Given a pd series of sentences, create a vocabulary object containing translation info
        """
        unique_chars = sorted(set("".join(string_series)))
        special_tokens = ['<PAD>', '<SOS>', '<EOS>']

        # Integer to string encodings
        itos = {idx + len(special_tokens): char for idx, char in enumerate(unique_chars)}

        for i, token in enumerate(special_tokens):
            itos[i] = token

        # String to integer encodings
        stoi = {c: i for i, c in itos.items()}

        return cls(stoi, itos)




