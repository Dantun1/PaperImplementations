import pandas as pd

MAX_LEN = 64

def load_and_clean_df(csv_filepath: str) -> pd.DataFrame:
    """
    Load and clean the raw csv dataset with English and French columns.

    1) Remove/translate invisible characters
    2) Filter out long examples

    Args:
        csv_filepath: path to raw English/French CSV

    Returns:
        clean_df: Final clean DataFrame
    """
    raw_df = pd.read_csv(csv_filepath)

    replacements = {
        '\xa0': ' ',
        '\xad': None,
        '\u2009': ' ',
        '\u200b': '',
        '\u202f': ' ',
    }
    table = str.maketrans(replacements)

    for c in raw_df.columns:
        raw_df[c] = raw_df[c].str.translate(table)

    clean_df = raw_df.loc[raw_df["French"].str.len() <= MAX_LEN,: ].copy()


    return clean_df
