from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"


def load_data(data_path: Path):
    """Loads data from feather file and returns dataframe"""
    return pd.read_feather(data_path)


def save_data(df: pd.DataFrame, output_path: Path):
    """Saves dataframe to feather file"""
    df.to_feather(output_path)


def remove_empty_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with empty sentences from the dataframe."""
    return df[df["sent_text"].str.strip() != ""].reset_index(drop=True)


def main():
    df = load_data(INPUT_PATH)
    df_cleaned = remove_empty_sentences(df)
    save_data(df_cleaned, OUTPUT_PATH)


if __name__ == "__main__":
    main()
