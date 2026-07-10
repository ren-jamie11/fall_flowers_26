"""Flatten the per-store JSON label outputs under Data/ into a single DataFrame.

Each JSON file maps a row-index string to {"img_link", "labels", "error"}.
This module collapses every entry across all files into one row of a pandas
DataFrame following the schema below. Missing/erroring values become NaN on a
per-cell basis, so a failure in one column never affects the others.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

# --- Schema (rename/add columns here) ---------------------------------------
COL_IMG_LINK     = "img_link"
COL_PRODUCT_TYPE = "product_type"
COL_PLANT_TYPE   = "plant_type"
COL_FLOWERS      = "flowers"
COL_LEAVES       = "leaves"
COL_FRUIT        = "fruit"

COLUMNS = [COL_IMG_LINK, COL_PRODUCT_TYPE, COL_PLANT_TYPE,
           COL_FLOWERS, COL_LEAVES, COL_FRUIT]

# Where each column is read from within a single JSON entry.
TOP_LEVEL_FIELDS = {COL_IMG_LINK: "img_link"}        # read from the entry root
LABEL_FIELDS = {                                     # read from entry["labels"]
    COL_PRODUCT_TYPE: "product_type",
    COL_PLANT_TYPE:   "plant_type",
    COL_FLOWERS:      "flowers",
    COL_LEAVES:       "leaves",
    COL_FRUIT:        "fruit",
}

DICT_COLUMNS  = [COL_FLOWERS, COL_LEAVES, COL_FRUIT]  # dict-valued label columns
COLORS_SUFFIX = "_colors"
COL_COLORS    = "colors"  # union of every {col}_colors on a row

# dict column -> name of the column holding that dict's species keys
SPECIES_COLUMNS = {COL_FLOWERS: "flower_types",
                   COL_LEAVES:  "leaf_types",
                   COL_FRUIT:   "fruit_types"}

DATA_DIR = "Data"

# The source CSVs name the image URL column differently from the JSON output.
CSV_IMG_LINK_COLUMN = "Main Img Link"  # mirrors IMG_LINK_COLUMN in ImageTextProcessor.py


def _safe_get(d, key):
    """Return d[key], or NaN if d isn't a mapping or the key is missing."""
    try:
        return d[key]
    except (KeyError, TypeError):
        return np.nan


def _row_from_entry(entry):
    """Build a single schema row from one JSON entry, isolating per-cell errors."""
    row = {}
    for col, field in TOP_LEVEL_FIELDS.items():
        row[col] = _safe_get(entry, field)

    labels = _safe_get(entry, "labels")  # NaN when entry has no/null labels
    for col, field in LABEL_FIELDS.items():
        row[col] = _safe_get(labels, field)
    return row


def build_table(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load every Data/**/*.json file and flatten its entries into a DataFrame."""
    rows = []
    for path in glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # a broken file shouldn't sink the rest

        if not isinstance(data, dict):
            continue

        for entry in data.values():
            rows.append(_row_from_entry(entry))

    return pd.DataFrame(rows, columns=COLUMNS)


def build_csv_table(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load every Data/**/*.csv file and concatenate them into one DataFrame."""
    frames = []
    for path in glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True):
        try:
            try:
                frames.append(pd.read_csv(path))
            except UnicodeDecodeError:
                # not UTF-8 (e.g. Windows-1252 export); latin-1 never fails
                frames.append(pd.read_csv(path, encoding="latin-1"))
        except (OSError, pd.errors.ParserError):
            continue  # a broken file shouldn't sink the rest

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def merge_tables(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Inner-merge the CSV catalog with the JSON labels on the image URL."""
    csv_df = build_csv_table(data_dir)
    json_df = build_table(data_dir)

    merged = pd.merge(
        csv_df, json_df,
        left_on=CSV_IMG_LINK_COLUMN, right_on=COL_IMG_LINK,
        how="inner",
    )
    # Drop the redundant duplicate key; keep the CSV's "Main Img Link".
    return merged.drop(columns=[COL_IMG_LINK])


def _unique_colors(cell):
    """Flatten a {name: [colors]} dict into an ordered list of unique colors."""
    if not isinstance(cell, dict):
        return np.nan  # NaN / non-dict cells stay NaN
    colors = []
    for values in cell.values():
        for color in values:
            if color not in colors:
                colors.append(color)
    return colors


def process_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add a {col}_colors column per dict column, plus a 'colors' union across them."""
    df = df.copy()
    for col in DICT_COLUMNS:
        if col in df.columns:
            df[f"{col}{COLORS_SUFFIX}"] = df[col].apply(_unique_colors)
    # NaN cells (label failures) aren't lists, so skip them rather than concatenate.
    df[COL_COLORS] = df[[f"{c}{COLORS_SUFFIX}" for c in DICT_COLUMNS if c in df.columns]].apply(
        lambda row: sorted({color for cell in row if isinstance(cell, list) for color in cell}), axis=1)
    return df


def _unique_species(cell):
    """Dict keys in model-emission order; [] for NaN / non-dict cells."""
    return list(cell) if isinstance(cell, dict) else []


def process_species(df: pd.DataFrame) -> pd.DataFrame:
    """Add a {singular}_types column of species keys for each dict-valued column."""
    df = df.copy()
    for col, types_col in SPECIES_COLUMNS.items():
        if col in df.columns:
            df[types_col] = df[col].apply(_unique_species)
    return df


def clean_null_rows(df, col = "plant_type"):
    """
    Remove rows where `plant_type` is null or empty/whitespace-only.
    """
    is_blank = df[col].isna() | (
        df[col].astype(str).str.strip() == ''
    )
    return df.loc[~is_blank].reset_index(drop=True)

def process_table(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Build the merged table with all derived color and species columns."""
    res = process_species(process_colors(merge_tables(data_dir)))
    res = clean_null_rows(res)
    return res


if __name__ == "__main__":
    df = process_table()
    print(df)
    print(df.columns)
    print("shape:", df.shape)

    df.to_parquet("Data/Fall Greenery/streamlit_df_output.parquet")
