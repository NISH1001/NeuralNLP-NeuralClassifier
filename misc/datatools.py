#!/usr/bin/env python3

import json
from typing import Callable, List

import pandas as pd
from loguru import logger

from .taxonomy import process_keywords_str, process_single_kws


def parse_kws(kw_str, level=2):
    res = kw_str.split(",")
    res = map(lambda kw: [_.strip().lower() for _ in kw.split(">")], res)
    res = map(lambda x: x[level if level < len(x) else len(x) - 1], res)
    return list(set(res))


def load_data(path: str, level: int = 0) -> pd.DataFrame:
    logger.info(f"Loading data from {path}. [KW Level={level}]")
    df = pd.read_csv(path)
    df = df.rename(columns={"desc": "text"})
    df["text"] = df["text"].apply(str.strip)
    df["labels"] = df["keywords"].apply(lambda x: parse_kws(x, level))
    df["textlen"] = df["text"].apply(len)
    df = df[df["textlen"] > 0]

    logger.debug(f"df shape : {df.shape}")
    return df


def tokenize_custom(text: str) -> List[str]:
    return text.split()


def standardize_data(df: pd.DataFrame, tokenizer_func: Callable = None) -> pd.DataFrame:
    df = df.copy()
    if "desc" in df:
        df = df.rename(columns={"desc": "text"})

    assert "text" in df
    assert "keywords" in df
    tokenizer_func = tokenizer_func or tokenize_custom

    df["tokens"] = df["text"].apply(tokenizer_func)

    def _process_single_row_kw(keywords: List[str]):
        keywords = list(map(process_single_kws, keywords))
        keywords = list(map(lambda kws: "--".join(kws), keywords))
        return keywords

    df["labels_tokenized"] = df["keywords"].apply(process_keywords_str)
    df["labels_standard"] = df["labels_tokenized"].apply(_process_single_row_kw)
    return df.reset_index(drop=True)


def generate_data(df: pd.DataFrame, outpath: str) -> List[dict]:
    logger.info("Generating data!")
    assert "text" in df
    assert "labels_standard" in df
    assert "tokens" in df
    data = []
    for _df in df.itertuples():
        tokens = _df.tokens
        labels = _df.labels_standard
        data.append(
            {
                "doc_label": labels,
                "doc_token": tokens,
                "doc_keyword": [],
                "doc_topic": [],
            }
        )
    logger.debug(f"NData = {len(data)}")
    if outpath:
        logger.info(f"Writing data to {outpath}")
        with open(outpath, "w") as f:
            for dct in data:
                json.dump(dct, f)
                f.write("\n")
    return data


def main():
    pass


if __name__ == "__main__":
    main()
