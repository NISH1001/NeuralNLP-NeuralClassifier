#!/usr/bin/env python3

from typing import List

import os

from loguru import logger
from sklearn.model_selection import train_test_split

from misc import cfg, datatools, taxonomy, utils

def tokenizer(text: str) -> List[str]:
    tokens = map(str.lower, text.split())
    return list(tokens)

def main():
    kw_path = "data/earth-science/keywords.txt"
    data_path = "data/earth-science/data.csv"
    taxonomy_path = "data/earth-science/custom.taxonomy"

    basedir = "data/earth-science/"
    iat = utils.IAmTime()
    outdir = os.path.join(
        basedir, f"{iat.year}-{iat.month}-{iat.day}-{iat.hour}-{iat.minute}"
    )
    utils.create_dirs(outdir)

    # train_path = "data/earth-science/train.hierar.json"
    # val_path = "data/earth-science/val.hierar.json"
    # test_path = "data/earth-science/test.hierar.json"
    # checkpoint_dir = "checkpoint_dir_custom/"
    # dict_dir = "data/earth-science/dict_dir/"
    # cfgpath = "data/earth-science/conf.json"
    # logfile = "data/earth-science/log.txt"
    train_path = os.path.join(outdir, "train.hierar.json")
    val_path = os.path.join(outdir, "val.hierar.json")
    test_path = os.path.join(outdir, "test.hierar.json")
    checkpoint_dir = os.path.join(outdir, "checkpoints")
    dict_dir = os.path.join(outdir, "dict_dir")
    cfgpath = os.path.join(outdir, "conf.json")
    logfile = os.path.join(outdir, "log.txt")

    keywords = taxonomy.load_keywords(kw_path)
    # taxonomy.generate_taxonomy(keywords, taxonomy_path)
    taxonomy.generate_taxonomy_new(keywords, taxonomy_path)

    df = datatools.load_data(data_path)
    #df = datatools.standardize_data(df, tokenizer)
    df = datatools.standardize_data(df, tokenizer)
    logger.debug(df.head())
    logger.debug(df.iloc[0])
    logger.debug(df.iloc[0]["labels_tokenized"])
    logger.debug(df.iloc[0]["labels_standard"])

    data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)
    data_val, data_test = train_test_split(data_test, test_size=0.5, random_state=42)
    logger.debug(
        f"[Train={data_train.shape}], [Val={data_val.shape}], [Test={data_test.shape}]"
    )

    # _ = datatools.generate_data(df, "data/earth-science/data.hierar.json")
    _ = datatools.generate_data(df, os.path.join(outdir, "data.hierar.json"))
    _ = datatools.generate_data(data_train, train_path)
    _ = datatools.generate_data(data_val, val_path)
    _ = datatools.generate_data(data_test, test_path)

    cfg.generate_cfg(
        {
            "trainpath": train_path,
            "testpath": test_path,
            "valpath": val_path,
            "taxonomy": taxonomy_path,
            "checkpoint_dir": checkpoint_dir,
            "model": "Transformer",
            #"model": "HMCN",
            #"model": "TextRNN",
            "batch_size": 64,
            "num_epochs": 50,
            "dict_dir": dict_dir,
            "device": "cuda",
            "logfile": logfile,
        },
        cfgpath=cfgpath,
    )


if __name__ == "__main__":
    main()
