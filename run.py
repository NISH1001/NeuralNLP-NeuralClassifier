#!/usr/bin/env python3

from loguru import logger
from sklearn.model_selection import train_test_split

from misc import cfg, datatools, taxonomy


def main():
    kw_path = "data/earth-science/keywords.txt"
    data_path = "data/earth-science/data.csv"

    train_path = "data/earth-science/train.hierar.json"
    val_path = "data/earth-science/val.hierar.json"
    test_path = "data/earth-science/test.hierar.json"
    taxonomy_path = "data/earth-science/custom.taxonomy"
    checkpoint_dir = "checkpoint_dir_custom/"
    dict_dir = "data/earth-science/dict_dir/"
    cfgpath = "data/earth-science/conf.json"
    logfile = "data/earth-science/log.txt"

    keywords = taxonomy.load_keywords(kw_path)
    taxonomy.generate_taxonomy(keywords, taxonomy_path)

    df = datatools.load_data(data_path)
    df = datatools.standardize_data(df)
    logger.debug(df.head())
    logger.debug(df.iloc[0])
    logger.debug(df.iloc[0]["labels_tokenized"])
    logger.debug(df.iloc[0]["labels_standard"])

    data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)
    data_val, data_test = train_test_split(data_test, test_size=0.5, random_state=42)
    logger.debug(
        f"[Train={data_train.shape}], [Val={data_val.shape}], [Test={data_test.shape}]"
    )

    _ = datatools.generate_data(df, "data/earth-science/data.hierar.json")
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
            "batch_size": 64,
            "num_epochs": 50,
            "dict_dir": dict_dir,
            "device": "cpu",
            "logfile": logfile,
        },
        cfgpath=cfgpath,
    )


if __name__ == "__main__":
    main()
