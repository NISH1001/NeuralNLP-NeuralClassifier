#!/usr/bin/env python3

from loguru import logger
from sklearn.model_selection import train_test_split

from misc import datatools, taxonomy


def main():
    kw_path = "data/earth-science/keywords.txt"
    data_path = "data/earth-science/data.csv"

    keywords = taxonomy.load_keywords(kw_path)
    taxonomy_path = "data/earth-science/custom.taxonomy"
    taxonomy.generate_taxonomy(keywords, taxonomy_path)

    df = datatools.load_data(data_path)
    keywords_str = df["keywords"]

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
    _ = datatools.generate_data(data_train, "data/earth-science/train.hierar.json")
    _ = datatools.generate_data(data_val, "data/earth-science/val.hierar.json")
    _ = datatools.generate_data(data_test, "data/earth-science/test.hierar.json")


if __name__ == "__main__":
    main()
