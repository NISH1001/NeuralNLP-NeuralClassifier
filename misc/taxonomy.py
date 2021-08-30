#!/usr/bin/env python3

from typing import List

from loguru import logger


def load_keywords(path) -> List[List[str]]:
    res = []
    with open(path) as f:
        text = f.read().strip()
        tags_str = text.split(",")
        res = map(lambda t: [_.strip().lower() for _ in t.split(">")], tags_str)
        res = filter(lambda x: len(x) > 0, res)
        res = list(res)
    nkws = len(set([kw for kws in res for kw in kws]))
    logger.debug(f"Unique labels => {nkws}")
    return res


def process_single_kws(kws: List[str]) -> List[str]:
    kws = map(lambda x: "_".join(x.split()).upper(), kws)
    kws = list(kws)
    return kws


def process_keywords_str(keywords_str: str, label_sep=",", hierar_sep=">") -> List[str]:
    """
    Convert the string labels (multiple) into proper keywords
    """
    keywords = keywords_str.upper().split(label_sep)
    keywords = map(str.strip, keywords)
    keywords = map(lambda x: [_.strip() for _ in x.split(hierar_sep)], keywords)
    keywords = list(keywords)
    return keywords


def get_levels(keywords: List[List[str]], level: int) -> List[str]:
    kws = map(lambda x: x[level if level < len(x) else len(x) - 1], keywords)
    kws = list(set(kws))
    return kws


def generate_taxonomy(keywords: List[List[str]], outpath: str) -> None:
    keywords_processed = []
    for kws in keywords:
        kws = process_single_kws(kws)
        keywords_processed.append(kws)
    keywords = keywords_processed or keywords
    roots = get_levels(keywords, level=0)
    roots = ["Root"] + roots
    logger.debug(f"[Nroots = {len(roots)}] == {roots}")
    logger.info(f"Writing the custom taxonomy to {outpath}")
    with open(outpath, "w") as f:
        f.write(" ".join(roots))
        f.write("\n")
        for kws in keywords:
            f.write(" ".join(kws))
            f.write("\n")


def build_trie(keywords) -> dict:
    """
    This generates a dictionary with:
        - key as a parent node string
        - value as a list of its immediate children
    """
    keywords_processed = []
    for kws in keywords:
        kws = process_single_kws(kws)
        keywords_processed.append(kws)
    keywords = keywords_processed
    roots = get_levels(keywords, level=0)
    trie = {}
    trie["Root"] = roots

    for kws in keywords:
        for parent, child in zip(kws, kws[1:]):
            if parent not in trie:
                trie[parent] = [child]
            elif parent in trie and child not in trie[parent]:
                trie[parent].append(child)
    return dict(trie)


def generate_taxonomy_new(keywords: List[List[str]], outpath: str) -> None:
    trie = build_trie(keywords)
    logger.debug(f"[Ntrie = {len(trie)}]")
    roots = trie.get("Root", [])
    logger.debug(f"[Nroots = {len(roots)}] == {roots}")
    if not roots:
        raise ValueError("No root nodes found. Terminating!")
    logger.info(f"Writing the custom taxonomy to {outpath}")
    with open(outpath, "w") as f:
        for node, children in trie.items():
            f.write("\t".join([node] + children))
            f.write("\n")


def main():
    pass


if __name__ == "__main__":
    main()
