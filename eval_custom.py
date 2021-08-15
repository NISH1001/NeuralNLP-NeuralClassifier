#!/usr/bin/env python3

import json
from pprint import pprint

from inferencer import Inferencer


def main():
    inferencer = Inferencer(
        config="data/earth-science/2021-08-15-18-57/conf.json",
        modelpath="data/earth-science/2021-08-15-18-57/checkpoints/Transformer_best",
    )
    # inf_docs = inferencer.predict_from_texts(
    #     ["solar storm is creating a big hurricane"]
    # )
    inf_docs = inferencer.predict_from_file(
        "data/earth-science/2021-08-15-18-57/test.hierar.json"
    )

    for inf in inf_docs:
        if not inf["predictions"]:
            continue
        print("-" * 30)
        print(inf["text"])
        gts = inf["gts"]
        print(f"GTS ==> {gts}")
        for p in inf["predictions"]:
            print(p["label"], p["prob"])

    # dumppath = "data/earth-science/2021-08-15-18-57/inference.test.json"
    # with open(dumppath, "w") as f:
    #     json.dump(inf_docs, f)


if __name__ == "__main__":
    main()
