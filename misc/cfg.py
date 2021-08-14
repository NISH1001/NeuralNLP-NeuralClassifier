#!/usr/bin/env python3

import json
import os

from loguru import logger

DEFAULT_CONF = {
    "task_info": {
        "label_type": "multi_label",
        "hierarchical": True,
        "hierar_taxonomy": "data/custom.taxonomy",
        "hierar_penalty": 0.000001,
    },
    "device": "cuda",
    "model_name": "Transformer",
    "checkpoint_dir": "checkpoint_dir_rcv1",
    "model_dir": "trained_model_rcv1",
    "data": {
        "train_json_files": ["data/train.hierar.json"],
        "validate_json_files": ["data/val.hierar.json"],
        "test_json_files": ["data/test.hierar.json"],
        "generate_dict_using_json_files": True,
        "generate_dict_using_all_json_files": True,
        "generate_dict_using_pretrained_embedding": False,
        "generate_hierarchy_label": False,
        "dict_dir": "dict_rcv1",
        "num_worker": 4,
    },
    "feature": {
        "feature_names": ["token"],
        "min_token_count": 2,
        "min_char_count": 2,
        "token_ngram": 0,
        "min_token_ngram_count": 0,
        "min_keyword_count": 0,
        "min_topic_count": 2,
        "max_token_dict_size": 1000000,
        "max_char_dict_size": 150000,
        "max_token_ngram_dict_size": 10000000,
        "max_keyword_dict_size": 100,
        "max_topic_dict_size": 100,
        "max_token_len": 256,
        "max_char_len": 1024,
        "max_char_len_per_token": 4,
        "token_pretrained_file": "",
        "keyword_pretrained_file": "",
    },
    "train": {
        "batch_size": 64,
        "start_epoch": 1,
        "num_epochs": 3,
        "num_epochs_static_embedding": 0,
        "decay_steps": 1000,
        "decay_rate": 1.0,
        "clip_gradients": 100.0,
        "l2_lambda": 0.0,
        "loss_type": "BCEWithLogitsLoss",
        "sampler": "fixed",
        "num_sampled": 5,
        "visible_device_list": "0",
        "hidden_layer_dropout": 0.5,
    },
    "embedding": {
        "type": "embedding",
        "dimension": 64,
        "region_embedding_type": "context_word",
        "region_size": 5,
        "initializer": "uniform",
        "fan_mode": "FAN_IN",
        "uniform_bound": 0.25,
        "random_stddev": 0.01,
        "dropout": 0.0,
    },
    "optimizer": {
        "optimizer_type": "Adam",
        "learning_rate": 0.008,
        "adadelta_decay_rate": 0.95,
        "adadelta_epsilon": 1e-08,
    },
    "TextCNN": {"kernel_sizes": [2, 3, 4], "num_kernels": 100, "top_k_max_pooling": 1},
    "TextRNN": {
        "hidden_dimension": 64,
        "rnn_type": "GRU",
        "num_layers": 1,
        "doc_embedding_type": "Attention",
        "attention_dimension": 16,
        "bidirectional": True,
    },
    "DRNN": {
        "hidden_dimension": 5,
        "window_size": 3,
        "rnn_type": "GRU",
        "bidirectional": True,
        "cell_hidden_dropout": 0.1,
    },
    "eval": {
        "text_file": "data/test.hierar.json",
        "threshold": 0.5,
        "dir": "eval_dir",
        "batch_size": 1024,
        "is_flat": True,
        "top_k": 100,
        "model_dir": "checkpoint_dir/Transformer_best",
    },
    "TextVDCNN": {"vdcnn_depth": 9, "top_k_max_pooling": 8},
    "DPCNN": {"kernel_size": 3, "pooling_stride": 2, "num_kernels": 16, "blocks": 2},
    "TextRCNN": {
        "kernel_sizes": [2, 3, 4],
        "num_kernels": 100,
        "top_k_max_pooling": 1,
        "hidden_dimension": 64,
        "rnn_type": "GRU",
        "num_layers": 1,
        "bidirectional": True,
    },
    "Transformer": {
        "d_inner": 128,
        "d_k": 32,
        "d_v": 32,
        "n_head": 4,
        "n_layers": 1,
        "dropout": 0.1,
        "use_star": True,
    },
    "AttentiveConvNet": {
        "attention_type": "bilinear",
        "margin_size": 3,
        "type": "advanced",
        "hidden_size": 64,
    },
    "HMCN": {
        "hierarchical_depth": [0, 384, 384, 384, 384],
        "global2local": [0, 4, 55, 43, 1],
    },
    "log": {"logger_file": "log_test_rcv1_hierar", "log_level": "warn"},
}


def generate_cfg(cfg: dict, cfgpath="conf/custom.conf.json") -> None:
    logger.debug(f"cfgdict ==> {cfg}")
    assert "trainpath" in cfg
    assert "valpath" in cfg
    assert "testpath" in cfg
    assert "taxonomy" in cfg
    assert "model" in cfg

    trainpath = cfg["trainpath"]
    if isinstance(trainpath, str):
        trainpath = [trainpath]

    valpath = cfg["valpath"]
    if isinstance(valpath, str):
        valpath = [valpath]

    testpath = cfg["testpath"]
    if isinstance(testpath, str):
        testpath = [testpath]

    final_conf = DEFAULT_CONF.copy()
    final_conf["data"]["train_json_files"] = trainpath
    final_conf["data"]["validate_json_files"] = valpath
    final_conf["data"]["test_json_files"] = testpath
    final_conf["data"]["dict_dir"] = cfg.get("dict_dir", "dictdir")

    final_conf["task_info"]["hierar_taxonomy"] = cfg["taxonomy"]

    final_conf["model_name"] = cfg["model"]
    final_conf["model_dir"] = cfg.get("model_dir", "trained_model_dir")
    final_conf["checkpoint_dir"] = cfg.get("checkpoint_dir", "checkpoint_dir")
    final_conf["log"]["logger_file"] = cfg.get("logfile", "log.txt")
    final_conf["train"]["batch_size"] = cfg.get("batch_size", 64)
    final_conf["train"]["num_epochs"] = cfg.get("num_epochs", 50)
    final_conf["device"] = cfg.get("device", "cuda")

    final_conf["eval"]["text_file"] = valpath[0]
    final_conf["eval"]["model_dir"] = (
        os.path.join(final_conf["checkpoint_dir"], final_conf["model_name"]) + "_best"
    )

    print(final_conf)

    assert cfgpath.endswith(".json")

    basedir = os.path.split(cfgpath)[0]
    logger.info(f"Creating directory = {basedir}")
    os.makedirs(basedir, exist_ok=True)

    logger.info(f"Dumping configuration to {cfgpath}")
    with open(cfgpath, "w") as f:
        json.dump(final_conf, f, indent=4)
    return final_conf


def main():
    pass


if __name__ == "__main__":
    main()
