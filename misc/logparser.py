#!/usr/bin/env python3

import re

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def parse_logs(logpath="log.txt") -> dict:
    metrics = dict(train=[], validate=[], test=[])
    lines = []
    best = ""
    with open(logpath) as f:
        lines = f.readlines()
    pattern_metrics = re.compile(
        r".*WARNING\s+(?P<ptype>.*)\s+performance.*at\s+epoch\s+(?P<epoch>\d+).*precision.*(?P<precision>\d+\.\d+).*recall.*(?P<recall>\d+\.\d+).*fscore.*(?P<f1>\d+\.\d+).*macro.*",
        flags=re.IGNORECASE,
    )
    pattern_loss = re.compile(r"Loss.*(?P<loss>\d+\.\d+)", flags=re.IGNORECASE)
    for line1, line2 in zip(lines, lines[1:]):
        line1 = line1.lower()
        line2 = line2.lower()
        mdict = {}
        loss = 0
        for match in pattern_metrics.finditer(line1):
            mdict = match.groupdict()
        if not mdict:
            continue
        for k in mdict:
            try:
                mdict[k] = float(mdict[k])
            except:
                continue
        if "best" in line1:
            best = mdict
            continue
        for match in pattern_loss.finditer(line2):
            ldict = match.groupdict()
            loss = float(ldict["loss"])
        mdict["loss"] = loss
        ptype = mdict.pop("ptype")
        metrics[ptype].append(mdict)
    metrics["best"] = best
    return metrics


def plot(metrics, ptype="train", mtype="f1"):
    if ptype in metrics:
        metrics = metrics[ptype]
    logger.info(f"Plotting metrics for {mtype}")
    metrics = sorted(metrics, key=lambda x: x["epoch"])
    scores = map(lambda x: x[mtype], metrics)
    scores = list(scores)
    epochs = list(map(lambda x: x["epoch"], metrics))
    plt.figure(figsize=(15, 7))
    plt.title(f"{ptype.upper()} {mtype} plot")
    plt.plot(epochs, scores)


def plot_overlayed(metrics, mtype="f1", dpi=200, figsize=(15, 7)):
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ptypes = ["train", "test", "validate"]
    colors = ["red", "green", "blue"]
    for color, ptype in zip(colors, ptypes):
        m = metrics[ptype].copy()
        m = sorted(m, key=lambda x: x["epoch"])
        scores = list(map(lambda x: x[mtype], m))
        epochs = list(map(lambda x: x["epoch"], m))
        ax.plot(epochs, scores, color=color, label=ptype)

    ax.legend(loc="upper right", frameon=False)
    ax.margins(0)
    ax.set_title(f"{mtype} plot")
    plt.show(ax)

    canvas.draw()
    image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    return image_from_plot


def main():
    pass


if __name__ == "__main__":
    main()
