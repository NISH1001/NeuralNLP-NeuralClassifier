#!/usr/bin/env python3


import cv2
from loguru import logger

try:
    from matplotlib import pyplot as plt

    MATPLOTLIB = True
except ModuleNotFoundError:
    MATPLOTLIB = False

logger.debug(f"MATPLOTLIB = {MATPLOTLIB}")


def show_image(img, size=(20, 15)):
    """Display an image."""
    if not MATPLOTLIB:
        return
    plt.figure(figsize=size)
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)


def main():
    pass


if __name__ == "__main__":
    main()
