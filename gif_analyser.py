"""
Utility for analysing structure of GIF files.
"""

import functools
import os
import subprocess

from wand.image import Image
from gif_editor import optimize

from typing import Tuple

Stats = Tuple[str, float, float, float, float]


def get_file_name(fp: str) -> str:
    """
    Get file name from a long file path.

    :param fp: file path
    :return: name of the file, not including its path
    """
    return fp.split(os.path.sep)[-1]


def transparent_percent(fp: str, magick_source: str) -> float:
    """
    Get the percentage of transparent pixels in an image. This function applies
    to both single-frame and multi-frame images.

    :param fp: file path of the image
    :param magick_source: path to ImageMagick
    :return: a number in percent, representing the percentage of transparent
        pixels in the image
    """
    instruction = [magick_source, "convert", fp, "-alpha", "extract", "-negate",
        "-format", "%[fx:mean*w*h] ", "info:"]
    process = subprocess.run(instruction, capture_output=True, encoding="utf-8")
    transparent = functools.reduce(
        lambda a, x: a + int(x), process.stdout.split(), 0
    )
    with Image(filename=fp) as im:
        transparent /= sum(frame.width * frame.height for frame in im.sequence)
    return transparent * 100


def compression_stats(fp: str, out: str) -> Stats:
    """
    Generate the compression statistics of compressing an image with various
    optimization technique detailed in gif_editor.optimize(). This function
    requires additional space to create a new file after optimizing.

    :param fp: file path of the image
    :param out: path of the compressed image
    :return: a tuple containing the name of the image and 4 numbers indicating
        the percentage of size gained after compressing the image with each of
        the 4 optimization techniques in gif_editor.optimize()
    """

    init_size = os.stat(fp).st_size
    data = [get_file_name(fp), init_size >> 10]
    for level in range(1, 5):
        optimize(fp, out, level)
        change = init_size - os.stat(out).st_size
        data.append(change / init_size * 100)
    return tuple(data)
