import functools
import os
import subprocess

from wand.image import Image
from gif_editor import optimize

from typing import Tuple


def get_file_name(fp: str) -> str:
    return fp.split(os.path.sep)[-1]


def transparent_percent(fp: str, magick_source: str) -> float:
    instruction = [magick_source, "convert", fp, "-alpha", "extract", "-negate",
        "-format", "%[fx:mean*w*h] ", "info:"]
    process = subprocess.run(instruction, capture_output=True, encoding="utf-8")
    transparent = functools.reduce(lambda a, x: a + int(x), process.stdout.split(), 0)
    with Image(filename=fp) as im:
        transparent /= sum(frame.width * frame.height for frame in im.sequence)
    return transparent * 100


def compression_stats(fp: str, out: str) -> Tuple[str, float, float, float, float]:
    init_size = os.stat(fp).st_size
    data = [get_file_name(fp), init_size >> 10]
    for level in range(1, 5):
        optimize(fp, out, level)
        change = init_size - os.stat(out).st_size
        data.append(change / init_size * 100)
    return tuple(data)
