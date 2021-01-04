"""
Collection of GIF editing utilities. This module makes use of wand and gifsicle
for GIF editing.
"""

import subprocess

from wand.color import Color
from wand.drawing import Drawing
from wand.image import BaseImage, Image

from typing import Tuple


class GIFImage(Image):
    """Extension of wand.image.Image for GIF file editing functionality."""

    def get_frame(self, frame: int) -> BaseImage:
        """
        Get a specific frame of a GIF file.
        
        :param frame: frame number
        :return: BaseImage object representing the frame
        """
        return self.sequence[frame]

    def get_pixel(self, frame: int, x: int, y: int) -> Color:
        """
        Get a pixel at a specific position of a specified frame.
        
        :param frame: frame number
        :param x: x-coordinate
        :param y: y-coordinate
        :return: color at specified pixel
        """
        return self.get_frame(frame).clone()[x, y]
    
    def set_pixel(self, frame: int, x: int, y: int, color: Color) -> None:
        """
        Set the pixel at a specific position of a specified frame to a
        particular color.

        :param frame: frame number
        :param x: x-coordinate
        :param y: y-coordinate
        :param color: color to set at pixel
        """
        with Drawing() as draw:
            draw.fill_color = color
            draw.point(x, y)
            with self.get_frame(frame) as im:
                draw(im)

    def clone_frame(self, frame: int) -> Tuple[Image, Image]:
        """
        Clone a frame and modify their delay times such that the sum of the
        delay times of the original frame and its clone are equal to the
        original delay time.

        :param frame: frame number
        :return: tuple containing two Images, both adjusted to add up to the
            original delay time
        """
        to_clone = self.get_frame(frame)
        im1, im2 = to_clone.clone(), to_clone.clone()
        im2.delay >>= 1
        im1.delay -= im2.delay
        return im1, im2

    def optimize_layers(self) -> None:
        """
        Applies the corresponding image optimization detailed in the wand.image
        documentation while also applying Image.coalesce() and fixing the frame
        delays.
        """
        delays = [frame.delay for frame in self.sequence]
        self.coalesce()
        super().optimize_layers()
        for frame, d in zip(self.sequence, delays):
            frame.delay = d
    
    def optimize_layers_and_transparency(self) -> None:
        """
        Applies the corresponding image optimizations detailed in the wand.image
        documentation.
        """
        self.optimize_layers()
        self.optimize_transparency()


def lzw_optimize(fp: str, out: str) -> None:
    """
    Applies the gifsicle LZW optimization to GIF files, detailed here:
    https://legacy.imagemagick.org/Usage/anim_opt/#opt_lzw

    :param fp: path to GIF file
    :param out: path to output GIF file
    """
    subprocess.run(["gifsicle", "-O2", fp, "-o", out])


def optimize(fp: str, out: str, level: int = 3) -> None:
    """
    Apply optimization to a GIF file to a specific level.

    Levels:
    1: Layer optimization
    2: Transparency optimization
    3: Layer and transparency optimization
    4: LZW optimization

    :param fp: path to GIF file
    :param out: path to output GIF file
    :param level: level of optimization
    """
    if level not in range(1, 5):
        raise ValueError("Optimization level must be between 1 and 4")
    if level == 4:
        lzw_optimize(fp, out)
    else:
        with GIFImage(filename=fp) as im:
            funcs = [
                im.optimize_layers,
                im.optimize_transparency,
                im.optimize_layers_and_transparency,
            ]
            funcs[level - 1]()
            im.save(filename=out)
