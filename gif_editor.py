import subprocess

from collections import deque
from wand.color import Color
from wand.drawing import Drawing
from wand.image import Image

from typing import Tuple


class GIFImage(Image):
    def get_frame(self, frame: int) -> Image:
        return self.sequence[frame]

    def get_pixel(self, frame: int, x: int, y: int) -> Color:
        return self.get_frame(frame).clone()[x, y]
    
    def set_pixel(self, frame: int, x: int, y: int, color: Color) -> None:
        with Drawing() as draw:
            draw.fill_color = color
            draw.point(x, y)
            with self.get_frame(frame) as im:
                draw(im)

    def clone_frame(self, frame: int) -> Tuple[Image, Image]:
        to_clone = self.get_frame(frame)
        im1, im2 = to_clone.clone(), to_clone.clone()
        im2.delay >>= 1
        im1.delay -= im2.delay
        return im1, im2
    
    def clone_frames(self) -> None:
        queue = deque()
        frames_to_clone = self.sequence[:-1]
        for i in range(len(frames_to_clone)):
            for fr in self.clone_frame(i):
                queue.append(fr)
        queue.append(self.get_frame(-1))
        for i in range(len(frames_to_clone)):
            self.sequence[i] = queue.popleft()
        self.sequence[-1] = queue.popleft()
        self.sequence.extend(queue)

    def optimize_layers(self) -> None:
        delays = [frame.delay for frame in self.sequence]
        self.coalesce()
        super().optimize_layers()
        for frame, d in zip(self.sequence, delays):
            frame.delay = d
    
    def optimize_layers_and_transparency(self) -> None:
        self.optimize_layers()
        self.optimize_transparency()


def lzw_optimize(fp: str, out: str):
    subprocess.run(["gifsicle", "-O2", fp, "-o", out])


def optimize(fp: str, out: str, level: int = 3):
    if level not in range(1, 5):
        raise ValueError("Optimization level must be between 1 and 3")
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
