import PIL.Image
import PIL.ImageSequence
import numpy as np
import re
import subprocess

from collections import deque
from wand.color import Color
from wand.drawing import Drawing
from wand.image import BaseImage, Image

from typing import Callable, Iterable, Tuple, TypeVar

T = TypeVar("T")


class GIFImage(Image):
    def get_frame(self, frame: int) -> BaseImage:
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


def identity(x: T) -> T:
    return x


def min_early_break(iterable: Iterable[T], break_when: float, key: Callable = identity) -> T:
    iterable_iter = iter(iterable)
    min_elem = next(iterable_iter)
    for elem in iterable_iter:
        if key(min_elem) <= break_when:
            break
        if key(elem) <= key(min_elem):
            min_elem = elem
    return min_elem


def preprocess(fp: str, out: str) -> None:
    queue = deque()
    gifsicle_commands = ["gifsicle"]
    magick_command = ["./magick", "identify", "-verbose", None]
    pattern = re.compile(r"(?<=Transparent color: )(\w)+(\(.+\))?")
    with GIFImage(filename=fp) as wand_im:
        with PIL.Image.open(fp) as pil_im:
            pil_iter = PIL.ImageSequence.Iterator(pil_im)
            pil_im_frames = (next(pil_iter) for _ in range(pil_im.n_frames - 1))
            for frame_no, pil_frame in enumerate(pil_im_frames):
                wand_frame, wand_clone = wand_im.clone_frame(frame_no)
                cur_frame_no = frame_no * 2
                clone_frame_no = cur_frame_no + 1
                trans_idx = pil_frame.info.get("transparency")
                if trans_idx is not None:
                    try:
                        wand_prev_arr = np.array(queue[-1])
                    except IndexError:
                        pass
                    else:
                        magick_command[-1] = f"{fp}[{frame_no}]"
                        magick_run = subprocess.run(magick_command, capture_output=True, encoding="utf-8")
                        wand_cur_arr = np.array(wand_clone)
                        modified = False
                        with Color(pattern.search(magick_run.stdout)[0]) as trans_color:
                            for i, row in enumerate(wand_frame):
                                for j, col in enumerate(row):
                                    if col == trans_color:
                                        wand_cur_arr[i, j] = wand_prev_arr[i, j]
                                        modified = True
                        if modified:
                            with Image.from_array(wand_cur_arr) as wand_new:
                                wand_clone.import_pixels(channel_map="RGBA", data=wand_new.export_pixels())
                    gifsicle_commands.extend((f"-t{trans_idx}", f"{out}", f"#{cur_frame_no}-{clone_frame_no}"))
                else:
                    histogram = pil_frame.histogram()
                    trans_idx = min_early_break(range(len(histogram)), 0, histogram.__getitem__)
                    gifsicle_commands.extend(("--no-transparent", f"{out}", f"#{cur_frame_no}", f"-t{trans_idx}", f"{out}", f"#{clone_frame_no}"))
                queue.append(wand_frame)
                queue.append(wand_clone)
            last_transparent = next(pil_iter).info.get("transparency")
            last_command = f"-t{last_transparent}" if last_transparent is not None else "--no-transparent"
            gifsicle_commands.extend((last_command, f"{out}", "#-1", "-o", f"{out}"))
        wand_seq = wand_im.sequence
        queue.append(wand_seq[-1])
        for i in range(len(wand_seq)):
            wand_seq[i] = queue.popleft()
        wand_seq.extend(queue)
        wand_im.clone_frames()
        wand_im.save(filename=out)
    subprocess.run(gifsicle_commands)


# if __name__ == "__main__":
#     fp = "Original/Draw1.gif"
#     out = "out.gif"
#     preprocess(fp, out)
    # with GIFImage(filename=out) as im:
    #     seq = im.sequence
    #     for i in range(0, len(seq) - 1, 2):
    #         original = np.array(seq[i])
    #         clone = np.array(seq[i + 1])
    #         if not np.array_equal(original, clone):
    #             print(i)
    #             print("Original", original)
    #             print("Clone", clone)
