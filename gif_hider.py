import PIL.Image
import PIL.ImageSequence
import itertools
import numpy as np
import re
import subprocess

from collections import deque
from enum import Enum, auto
from gif_editor import GIFImage
from tqdm import tqdm, trange
from wand.color import Color
from wand.image import Image

from typing import Callable, Iterable, Iterator, TypeVar, Union

T = TypeVar("T")


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


def random_bitstream(rng: np.random.Generator = None) -> Iterator[str]:
    if rng is None:
        rng = np.random.default_rng()
    while True:
        yield str(rng.integers(2))


def rzl(data: Iterable[Union[int, str]], k: int) -> Iterator[str]:
    res = iter([])
    data = "".join(str(char) for char in data)
    for i in range(0, len(data), k):
        res = itertools.chain(res, ("0" for _ in range(int(data[i:i+k], base=2))), "1")
    return res


def extract_rzl(data: Iterable[str], k: int) -> Iterator[str]:
    data = "".join(data).split("1")
    data.pop()
    return itertools.chain(*[iter(format(len(i), f"0{k}b")) for i in data])


class GIFHiderState(Enum):
    RAW = auto()
    PREPROCESSED = auto()
    EMBEDDED = auto()


class GIFHider:
    transparent_pattern = re.compile(r"(?<=Transparent color: )\w+(\(.+\))?")

    def __init__(self, file: str, out: str, state: GIFHiderState = GIFHiderState.RAW, magick_path: str = "./magick") -> None:
        self.file = file
        self.out = out
        self.state = state
        self.magick_identify_cmd = [magick_path, "identify", "-verbose", None]
        self.magick_recent_out = None
        self.stored_data = None
        self.k = None
        self.set_image(file)

    def __enter__(self):
        return self
    
    def __exit__(self, *_) -> None:
        self.close_image()

    def set_image(self, file: str) -> None:
        try:
            self.close_image()
        except (AttributeError, TypeError):
            pass
        self.wand_im = GIFImage(filename=file)
        self.pil_im = PIL.Image.open(file)

    def close_image(self) -> None:
        self.wand_im.close()
        self.pil_im.close()
    
    def magick_identify(self, frame_no: int) -> None:
        file = self.file if self.state == GIFHiderState.RAW else self.out
        self.magick_identify_cmd[-1] = f"{file}[{frame_no}]"
        proc = subprocess.run(self.magick_identify_cmd, capture_output=True, encoding="utf-8")
        self.magick_recent_out = proc.stdout

    def wand_transparent_color(self, frame_no: int) -> str:
        self.magick_identify(frame_no)
        return self.transparent_pattern.search(self.magick_recent_out)[0]

    def wand_trans_arr(self, frame_no: int) -> np.ndarray:
        color_name = self.wand_transparent_color(frame_no)
        arr = re.search(f"(?<=: ).+(?={re.escape(color_name)})", self.magick_recent_out)
        arr = arr[0].split()[0][1:-1].split(",")
        try:
            arr = np.array([int(i) for i in arr])
        except TypeError:
            arr = np.zeros(4, int)
        return arr

    def preprocess(self) -> None:
        if self.state != GIFHiderState.RAW:
            raise Exception("GIF object alreday preprocessed")
        queue = deque()
        gifsicle_commands = ["gifsicle"]
        wand_im = self.wand_im
        pil_im = self.pil_im
        pil_im.seek(0)
        pil_iter = PIL.ImageSequence.Iterator(pil_im)
        pil_im_frames = (next(pil_iter) for _ in range(pil_im.n_frames - 1))
        out = self.out
        for frame_no, pil_frame in tqdm(enumerate(pil_im_frames), initial=1, total=pil_im.n_frames, desc="Preprocessing frames"):
            wand_frame, wand_clone = wand_im.clone_frame(frame_no)
            cur_frame_no = frame_no * 2
            clone_frame_no = cur_frame_no + 1
            trans_idx = pil_frame.info.get("transparency")
            if trans_idx is not None:
                trans_color = self.wand_transparent_color(frame_no)
                wand_cur_arr = np.array(wand_clone)
                background = wand_frame.background_color
                need_modification = True
                modified = False
                with Color(trans_color) as trans_color:
                    try:
                        wand_prev_arr = np.array(queue[-1])
                    except IndexError:
                        need_modification = background != trans_color
                    if need_modification:
                        for i, row in enumerate(wand_frame):
                            for j, col in enumerate(row):
                                if col == trans_color:
                                    try:
                                        wand_cur_arr[i, j] = wand_prev_arr[i, j]
                                    except NameError:
                                        wand_cur_arr[i, j] = background
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
        wand_im.save(filename=out)
        subprocess.run(gifsicle_commands)
        self.state = GIFHiderState.PREPROCESSED
        self.set_image(out)

    def embed_data(self, data: Iterable[str], k: int = None) -> None:
        if self.state != GIFHiderState.PREPROCESSED:
            raise Exception("GIF object not preprocessed, run the preprocess() method first")
        data = iter(data)
        cur_data = deque()
        msg = []
        wand_seq = self.wand_im.sequence
        pil_im = self.pil_im
        for frame_no in trange(1, pil_im.n_frames, 2, desc="Embedding data"):
            trans_arr = self.wand_trans_arr(frame_no)
            prev_frame_no = frame_no - 1
            pil_im.seek(prev_frame_no)
            prev_has_trans = "transparency" not in pil_im.info
            with wand_seq[prev_frame_no].clone() as prev_frame:
                prev_arr = np.array(prev_frame)
            modified = False
            with wand_seq[frame_no].clone() as cur_frame:
                cur_arr = np.array(cur_frame)
                for i, row in enumerate(cur_arr):
                    for j, _ in enumerate(row):
                        if prev_has_trans or not np.array_equal(prev_arr[i, j], trans_arr):
                            if k is not None:
                                if not cur_data:
                                    newdata = [next(data) for _ in range(k)]
                                    cur_data.extend(rzl(newdata, k))
                                    msg.extend(newdata)
                                next_bit = cur_data.popleft()
                            else:
                                next_bit = next(data)
                                msg.append(next_bit)
                            if next_bit == "0":
                                cur_arr[i, j] = trans_arr
                                modified = True
                if modified:
                    with Image.from_array(cur_arr) as new_frame:
                        cur_frame.import_pixels(channel_map="RGBA", data=new_frame.export_pixels())
                    wand_seq[frame_no] = cur_frame
        self.wand_im.save(filename=self.out)
        subprocess.run(["gifsicle", self.out, "-o", self.out])
        msg = "".join(msg)
        self.stored_data = msg[:-k] if cur_data else msg
        self.k = k
        self.state = GIFHiderState.EMBEDDED
        self.set_image(self.out)

    def extract_payload(self, validate: bool = False) -> str:
        if self.state != GIFHiderState.EMBEDDED:
            raise Exception("Nothing embedded in GIF image")
        payload = []
        wand_seq = self.wand_im.sequence
        pil_im = self.pil_im
        for frame_no in trange(1, pil_im.n_frames, 2, desc="Extracting payload"):
            trans_arr = self.wand_trans_arr(frame_no)
            prev_frame_no = frame_no - 1
            pil_im.seek(prev_frame_no)
            prev_trans = pil_im.info.get("transparency")
            pil_im.seek(frame_no)
            cur_trans = pil_im.info["transparency"]
            with wand_seq[prev_frame_no].clone() as prev_frame:
                prev_arr = np.array(prev_frame)
            with wand_seq[frame_no].clone() as cur_frame:
                cur_arr = np.array(cur_frame)
            for i, row in enumerate(cur_arr):
                for j, col in enumerate(row):
                    if prev_trans is None or prev_trans == cur_trans or not np.array_equal(prev_arr[i, j], trans_arr):
                        payload.append("0" if np.array_equal(col, trans_arr) else "1")
        if self.k is not None:
            while payload and payload[-1] == "0":
                payload.pop()
            payload = extract_rzl(payload, self.k)
        payload = "".join(payload)
        if validate:
            if payload == self.stored_data:
                print("Data extracted succesfully")
            else:
                raise Exception("Payload not equal to embedded data")
        return payload
