"""
GIF data hiding utility, also containing a random bitstream generation function.
"""

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
    """The identity function."""
    return x


def min_early_break(
    iterable: Iterable[T],
    break_when: float,
    key: Callable = identity
) -> T:
    """
    Equivalent of min() function that breaks early when a condition is met.

    :param iterable: iterable object
    :param break_when: a numeric value to meet
    :param key: the key function to act upon each object in the iterable
    :return: the minimum value of the iterable
    """
    iterable_iter = iter(iterable)
    min_elem = next(iterable_iter)
    for elem in iterable_iter:
        if key(min_elem) <= break_when:
            break
        if key(elem) <= key(min_elem):
            min_elem = elem
    return min_elem


def random_bitstream(rng: np.random.Generator = None) -> Iterator[str]:
    """
    Infinite random bitstream that outputss the bits as strings.

    :param rng: random number generator
    :return: an infinite iterator of random bits as strings
    """
    if rng is None:
        rng = np.random.default_rng()
    while True:
        yield str(rng.integers(2))


def rzl(data: Iterable[Union[int, str]], k: int) -> Iterator[str]:
    """
    Encode a bitstream with reverse zerorun length (RZL) encoding.

    :param data: an iterable sequence of bits
    :param k: the k value to encode the string
    :return: an iterator that returns the RZL-encoded bitstream bit by bit
    """
    res = iter([])
    data = "".join(str(char) for char in data)
    for i in range(0, len(data), k):
        zeros = ("0" for _ in range(int(data[i:i+k], base=2)))
        res = itertools.chain(res, zeros, "1")
    return res


def extract_rzl(data: Iterable[str], k: int) -> Iterator[str]:
    """
    Decode a bitstream that has been encoded with reverse zerorun length (RZL)
    encoding.

    :param data: a RZL-encoded iterable sequence of bits
    :param k: the k value used to encode the string
    :return: an iterator that returns the decoded bitstream bit by bit
    """
    data = "".join(data).split("1")
    data.pop()
    return itertools.chain(*[iter(format(len(i), f"0{k}b")) for i in data])


class GIFHiderState(Enum):
    """
    Enum class to represent the state of a GIFHider object, which signifies the
    state of operations we are currently at.

    Details are as below:

    RAW: nothing has been encoded in the GIF file
    PREPROCESSED: the GIF file has been preprocessed to have (2n - 1) frames and
        every added frame has the transparency pixel
    EMBEDDED: data has been embedded in the output GIF file
    """

    RAW = auto()
    PREPROCESSED = auto()
    EMBEDDED = auto()


class GIFHider:
    """
    Main class for hiding data in processed GIF files.

    Class Attributes:
        transparent_pattern: regular expression pattern to match for finding the
            transparent colour in a GIF frame
    
    Instance Attributes:
        file: file path to input GIF file
        out: output file path of processed GIF file
        state: state of a GIFHider instance, which GIFHiderState object
        magick_identify_cmd: list containing command to be issued when
            ImageMagick commands are called
        magick_recent_out: most recent output of Magick command
        stored_data: data stored in GIF object. Used for validation when
            extracting data
        k: k value for RZL encoding of data
        wand_im: GIFImage object representing the loaded GIF file
        pil_im: PIL object representing the loading GIF file
    """

    transparent_pattern = re.compile(r"(?<=Transparent color: )\w+(\(.+\))?")

    def __init__(
        self,
        file: str,
        out: str,
        state: GIFHiderState = GIFHiderState.RAW,
        magick_path: str = "./magick"
    ) -> None:
        """
        Initialises a GIFHider instance.

        :param magick_path: path to ImageMagick
        """
        self.file = file
        self.out = out
        self.state = state
        self.magick_identify_cmd = [magick_path, "identify", "-verbose", None]
        self.magick_recent_out = None
        self.stored_data = None
        self.k = None
        self.set_image(file)

    def __enter__(self):
        """Allow this object to be opened with a Python `with` statement."""
        return self
    
    def __exit__(self, *_) -> None:
        """Close the image."""
        self.close_image()

    def set_image(self, file: str) -> None:
        """
        Set the current referenced image.

        :param file: file path to current GIF file
        """
        try:
            self.close_image()
        except (AttributeError, TypeError):
            pass
        self.wand_im = GIFImage(filename=file)
        self.pil_im = PIL.Image.open(file)

    def close_image(self) -> None:
        """Close the image objects for both wand.image and PIL.Image."""
        self.wand_im.close()
        self.pil_im.close()
    
    def magick_identify(self, frame_no: int) -> None:
        """
        Perform a verbose ImageMagick `identify` operation on a GIF frame.

        :param frame_no: frame number
        """
        file = self.file if self.state == GIFHiderState.RAW else self.out
        self.magick_identify_cmd[-1] = f"{file}[{frame_no}]"
        proc = subprocess.run(
            self.magick_identify_cmd, capture_output=True, encoding="utf-8")
        self.magick_recent_out = proc.stdout

    def wand_transparent_color(self, frame_no: int) -> str:
        """
        Identify the name of the transparent colour of a frame.

        :param frame_no: frame number
        :return: name of transparent colour of a frame
        """
        self.magick_identify(frame_no)
        return self.transparent_pattern.search(self.magick_recent_out)[0]

    def wand_trans_arr(self, frame_no: int) -> np.ndarray:
        """
        Identify the transparent colour of a frame as a NumPy array.

        :param frame_no: frame number
        :return: array representing transparent colour of a frame
        """
        color_name = self.wand_transparent_color(frame_no)
        pattern = f"(?<=: ).+(?={re.escape(color_name)})"
        arr = re.search(pattern, self.magick_recent_out)
        arr = arr[0].split()[0][1:-1].split(",")
        try:
            arr = np.array([int(i) for i in arr])
        except TypeError:
            arr = np.zeros(4, int)
        return arr

    def preprocess(self) -> None:
        """
        Perform pre-processing of the GIF file for data hiding. This involves:
        1. Splitting the frames into (2n - 1) frames.
        2. Adjusting the delay times of each frame.
        3. Adjust every added (odd) frame to have a transparent pixel.
        """
        if self.state != GIFHiderState.RAW:
            raise Exception("GIF object alreday preprocessed")

        # Setting up
        queue = deque()
        gifsicle_commands = ["gifsicle"]
        wand_im = self.wand_im
        pil_im = self.pil_im
        pil_im.seek(0)
        pil_iter = PIL.ImageSequence.Iterator(pil_im)
        pil_im_frames = (next(pil_iter) for _ in range(pil_im.n_frames - 1))
        out = self.out

        # Iterate over each frame
        progress_bar = tqdm(
            enumerate(pil_im_frames),
            initial=1,
            total=pil_im.n_frames,
            desc="Preprocessing frames"
        )
        for frame_no, pil_frame in progress_bar:
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
                # Remove all transparent pixels in newly inserted frame
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
                # We can avoid modifying the frame if we have performed no
                # modification above
                if modified:
                    with Image.from_array(wand_cur_arr) as wand_new:
                        wand_clone.import_pixels(channel_map="RGBA",
                            data=wand_new.export_pixels())
                # Preserve transparency index of the GIF frame
                gifsicle_commands.extend((
                    f"-t{trans_idx}",
                    f"{out}",
                    f"#{cur_frame_no}-{clone_frame_no}"
                ))
            else:
                # Get least frequent color in histogram
                histogram = pil_frame.histogram()
                colors = range(len(histogram))
                trans_idx = min_early_break(colors, 0, histogram.__getitem__)
                gifsicle_commands.extend((
                    "--no-transparent",
                    f"{out}",
                    f"#{cur_frame_no}",
                    f"-t{trans_idx}",
                    f"{out}",
                    f"#{clone_frame_no}"
                ))
            queue.append(wand_frame)
            queue.append(wand_clone)
        last_trans = next(pil_iter).info.get("transparency")
        last_cmd = f"-t{last_trans}" if last_trans is not None else "--no-transparent"
        gifsicle_commands.extend((last_cmd, f"{out}", "#-1", "-o", f"{out}"))
        wand_seq = wand_im.sequence
        queue.append(wand_seq[-1])
        
        # Write changes to file
        for i in range(len(wand_seq)):
            wand_seq[i] = queue.popleft()
        wand_seq.extend(queue)
        wand_im.save(filename=out)
        subprocess.run(gifsicle_commands)
        self.state = GIFHiderState.PREPROCESSED
        self.set_image(out)

    def embed_data(self, data: Iterable[str], k: int = None) -> None:
        """
        Embed data in a pre-processed GIF file. This is where data hiding takes
        place. This method also optimizes the output GIF file to make data
        hiding less obvious.

        :param data: data to embed in GIF file
        :param k: the k value to encode the data with RZL (if any)
        """
        if self.state != GIFHiderState.PREPROCESSED:
            raise Exception("GIF object not preprocessed, run the preprocess() method first")
        # Setting up
        data = iter(data)
        cur_data = deque()
        msg = []
        wand_seq = self.wand_im.sequence
        pil_im = self.pil_im

        # Iterate over each odd frame
        progress_bar = trange(1, pil_im.n_frames, 2, desc="Embedding data")
        for frame_no in progress_bar:
            trans_arr = self.wand_trans_arr(frame_no)
            prev_frame_no = frame_no - 1
            pil_im.seek(prev_frame_no)
            prev_has_trans = "transparency" not in pil_im.info
            with wand_seq[prev_frame_no].clone() as prev_frame:
                prev_arr = np.array(prev_frame)
            modified = False

            # Embed data into each frame, encoding data with RZL if k is given
            with wand_seq[frame_no].clone() as cur_frame:
                cur_arr = np.array(cur_frame)
                for i, row in enumerate(cur_arr):
                    for j, _ in enumerate(row):
                        # Check if pixel is usable for data hiding:
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
                        cur_frame.import_pixels(channel_map="RGBA",
                            data=new_frame.export_pixels())
                    wand_seq[frame_no] = cur_frame
        
        # Write changes to file
        self.wand_im.save(filename=self.out)

        # This performs optimization, i.e. removes all unnecessary transparency
        # pixel data and local colour tables
        subprocess.run(["gifsicle", self.out, "-o", self.out])
        msg = "".join(msg)
        self.stored_data = msg[:-k] if cur_data else msg
        self.k = k
        self.state = GIFHiderState.EMBEDDED
        self.set_image(self.out)

    def extract_payload(self, validate: bool = False) -> str:
        """
        Extract data from a GIF file where data is embedded.

        :param validate: optional parameter to compare extracted data with
            stored data when embedding to validate our data hiding
        :return: data hidden in GIF file
        """
        if self.state != GIFHiderState.EMBEDDED:
            raise Exception("Nothing embedded in GIF image")

        # Setting up
        payload = []
        wand_seq = self.wand_im.sequence
        pil_im = self.pil_im

        # Iterate over each odd frame
        progress_bar = trange(1, pil_im.n_frames, 2, desc="Extracting payload")
        for frame_no in progress_bar:
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
                    # Check if pixel was usable for data hiding:
                    if prev_trans is None or prev_trans == cur_trans or not np.array_equal(prev_arr[i, j], trans_arr):
                        payload.append(
                            "0" if np.array_equal(col, trans_arr) else "1"
                        )
        # RZL decoding
        if self.k is not None:
            while payload and payload[-1] == "0":
                payload.pop()
            payload = extract_rzl(payload, self.k)
        payload = "".join(payload)
        # Validating extracted data
        if validate:
            if payload == self.stored_data:
                print("Data extracted succesfully")
            else:
                raise Exception("Payload not equal to embedded data")
        return payload
