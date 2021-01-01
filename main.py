from gif_hider import GIFHider, random_bitstream
from numpy.random import default_rng


def main() -> None:
    file = "Original/Draw1.gif"
    out = "out.gif"
    stream = random_bitstream(default_rng(0))
    with GIFHider(file, out) as hider:
        hider.preprocess()
        hider.embed_data(stream, 3)
        hider.extract_payload(True)


if __name__ == "__main__":
    main()
