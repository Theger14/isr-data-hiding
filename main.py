import argparse

from gif_hider import GIFHider, random_bitstream
from numpy.random import default_rng


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform data hiding in a GIF file"
    )
    parser.add_argument("file", help="input GIF file")
    parser.add_argument("out", help="output GIF file")
    parser.add_argument(
        "-v",
        "--validate",
        help="validate data extracted is correct",
        action="store_true"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream = random_bitstream(default_rng(0))
    with GIFHider(args.file, args.out) as hider:
        hider.preprocess()
        hider.embed_data(stream, 3)
        hider.extract_payload(args.validate)


if __name__ == "__main__":
    main()
