"""
Main module for the project. Run this file with commands like this:

python main.py {INPUT_GIF} {OUTPUT_GIF} [-v]

{INPUT_GIF} is the source GIF file, {OUTPUT_GIF} is the GIF file to output the
GIF file with embedded data, and [-v] or [--validate] is used to validate the
data {OUTPUT_FILE} after processing.
"""

import argparse

from gif_hider import GIFHider, random_bitstream
from numpy.random import default_rng


def parse_args() -> argparse.Namespace:
    """
    Argument parser that accept arguments from the terminal.
    
    :return: argument namespace. Use this as the arguments to run the program.
    """
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
    """Driver code. Run everything here."""
    args = parse_args()
    # Set the random seed here
    stream = random_bitstream(default_rng(0))
    with GIFHider(args.file, args.out) as hider:
        # Comment out any of these processes to omit them
        hider.preprocess()
        hider.embed_data(stream, 3)
        hider.extract_payload(args.validate)


if __name__ == "__main__":
    main()
