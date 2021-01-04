# Monash IT Student Research (iSR) Scheme 2020

## Topic: Data Hiding (Animated GIF)

By: Yap Jin Heng

Mentor: A/Prof. Dr. Wong Kok Sheik

## Dependencies
1. [Gifsicle](https://github.com/kohler/gifsicle)
2. [ImageMagick](https://imagemagick.org/script/download.php)
3. Python and the following libraries:
    - [NumPy](https://numpy.org/)
    - [Pillow](https://pillow.readthedocs.io/)
    - [tqdm](https://tqdm.github.io/)

## HOWTO:

### Hide and extract data in a GIF file
1. Download and install the dependencies above.
2. From the terminal, run the following command:
    ```
    python main.py {INPUT_GIF} {OUTPUT_GIF} [-v]
    ```
    - The `-v` flag is optional, where it performs additional validation after extracting data.
3. If any modifications need to be done, perform them in `main.py`.

### Analyse GIF files
1. Download and install the dependencies above.
2. Edit and run the functions in `gif_analyser.py` as necessary.
