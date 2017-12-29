#   Python 2.7
#
#   Pre-requisite packages:
#       Opencv 3
#       Numpy
#       Scipy
#
#   How to execute:
#       1. Run a single image
#           python fft.py [-h] [--dir DIR] [--filename FILENAME]
#           Ex:
#               python fft.py --filename Q1.tif
#
#       2. You can put Q1.tif~Q4.tif into "images" folder and execute
#           ./run.sh
#
#
#    optional arguments:
#      -h, --help  show this help message and exit
#      --filename FILENAME  Specify the input image's name
#      --dir DIR   Specify location of the input image. Default value will be current directory  [Default: current dir]
#

import argparse, os
import cv2
import numpy as np
from scipy import ndimage, misc


def zero_padding(image):
    """
    Zero padding the image two the nearest power-of-2 rows and cols
    """
    assert len(image.shape) == 2

    M, N = image.shape

    new_row = M
    new_col = N

    while new_row&(new_row-1):
        new_row+=1
    while new_col&(new_col-1):
        new_col+=1

    padded_image = np.zeros((new_row, new_col))
    padded_image[:M,:N] = image[:,:]

    return padded_image


def cooley_tukey_dit_fft(x, inverse=False):
    """
    1. This transform is based on Cooley-Tukey decimation-in-time radix-2 algorithm.
    2. Require the input x's length to be the power-of-2. In order to compute the FFT of the input x.
    """
    # Returns the integer whose value is the reverse of the lowest 'bits' bits of the integer 'x'.
    def bit_reversal_permutation(_x, bits):
        y = 0
        for i in xrange(bits):
            y = (y << 1) | (_x & 1)
            _x >>= 1
        return y

    N = x.shape[0]
    levels = N.bit_length()-1   # levels = log2(n)

    if 2 ** levels != N:
        raise ValueError("Length is not a power of 2")

    coef = (2j if inverse else -2j) * np.pi / N
    W_exp = np.exp(np.arange(N//2) * coef)

    # Copy with bit-reversed permutation
    x = [x[bit_reversal_permutation(i, levels)] for i in range(N)]

    # Radix-2 decimation-in-time FFT
    size = 2
    while size <= N:
        half_size = size // 2
        table_step = N // size

        for i in xrange(0, N, size):
            k = 0
            for j in range(i, i + half_size):
                temp = x[j + half_size] * W_exp[k]
                x[j + half_size] = x[j] - temp
                x[j] += temp
                k += table_step

        size *= 2

    return np.asarray(x)


def FFT2D(image):
    """ implementation of 2-D Fast Fourier Transform """
    M, N = image.shape

    FFT_result = np.zeros_like(image, dtype=complex)

    for i in xrange(M):
        FFT_result[i,:] = cooley_tukey_dit_fft(image[i,:])

    for j in xrange(N):
        FFT_result[:, j] = cooley_tukey_dit_fft(FFT_result[:, j])

    return FFT_result


def FFT2D_shift(fft):
    """ Shift the zero frequency to the center of the 2-D Fourier Transform """
    rows, cols = fft.shape
    tmp = np.zeros_like(fft)
    ret = np.zeros_like(fft)

    for i in xrange(rows):
        for j in xrange(cols):
            index = (cols/2 + j) % cols
            tmp[i, index] = fft[i, j]

    for j in xrange(cols):
        for i in xrange(rows):
            index = (rows/2 + i) % rows
            ret[index, j] = tmp[i, j]

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        type=str,
        help='Specify location of the input image. Default value will be current directory',
    )
    parser.add_argument(
        '--filename',
        type=str,
        help='Specify the input image\'s name',
    )

    args = parser.parse_args()
    filename = "noname.png" if not args.filename else args.filename

    if args.dir is None:
        cwd = os.getcwd()
        image_path = os.path.join(cwd, "images", filename)
    else:
        image_path = os.path.join(args.dir, filename)

    img = ndimage.imread(image_path, flatten=True)
    padded_img = zero_padding(img)

    # Compute the unshifted FFT of the padded image
    unshifted_fft = FFT2D(padded_img)
    spectrum = np.log10(np.absolute(unshifted_fft) + np.ones_like(padded_img))
    misc.imsave("images/%s_unshifted_fft.png" % filename.split('.')[0], spectrum)

    # Compute the shifted FFT of the padded image
    shifted_fft = FFT2D_shift(unshifted_fft)
    spectrum = np.log10(np.absolute(shifted_fft) + np.ones_like(padded_img))
    misc.imsave("images/%s_shifted_fft.png" % filename.split('.')[0], spectrum)
