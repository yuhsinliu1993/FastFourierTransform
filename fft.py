#   Python 2.7
#
#   Pre-requisite packages:
#       Opencv 3
#       Numpy
#       Scipy
#
#   How to execute:
#       python hw7.py [-h] [--dir DIR] [--filename FILENAME]
#       EX:
#           python hw7.py --filename Q1.tif
#
#    optional arguments:
#      -h, --help  show this help message and exit
#      --filename FILENAME  Specify the input image's name
#      --dir DIR   Specify location of the input image. Default value will be current directory  [Default: current dir]
#

import argparse, os
import cv2
import numpy as np
import cmath, sys
from scipy import ndimage, misc


# Computes the discrete Fourier transform (DFT) or inverse transform of the given complex vector, returning the result as a new vector.
# The vector can have any length. This is a wrapper function. The inverse transform does not perform scaling, so it is not a true inverse.
def transform(x, inverse=False):
    N = x.shape[0]

    if N == 0:
        return []
    elif N & (N - 1) == 0:
        # check if it is power of 2
        return transform_radix2(x, inverse)
    else:
        # More complicated algorithm for arbitrary sizes
        return transform_bluestein(x, inverse)


# Returns the integer whose value is the reverse of the lowest 'bits' bits of the integer 'x'.
def reverse(_x, bits):
    y = 0
    for i in xrange(bits):
        y = (y << 1) | (_x & 1)
        _x >>= 1
    return y


def transform_radix2(x, inverse=False):
    N = x.shape[0]
    levels = N.bit_length()-1
    if 2 ** levels != N:
        raise ValueError("Length is not a power of 2")

    # Now, levels = log2(n)
    coef = (2j if inverse else -2j) * np.pi / N
    W_exp = np.exp(np.arange(N//2) * coef)
    # Copy with bit-reversed permutation
    x = [x[reverse(i, levels)] for i in range(N)]

    # Radix-2 decimation-in-time FFT
    size = 2
    while size<=N:
        halfsize = size // 2
        tablestep = N // size

        for i in range(0, N, size):
            k = 0
            for j in range(i, i + halfsize):
                temp = x[j + halfsize] * W_exp[k]
                x[j + halfsize] = x[j] - temp
                x[j] += temp
                k += tablestep

        size *= 2

    return np.asarray(x)


# Computes the discrete Fourier transform (DFT) of the given complex vector, returning the result as a new vector.
# The vector can have any length. This requires the convolution function, which in turn requires the radix-2 FFT function.
# Uses Bluestein's chirp z-transform algorithm.
def transform_bluestein(x, inverse=False):
    # Find a power-of-2 convolution length m such that m >= n * 2 + 1
    N = x.shape[0]
    M = 2**((N*2).bit_length())

    coef = (1j if inverse else -1j) * np.pi / N
    W_exp = np.exp((np.arange(N)**2 % (N * 2)) * coef)

    zero_paddings_a = np.asarray([0]*(M-N))
    a_n = np.concatenate((x*W_exp, zero_paddings_a))

    zero_paddings_b = np.asarray([0] * (M - (N*2-1)))
    b_n = np.concatenate((W_exp[:N], zero_paddings_b, W_exp[:0:-1])).conjugate()

    c = convolve(a_n, b_n, False)[:N]  # Convolution

    return c * W_exp
    # return [(x * y) for (x, y) in zip(c, exptable)]  # Postprocessing

# Computes the circular convolution of the given real or complex vectors, returning the result as a new vector. Each vector's length must be the same.
# realoutput=True: Extract the real part of the convolution, so that the output is a list of floats. This is useful if both inputs are real.
# realoutput=False: The output is always a list of complex numbers (even if both inputs are real).
def convolve(a, b, real=True):
    assert a.shape == b.shape

    N = a.shape[0]

    a = transform(a)
    b = transform(b)

    for i in range(N):
        a[i] = a[i] * b[i]

    # IDFT
    a = transform(a, True)

    # Scaling (because this FFT implementation omits it) and postprocessing
    if real:
        return np.asarray([(val.real/N) for val in a])
    else:
        return np.asarray([(val/N) for val in a])


def FFT2D(image):
    """ implementation of 2-D Fast Fourier Transform """
    M, N = image.shape
    FFT_result = np.zeros_like(image, dtype=complex)

    for i in xrange(M):
        FFT_result[i,:] = np.asarray(transform_bluestein(image[i,:]))

    for j in xrange(N):
        FFT_result[:, j] = np.asarray(transform_bluestein(FFT_result[:, j]))

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


def DFT1D_slow(signal):
    """ 1-D Discrete Fourier Transform """
    x = np.asarray(signal, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)

    return np.dot(M, x)


def DFT2D_slow(image, name=None):
    global M, N
    M, N = image.shape

    dft_result = np.zeros_like(image, dtype=float)
    dft_img = np.zeros_like(image, dtype=float)

    for k in xrange(M):
        for l in xrange(N):
            _sum = 0.0

            for m in xrange(M):
                for n in xrange(N):
                    e_part = np.exp(2 * np.pi * (-1j) * (float(k*m)/M + float(l*n)/N))
                    _sum += image[m][n] * e_part

            print "(",k,", ",l,") => SUM: ",_sum
            dft_result[k][l] = _sum
            dft_img[k][l] = int(_sum.real)

    if name:
        cv2.imwrite(name+"_dft.jpg", dft_img)
    else:
        cv2.imwrite("dft.jpg", dft_img)

    return dft_result



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
    unshifted_fft = FFT2D(img)
    spectrum = np.log10(np.absolute(unshifted_fft) + np.ones_like(img))
    misc.imsave("images/%s_unshifted_fft.png" % filename.split('.')[0], spectrum)

    shifted_fft = FFT2D_shift(unshifted_fft)
    spectrum = np.log10(np.absolute(shifted_fft) + np.ones_like(img))
    misc.imsave("images/%s_shifted_fft.png" % filename.split('.')[0], spectrum)
