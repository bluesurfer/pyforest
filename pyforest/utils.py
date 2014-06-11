#!/usr/bin/env python
"""Support library with useful utilities."""

import cPickle as pkl
import numpy as np
import os
import Image

from scipy.ndimage.filters import maximum_filter
from numpy.lib.stride_tricks import as_strided


__author__ = "Andrea Casini"
__copyright__ = "Copyright 2014"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "acasini@dsi.unive.it"


__all__ = ['Dataset',
           'write_to_file',
           'read_from_file',
           'extract_patches',
           'create_heat_map']


class Dataset():
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels


def write_to_file(obj, filename):
    with open(filename, 'wt') as f:
        pkl.dump(obj, f)


def read_from_file(filename):
    with open(filename, 'rt') as f:
        return pkl.load(f)


def extract_patches(im, patch_size):
    """ Reshape a 2D image into a collection of patches

    The resulting patches are allocated in a dedicated array.

    Parameters
    ----------
    image: array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size: tuple of ints (patch_height, patch_width)
        the dimensions of one patch

    """
    shape = np.array(im.shape * 2)
    strides = np.array(im.strides * 2)
    patch_size = np.asarray(patch_size)
    shape[im.ndim:] = patch_size  # new dimensions size
    shape[:im.ndim] -= patch_size - 1

    if np.any(shape < 1):
        raise ValueError('window size is too large')
    chunks = as_strided(im, shape=shape, strides=strides).flatten()
    dot = patch_size[0] * patch_size[1]

    return chunks.reshape((chunks.shape[0] / dot), dot)


def create_heat_map(im, patch_size, probs, thresh):
    """ Create a heat map.

    Hotter spots indicates the presence of a face.

    """
    rows, cols = patch_size
    bound = im.shape[1] - cols + 1

    heat_map = np.zeros(im.shape)

    indices = np.arange(probs.shape[1])

    r = indices / bound
    c = indices % bound

    heat_map[r + rows / 2, c + cols / 2] = probs[1, :]

    # Apply a maximum filter.
    max_map = maximum_filter(heat_map, patch_size)
    max_map = np.ma.greater_equal(max_map, thresh)

    # Make transparent not detected.
    max_map = np.ma.masked_where(~max_map, max_map)

    return heat_map, max_map
