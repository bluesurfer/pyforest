#!/usr/bin/env python
"""Faces detection method."""

import argparse
import pylab as pl
import Image
import logging
import cPickle as pkl

from scipy.misc import imresize
from pyforest import utils


__author__ = "Andrea Casini"
__copyright__ = "Copyright 2014"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "acasini@dsi.unive.it"


def main():

    parser = argparse.ArgumentParser(description='Face detection using a random forest.')

    # Positional.
    parser.add_argument('randomforest',
                        metavar='randomforest',
                        help='a trained random forest',
                        type=argparse.FileType('rt'))

    parser.add_argument('image',
                        metavar='image',
                        help='a PNG image',
                        type=argparse.FileType('rt'))

    # Optional.
    parser.add_argument('-r',
                        dest='resize',
                        help='image resizing factor',
                        type=float,
                        default=1.)

    parser.add_argument('-t',
                        dest='thresh',
                        help='threshold over probability of being a face',
                        type=float,
                        default=.6)

    parser.add_argument('--quiet',
                        action='store_true',
                        dest='quiet',
                        help='do not print additional output',
                        default=False)

    try:

        args = parser.parse_args()

        # Set verbosity level.
        if not args.quiet:
            logging.basicConfig(level=logging.DEBUG,
                                format='[%(levelname)s %(asctime)s] %(funcName)s: %(message)s',
                                datefmt='%H:%M:%S')

        # Load a given random forest.
        random_forest = pkl.load(args.randomforest)

        # Read image into memory and convert it to gray scale.
        im_gray = Image.open(args.image).convert('L')

        # Resize image (cast to array).
        im_gray = imresize(im_gray, size=args.resize)

        logging.info('Extracting patches from image.')

        # Break up the image in 19x19 patches.
        patch_size = (19, 19)
        x_test = utils.extract_patches(im_gray, patch_size)

        logging.info('Detecting faces.')

        # Detect faces (test the forest).
        y_pred, probs = random_forest.predict(x_test)

        # Create useful masks to plot detection results.
        heat_map, max_map = utils.create_heat_map(im_gray,
                                                  patch_size,
                                                  probs,
                                                  args.thresh)

        logging.info('Plotting results.')

        # Plot these masks.
        pl.figure()
        pl.imshow(im_gray, cmap='gray')
        pl.imshow(heat_map, interpolation='nearest', cmap='jet', alpha=0.5)
        pl.title('Heat Map')
        pl.axis('off')

        pl.figure()
        pl.imshow(im_gray, cmap='bone')
        pl.imshow(max_map, interpolation='nearest', cmap='jet_r', alpha=0.5)
        pl.title('Faces Detected (threshold >= %.2f)' % args.thresh)
        pl.axis('off')

        pl.show()

    except IOError, msg:
        parser.error(str(msg))


if __name__ == '__main__':
    main()