#!/usr/bin/env python
"""Train forest method."""

import argparse
import cPickle as pkl
import logging
import os

from pyforest.forest import RandomForest
from pyforest.utils import write_to_file


__author__ = "Andrea Casini"
__copyright__ = "Copyright 2014"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "acasini@dsi.unive.it"


def main():
    parser = argparse.ArgumentParser(description='Random Forest Training')

    # Positional.
    parser.add_argument('trainingset',
                        metavar='trainingset',
                        help='training dataset (samples, labels)',
                        type=argparse.FileType('rt'))

    # Optional.
    parser.add_argument('-t',
                        dest='n_trees',
                        help='set the number of trees',
                        type=int,
                        default=20)

    parser.add_argument('-d',
                        dest='max_depth',
                        help='set maximum depth of each tree',
                        type=int,
                        default=-1)

    parser.add_argument('-s',
                        dest='min_samples_split',
                        help='set the minimum number of samples required to split an internal node',
                        type=int,
                        default=5)

    parser.add_argument('-r',
                        action='store',
                        dest='n_rounds',
                        help='set number of rounds at each split',
                        type=int,
                        default=200)

    parser.add_argument('-n',
                        dest='n_jobs',
                        help='number of processes to run in parallel',
                        type=int,
                        default=-1)

    parser.add_argument('--quiet',
                        action='store_true',
                        dest='quiet',
                        help='do not print additional output',
                        default=False)

    try:
        args = parser.parse_args()

        rf = RandomForest(args.n_trees,
                          args.max_depth,
                          args.n_rounds,
                          args.min_samples_split,
                          args.n_jobs)

        # Set verbosity level.
        if not args.quiet:
            logging.basicConfig(level=logging.DEBUG,
                                format='[%(levelname)s %(asctime)s] %(funcName)s: %(message)s',
                                datefmt='%H:%M:%S')

        logging.info('Loading dataset.')
        train_set = pkl.load(args.trainingset)

        logging.info('Training forest.')
        rf.fit(train_set.samples, train_set.labels)

        filename = 'forests/rf_t{}_d{}_r{}_s{}.pkl'.format(
            args.n_trees,
            args.max_depth,
            args.n_rounds,
            args.min_samples_split)

        write_to_file(rf, filename)
        print('\nForest saved in: ' + filename + '\n')

    except IOError, msg:
        parser.error(str(msg))


if __name__ == '__main__':
    main()
