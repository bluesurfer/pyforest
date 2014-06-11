#!/usr/bin/env python
"""Implementation of internal node and leaf of the decision tree."""

from __future__ import division

from math import log

import numpy as np


__author__ = "Andrea Casini"
__copyright__ = "Copyright 2014"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "acasini@dsi.unive.it"


def entropy(y):
    """ Computes entropy of a label distribution. """
    counts = np.bincount(y)
    n_classes = np.count_nonzero(counts)

    if n_classes <= 1:
        return 0.

    result = 0.
    probs = counts / len(y)

    # Entropy standard formula.
    for p in probs:
        if p != 0.:
            result -= p * log(p, n_classes)

    return result


def information_gain(y_l, y_r):
    """ Computes expected information gain about the label
        distribution due to a binary split. (Note: LOWER IS BETTER).

        Reference
        ---------
        http://www.icg.tugraz.at/Members/kontschieder/iccv11.pdf

    """
    n, m = len(y_l), len(y_r)
    n_labels = len(y_l) + len(y_r)

    inf_l = 0. if n <= 1 else (n / n_labels) * entropy(y_l)
    inf_r = 0. if m <= 1 else (m / n_labels) * entropy(y_r)

    return inf_l + inf_r


#______________________________________________________________________________


def threshold1(x, col, thresh):
    """ 1-dimensional classification using a threshold function.

    """
    return x[:, col] <= thresh


def threshold2(x, col1, col2, thresh):
    """ 2-dimensional classification using a threshold function.
        Consider the DIFFERENCE between the two columns.
    """
    return (x[:, col1] - x[:, col2]) <= thresh


def threshold3(x, col1, col2, thresh):
    """ 2-dimensional classification using a threshold function.
        Consider the SUM between the two columns.

    """
    return (x[:, col1] + x[:, col2]) <= thresh


def threshold4(x, col1, col2, thresh):
    """ 2-dimensional classification using a threshold function.
        Consider the ABSOLUTE VALUE of the difference.

    """
    return (np.abs(x[:, col1] - x[:, col2])) <= thresh


#______________________________________________________________________________


class Node():
    """ Binary tree intern node class.

    """

    def __init__(self, func=None, args=None):

        # Parameters to be learned.
        self.func = func
        self.args = args

        # Node's children.
        self.left = None
        self.right = None

        # A leaf is a node without children.
        self.is_a_leaf = False

    def train(self, x_train, y_train, n_rounds, random_state):
        """ Random classify the training data.

            Returns
            -------
            self : Node() with trained func and args.

        """
        min_inf = 1.
        n_features = x_train.shape[1]
        test_funcs = [threshold1, threshold2, threshold3, threshold4]

        # Random training process.
        for _ in xrange(n_rounds):

            # Choose a test function randomly.
            i = random_state.randint(0, len(test_funcs) - 1)
            func = test_funcs[i]

            # Generate random inputs according to the chosen function.
            # threshold1(...)
            if i == 0:

                # Select one feature randomly.
                col = random_state.randint(0, n_features)
                thresh = random_state.uniform(x_train[:, col].min(),
                                              x_train[:, col].max())

                args = col, thresh
                z = x_train[:, col] <= thresh

            # Other functions.
            else:

                # Select two features randomly.
                selected = random_state.randint(0, n_features, 2)
                col1, col2 = selected[0], selected[1]

                # threshold2(...)
                if i == 1:
                    diff = x_train[:, col1] - x_train[:, col2]
                # threshold3(...)
                elif i == 2:
                    diff = x_train[:, col1] + x_train[:, col2]
                # threshold5(...)
                elif i == 3:
                    diff = np.abs(x_train[:, col1] - x_train[:, col2])

                thresh = random_state.uniform(diff.min(), diff.max())
                args = col1, col2, thresh
                z = diff <= thresh

            # Evaluate the split quality.
            inf = information_gain(y_train[~z], y_train[z])

            # Compute best split according to the information gain.
            if inf <= min_inf:
                min_inf = inf
                # Store chosen function ...
                self.func = func
                # ... and its input parameters into node.
                self.args = args

        return self

    def split(self, x):
        """ Classify data x using this node.
        Of course, the node must be trained first.

        """
        return self.func(x, *self.args)


#______________________________________________________________________________


class Leaf():
    """ Leaf implementation class.

    """

    def __init__(self, label):
        self.label = label
        self.is_a_leaf = True
