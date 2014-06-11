#!/usr/bin/env python
"""Core library to build and test the random forest."""

from __future__ import division

import numpy as np

from multiprocessing import Manager, cpu_count

from parallel import Worker, Task, retrieve
from base import Node, Leaf


__author__ = "Andrea Casini"
__copyright__ = "Copyright 2014"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "acasini@dsi.unive.it"


def _build_tree(x, y, f, random_state, root=Node(), depth=0):
    """ Build a decision tree for a given data.

    """

    n_samples = len(y)

    # 1st base case : empty arrays.
    if n_samples == 0:
        return

    counts = np.bincount(y)
    n_classes = np.count_nonzero(counts)

    # 2nd base case : all node's labels are equal.
    if n_classes <= 1:
        return Leaf(y[0])

    # 3rd base case : maximum depth or minimum sample size reached.
    if depth >= f.max_depth != -1 or n_samples <= f.min_samples_split != -1:
        return Leaf(counts.argmax())

    # Train this node ...
    root.train(x, y, f.n_rounds, random_state)
    # ... and use it to split x
    z = root.split(x)

    assert z is not None

    # Recursive calls.
    root.left = _build_tree(x[~z], y[~z], f, random_state, Node(), depth + 1)
    root.right = _build_tree(x[z], y[z], f, random_state, Node(), depth + 1)

    return root


def _tree_predict(x, root, indices=None, output=None):
    """ Compute labels predictions of a single tree.

    """
    if indices is None and output is None:
        indices = np.arange(x.shape[0])
        output = np.zeros(x.shape[0])

    if len(indices) == 0 or root is None:
        return

    if root.is_a_leaf:
        output[indices] = root.label
        return

    z = root.split(np.take(x, indices, axis=0))

    _tree_predict(x, root.left, indices[~z], output)
    _tree_predict(x, root.right, indices[z], output)

    return output


#______________________________________________________________________________


class RandomForest():
    """ A random forest classifier.

    A random forest is a meta estimator that fits a number of classifical
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.

    Parameters
    ----------
    n_trees : integer, optional (default = 20)
        The number of trees in the forest.

    max_depth : integer, optional (default = -1)
        The maximum depth of the tree. If -1, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Note: this parameter is tree-specific.

    n_rounds : integer, optional (default = 10)
        The number of splits to perform in order to find the best one.

    min_samples_split : integer, optional (default = 1)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.

    n_jobs : integer, optional (default = -1)
        The number of jobs to run in parallel. If -1, then the number of jobs
        is set to the number of cores.

    """

    def __init__(self,
                 n_trees=20,
                 max_depth=-1,
                 n_rounds=100,
                 min_samples_split=1,
                 n_jobs=-1,
                 forest=None):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.forest = forest

    def fit(self, x, y):
        """ Build a random forest of trees from the training set (x, y).

        Parameters
        ----------
        x : array-like of shape = [n_samples, n_features]
           The training input samples.

        y : array-like of shape = [n_samples]
           The target values (integers that correspond to classes).

        Returns
        -------
        self : object
           Returns self.

        """
        if self.n_jobs == -1:
            n_workers = min(cpu_count(), self.n_trees)
        else:
            n_workers = min(self.n_jobs, self.n_trees)

        # Establish communication queues.
        tasks = Manager().JoinableQueue()
        results = Manager().Queue()

        # Start workers.
        workers = [Worker(tasks, results) for _ in xrange(n_workers)]

        for w in workers:
            w.start()

        # Populate task's queue.
        for i in xrange(self.n_trees):
            # Create a new random state for each tree.
            random_state = np.random.RandomState(i)
            tasks.put(Task(_build_tree, (x, y, self,random_state), i))

        # Add a poison pill for each worker.
        for i in xrange(n_workers):
            tasks.put(None)

        # Wait for all of the tasks to finish.
        tasks.join()

        # Retrieve results i.e. the trees from the queue.
        self.forest = retrieve(results, self.n_trees)

        return self

    def predict(self, x):
        """ Predict class for test set x.

        The predicted class of an input sample is computed as the majority
        prediction of the trees in the forest.

        Parameters
        ----------
        x : array-like of shape = [n_samples, n_features]
           The test input samples.

        Returns
        -------
        y_pred : array of shape = [n_samples]
                The predicted classes.

        probs : array of shape = [n_samples]
                Probabilities of each sample to belong to the predicted class.

        """
        if self.n_jobs == -1:
            n_workers = min(cpu_count(), self.n_trees)
        else:
            n_workers = min(self.n_jobs, self.n_trees)

        # Establish communication queues.
        tasks = Manager().JoinableQueue()
        results = Manager().Queue()

        # Start workers.
        workers = [Worker(tasks, results) for _ in xrange(n_workers)]

        for w in workers:
            w.start()

        # Populate task's queue.
        for i in xrange(self.n_trees):
            tasks.put(Task(_tree_predict, (x, self.forest[i]), i))

        # Add a poison pill for each worker.
        for i in xrange(n_workers):
            tasks.put(None)

        # Wait for all of the tasks to finish.
        tasks.join()

        # Retrieve results i.e. the votes of the trees from the queue i.e
        # an array of shape [n_trees, n_samples].
        votes = np.array(retrieve(results, self.n_trees), int)

        # Count up the votes of the trees.
        n_classes = len(np.unique(votes))
        counts = np.apply_along_axis(
            lambda z: np.bincount(z, minlength=n_classes), 0, votes)

        # Classify each sample according to the majority of the votes.
        y_pred = np.argmax(counts, axis=0)

        return y_pred, counts / self.n_trees