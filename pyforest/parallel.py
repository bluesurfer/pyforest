#!/usr/bin/env python
"""Support module for parallel computation."""

import multiprocessing as mc
import time
import logging


__author__ = "Andrea Casini"
__copyright__ = "Copyright 2014"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "acasini@dsi.unive.it"

__all__ = ['retrieve', 'Worker']


def retrieve(result_queue, queue_size):
    """ Retrieve results from a shared queue.

    """
    results = []
    while queue_size:
        results.append(result_queue.get())
        queue_size -= 1
    return results


class Worker(mc.Process):
    """ Worker class for multiprocessing.

    """

    def __init__(self, task_queue, result_queue, verbose=True):
        mc.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.verbose = verbose

    def run(self):

        proc_name = self.name

        while True:

            next_task = self.task_queue.get()

            if next_task is None:
                # Poison pill means shutdown.
                logging.info('%s Exiting' % proc_name)
                self.task_queue.task_done()
                break

            logging.info('%s: %s' % (proc_name, next_task))

            start = time.clock()
            answer = next_task()
            elapsed = time.clock() - start

            logging.info('%s: %s | Done in %1.2fs' % (proc_name, next_task, elapsed))

            self.task_queue.task_done()
            self.result_queue.put(answer)

        return


class Task():

    def __init__(self, do_work, args, task_id):
        self.do_work = do_work
        self.args = args
        self.id = task_id

    def __call__(self):
        """ Do work call.

        """
        return self.do_work(*self.args)

    def __str__(self):
        """ This is the name of the task.

        """
        return '%s %s' % (self.do_work.__name__, self.id)
