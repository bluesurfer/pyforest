pyForest
========
A pure Python implementation of Random Forests specifically developed for face detection purposes.

*Random forests* are an ensemble learning method for **classification** (and regression) that operate by constructing a multitude of **decision trees** at training time and outputting the class that is the mode of the classes output by individual trees.


Dataset
-------
We use the `CBCL dataset <http://cbcl.mit.edu/software-datasets/FaceData2.html>`_ to train a random forest for face detection.


Usage Example
-------------
::

	python trainforest.py datasets/CBCLfaces.pkl
	python detectfaces.py forests/rf_t20_d-1_r200_s5.pkl images/astro.png
