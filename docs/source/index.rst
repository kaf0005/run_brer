.. run BRER documentation master file, created by
   sphinx-quickstart on Thu Jun 27 11:01:26 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to run BRER's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


   user_api
   developer_api

README
========

|Documentation Status|
|Language grade: Python|
|Total alerts| 
|https://www.singularity-hub.org/static/img/hosted-singularity–hub-%23e32929.svg|
|Build Status|
|codecov|

Set of scripts for running BRER simulations using gmxapi. Details of
this method may be found at:

Hays, J. M., Cafiso, D. S., & Kasson, P. M. Hybrid Refinement of
Heterogeneous Conformational Ensembles using Spectroscopic Data. *The
Journal of Physical Chemistry Letters*. DOI:
`10.1021/acs.jpclett.9b01407 <https://pubs.acs.org/doi/10.1021/acs.jpclett.9b01407>`__

Installation
------------

Requirements
~~~~~~~~~~~~

If you’re going to use a pip or a conda environment, you’ll need:

-  Python 3. X
-  An installation of
   `gromacs-gmxapi <http://github.com/kassonlab/gromacs-gmxapi>`__.
   Currently, ``gmxapi`` does not support

domain decomposition with MPI, so if you want these simulations to run
fast, be sure to compile with GPU support.

-  An installation of `gmxapi <https://github.com/kassonlab/gmxapi>`__.

This code has only been tested with `Gromacs
2019 <http://manual.gromacs.org/documentation/2019/index.html>`__.

-  The `plugin
   code <https://github.com/jmhays/sample_restraint/tree/corr-struct>`__
   for BRER. Please make sure you install the

``corr-struct`` branch, *NOT* ``master`` .

Otherwise, you can just use a Singularity container!

Singularity
~~~~~~~~~~~

By far the easiest option! If you are working with an older Singularity
version (< 3), pull the container hosted on singularity hub:

``singularity pull -name myimage.simg shub://jmhays/singularity-brer``

If you have the latest and greatest Singuarity (v > 3), you can pull the
container from the *new* cloud repository:

``singularity pull library://jmhays/default/brer:latest``

For instructions on using the container, please see
`this <https://github.com/jmhays/singularity-brer>`__ repository.

Conda environment
~~~~~~~~~~~~~~~~~

I suggest running this in a conda environment rather than
``pip install`` . The following conda command will handle all the
``gmxapi`` and ``sample_restraint`` python dependencies, as well as the
ones for this repository.

1. ``conda create -n BRER numpy scipy networkx setuptools mpi4py cmake``

   If you want to run the tests, then install ``pytest`` as well.

2. Source, the environment, then use the standard Python ``setup.py``
   protocol:

::

   source activate BRER
   git clone https: //github.com/jmhays/run_brer.git
       cd run_brer
   python setup.py install

Running BRER
------------

Launching a single ensemble member.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example script, ``run.py`` , is provided for ensemble simulations.

Let’s work through it piece by piece.

::

   #!/usr/bin/env python

   ""
   "
   Example run script
   for BRER simulations
       ""
   "

   import run_brer.run_config as rc
   import sys

The ``import run_brer.run_config`` statement imports a ``RunConfig``
object, which handles the following things **for a single ensemble
member**:

1. Initializing/setting up parameters for the BRER run.
2. Launching the run.

Then we provide some files and directory paths to the ``RunConfig``
object.

::

   init = {
       'tpr': '/home/jennifer/Git/run_brer/tests/syx.tpr',
       'ensemble_dir': '/home/jennifer/test-brer',
       'ensemble_num': 5,
       'pairs_json': '/home/jennifer/Git/run_brer/tests/pair_data.json'
   }

   config = rc.RunConfig( ** init)

In order to run a BRER simulation, we need to provide :

1. a ``tpr`` (compatible with GROMACS 2019).
2. The path to our ensemble. This directory should contain
   subdirectories of the form ``mem_<my ensemble number>``
3. The ensemble number. This is an integer used to identify which
   ensemble member we are running and thus, the subdirectory in which we
   will be running our simulations.
4. The path to the DEER metadata. Please see the example json in this
   repository: ``run_brer/data/pair_data.json``

Finally, we launch the run!

::

   config.run()

You may change various parameters before launching the run using
``config.set(**kwargs)`` . For example:

::

   config = rc.RunConfig( ** init)
   config.set(A = 100)
   config.run()

resets the energy constant A to 100 kcal/mol/nm^2 before launching a
run.

Launching an ensemble
^^^^^^^^^^^^^^^^^^^^^

Right now, the way to launch an ensemble is to launch multiple jobs. We
hope to soon use the ``gmxapi``
`features <https://github.com/kassonlab/gmxapi>`__ that allow a user to
launch many ensemble members in one job.

.. |Documentation Status| image:: https://readthedocs.org/projects/run-brer/badge/?version=latest
.. |Language grade: Python| image:: https://img.shields.io/lgtm/grade/python/g/jmhays/run_brer.svg?logo=lgtm&logoWidth=18
.. |Total alerts| image:: https://img.shields.io/lgtm/alerts/g/jmhays/run_brer.svg?logo=lgtm&logoWidth=18
.. |https://www.singularity-hub.org/static/img/hosted-singularity–hub-%23e32929.svg| image:: https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg
.. |Build Status| image:: https://travis-ci.com/jmhays/run_brer.svg?token=zQbC3QZqV1zHSGhQXUTP&branch=devel
.. |codecov| image:: https://codecov.io/gh/jmhays/run_brer/branch/devel/graph/badge.svg

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
