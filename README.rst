MRSim
=====

MRSim is a Pytorch-based MR simulator, including analitical and EPG model.

|Coverage| |CI| |CD| |License| |Codefactor| |Sphinx| |PyPi| |Black| |PythonVersion|

.. |Coverage| image:: https://infn-mri.github.io/mrsim/_static/coverage_badge.svg
   :target: https://infn-mri.github.io/mrsim

.. |CI| image:: https://github.com/INFN-MRI/mrsim/workflows/CI/badge.svg
   :target: https://github.com/INFN-MRI/mrsim

.. |CD| image:: https://github.com/INFN-MRI/mrsim/workflows/CD/badge.svg
   :target: https://github.com/INFN-MRI/mrsim

.. |License| image:: https://img.shields.io/github/license/INFN-MRI/mrsim
   :target: https://github.com/INFN-MRI/mrsim/blob/main/LICENSE.txt

.. |Codefactor| image:: https://www.codefactor.io/repository/github/INFN-MRI/mrsim/badge
   :target: https://www.codefactor.io/repository/github/INFN-MRI/mrsim

.. |Sphinx| image:: https://img.shields.io/badge/docs-Sphinx-blue
   :target: https://infn-mri.github.io/mrsim

.. |PyPi| image:: https://img.shields.io/pypi/v/mrsim
   :target: https://pypi.org/project/mrsim

.. |Black| image:: https://img.shields.io/badge/style-black-black

.. |PythonVersion| image:: https://img.shields.io/badge/Python-%3E=3.10-blue?logo=python&logoColor=white
   :target: https://python.org

Features
--------


Installation
------------

MRSim can be installed via pip as:

.. code-block:: bash

    pip install mrsim

Basic Usage
-----------

Development
~~~~~~~~~~~

If you are interested in improving this project, install MRSim in editable mode:

.. code-block:: bash

    git clone git@github.com:INFN-MRI/mrsim
    cd mrsim
    pip install -e .[dev,test,doc]


Related projects
----------------

This package is inspired by the following excellent projects:

- epyg <https://github.com/brennerd11/EpyG>
- sycomore <https://github.com/lamyj/sycomore/>
- mri-sim-py <https://somnathrakshit.github.io/projects/project-mri-sim-py-epg/>
- ssfp <https://github.com/mckib2/ssfp>
- erwin <https://github.com/lamyj/erwin>

