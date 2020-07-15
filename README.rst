===========================================================================
BCGN: Block-Coordinate Gauss-Newton |License| |Build Status| |PyPI Version|
===========================================================================

BCGN is a python package for sketched Gauss-Newton in either the variable or the observation dimension.

Requirements
------------
BCGN requires the following software to be installed:

* Python 2.7 or Python 3 (http://www.python.org/)

Additionally, the following python packages should be installed (these will be installed automatically if using *pip*, see `Installation using pip`_):

* NumPy 1.11 or higher (http://www.numpy.org/)
* SciPy 0.18 or higher (http://www.scipy.org/)

Installation using pip
----------------------
For easy installation, use `pip <http://www.pip-installer.org/>`_ as root:

 .. code-block:: bash
 
    $ [sudo] pip install BCGN

If you do not have root privileges or you want to install BCGN for your private use, you can use:

 .. code-block:: bash
 
    $ pip install --user BCGN

which will install BCGN in your home directory.

Note that if an older install of BCGN is present on your system you can use:

 .. code-block:: bash

    $ [sudo] pip install --upgrade BCGN

to upgrade BCGN to the latest version.

Manual installation
-------------------
Alternatively, you can download the source code from `Github <https://github.com/jfowkes/BCGN>`_ and unpack as follows:

 .. code-block:: bash

    $ git clone https://github.com/jfowkes/BCGN
    $ cd BCGN

BCGN is written in pure Python and requires no compilation. It can be installed using:

 .. code-block:: bash

    $ [sudo] pip install .

If you do not have root privileges or you want to install BCGN for your private use, you can use:

 .. code-block:: bash

    $ pip install --user .

which will install BCGN in your home directory.

To upgrade BCGN to the latest version, navigate to the top-level directory (i.e. the one containing :code:`setup.py`) and rerun the installation using :code:`pip`, as above:

 .. code-block:: bash

    $ git pull
    $ [sudo] pip install .  # with root privileges

Testing
-------
If you installed BCGN manually, you can test your installation by running:

 .. code-block:: bash

    $ python setup.py test

Uninstallation
--------------
If BCGN was installed using *pip* you can uninstall as follows:

 .. code-block:: bash

    $ [sudo] pip uninstall BCGN

otherwise you have to remove the installed files by hand (located in your python site-packages directory).

Bugs
----
Please report any bugs using GitHub's issue tracker.

License
-------
This algorithm is released under the GNU GPL license.

.. |License| image::  https://img.shields.io/badge/License-GPL%20v3-blue.svg
             :target: https://www.gnu.org/licenses/gpl-3.0
             :alt: GNU GPL v3 License
.. |Build Status| image::  https://travis-ci.org/jfowkes/BCGN.svg?branch=master
                  :target: https://travis-ci.org/jfowkes/BCGN
.. |PyPI Version| image:: https://img.shields.io/pypi/v/BCGN.svg
                  :target: https://pypi.python.org/pypi/BCGN

