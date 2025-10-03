.. _installation:

================
Installation
================

As a prerequisite, you need to have the following packages installed:

- aiida-core
- aiida-wannier90-workflows
- aiida-quantumespresso

To use aiida-epw-workflows as a broker to your EPW jobs, I strongly recommend you to install the newest version of quantum ESPRESSO 7.5 (with EPW 6.0).

Early versions of EPW are not well supported.

The package is not packed in PyPI. You can install it from the source code.

.. code-block:: bash

   git clone https://github.com/ymzhang0/aiida-epw-workflows.git
   cd aiida-epw-workflows
   pip install (-e) .

