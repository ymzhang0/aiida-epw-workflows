.. _overview:

===================
Project Structure
===================

This page provides an overview of the ``aiida-epw-workflows`` repository structure, explaining the role of each key directory and module. The layout is based on standard AiiDA plugin conventions to facilitate maintainability and collaboration.

High-Level Overview
-------------------

The project follows a layered architecture to maximize code reuse and separate concerns.

.. code-block:: text

   aiida-epw-workflows/
   ├── aiida_epw_workflows/
   │   ├── workflows/
   │   │   ├── __init__.py
   │   │   ├── base.py
   │   │   ├── b2w.py
   │   │   ├── intp.py
   │   │   ├── a2f.py
   │   │   ├── iso.py
   │   │   ├── aniso.py
   │   │   └── supercon.py
   │   ├── parsers/
   │   │   └── epw.py
   │   ├── controllers/
   │   │   └── supercon.py
   │   └── utils/
   │       ├── __init__.py
   │       └── helpers.py
   ├── docs/
   ├── tests/
   └── pyproject.toml

Module Breakdown
=================

``aiida_epw_workflows/workflows/``
**********************************

This is the core of the plugin, containing all the AiiDA ``WorkChain`` definitions.

``base.py``: `EpwBaseWorkChain`
================================
This is the lowest-level workchain wrapper around a single ``EpwCalculation``. It should include robust error handling and restart logic for a single ``epw.x`` run, but does not handle complex, multi-step physics workflows.

``b2w.py``: `EpwB2WWorkChain`
===============================
This is a wrapper around the `Wannier90OptimizeWorkChain`, `PhBaseWorkChain` and `EpwBaseWorkChain`. It internally runs the full `Wannier90OptimizeWorkChain`, a `PhBaseWorkChain`, and a final `EpwBaseWorkChain` to generate the electron-phonon matrix elements in the Wannier representation (``.epmatwp`` files). It can be run standalone or as a preparatory step for other calculations.

``intp.py``: `EpwBaseIntpWorkChain`
====================================
This is a crucial **base class** that defines a generic **two-step computational template**:

1.  A preparatory step (usually running `EpwB2WWorkChain`).
2.  A main interpolation step (running `EpwBaseWorkChain` using the results from step 1).

It contains all the shared logic for handling restarts, setting up inputs, and passing the ``parent_folder`` from the first step to the second. It is not meant to be run directly.

``bands.py``, ``a2f.py``, ``iso.py``, ``aniso.py``, ``bte.py``: Concrete Implementations
=========================================================================================
These are the workchains you would typically run for a specific task. They all **inherit** from ``EpwBaseIntpWorkChain``.

Their code is very concise, as they only need to:
1.  Define their specific namespace (e.g., ``_INTP_NAMESPACE = 'a2f'``).
2.  Provide the specific `parameters` and `settings` needed for their main calculation (e.g., setting ``a2f = .true.``). This is done by overriding the ``prepare_process`` "hook" method.
3.  Define their unique outputs and results processing in the ``inspect_process`` method.

Any new workchain is recommended to inherit from ``EpwBaseIntpWorkChain`` if it is a two-step workflow.

``supercon.py``: `EpwSuperconWorkChain`
=========================================
This is the highest-level **orchestrator** workchain. It coordinates a complex computational pipeline by **composing** the other workchains:

1.  It runs the `EpwB2WWorkChain` **once** to get the shared Wannier-representation matrix elements.
2.  It then uses the output of this single `b2w` run to launch the `EpwBandWorkChain` to get the interpolated electron and phonon bands.
3.  It then uses the output of this single `b2w` run to launch the `EpwA2fWorkChain` on different fine grids for a convergence test with respect to the Allen-Dynes Tc.
4.  It subsequently runs the `EpwIsoWorkChain` and `EpwAnisoWorkChain` to get the isotropic and anisotropic critical temperatures.

``transport.py``: `EpwTransportWorkChain`
=========================================
This is the workchain for calculating the transport properties.




``aiida_epw_workflows/controllers/``
*************************************
This module contains submission controllers based on `aiida-submission-controller`. The ``EpwSuperconWorkChainController`` provides a powerful interface for submitting large batches of ``EpwSuperconWorkChain`` calculations, for instance, for a high-throughput screening campaign across a group of structures. It handles duplicate checking and concurrency management. It is to be developed in the future.