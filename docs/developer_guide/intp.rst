=======================
EpwBaseIntpWorkChain
=======================

This work chain is the base work chain for the interpolation of the electron-phonon coupling.

It provides the interface for any workchain with the following logic:

.. code-block:: python
    spec.outline(
        cls.setup,
        if_(cls.should_run_b2w)(
            cls.run_b2w,
            cls.inspect_b2w,
        ),
        cls.prepare_process,
        cls.run_process,
        cls.inspect_process,
        cls.results
    )

The workflow is an aggregation of the following work chains:

- EpwB2WWorkChain: transformation from coarse-grid Bloch basis to Wannier basis
- EpwBaseWorkChain: interpolation from Wannier to Bloch representation (fine grid)

For the EpwB2WWorkChain, it will accept the following inputs:

.. code-block:: python
    spec.expose_inputs(
        EpwB2WWorkChain, namespace=cls._B2W_NAMESPACE, exclude=(
            'structure',
            'clean_workdir',
        ),
        namespace_options={
            'required': False,
            'populate_defaults': False,
            'help': 'Inputs for the `EpwB2WWorkChain`.'
        }
    )


For the EpwBaseWorkChain, it will accept the following inputs:

.. code-block:: python

    spec.expose_inputs(
        EpwBaseWorkChain, namespace=cls._INTP_NAMESPACE, exclude=(
            'structure',
            'clean_workdir',
            'parent_folder_nscf',
            'parent_folder_ph',
            'parent_folder_chk',
            # 'parent_folder_epw',
        ),
        namespace_options={
            'required': False,
            'populate_defaults': False,
            'help': 'Inputs for the a2f `EpwBaseWorkChain`.'
        }
    )