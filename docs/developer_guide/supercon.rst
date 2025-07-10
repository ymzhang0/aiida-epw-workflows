=======================
EpwSuperConWorkChain
=======================

This work chain is used to compute the superconductor critical temperature on different level of theories.

It is an aggregation of the following work chains:

- EpwB2WWorkChain: transformation from coarse-grid Bloch basis to Wannier basis
- EpwBandsWorkChain: Interpolation of the electron and phonon bands
- EpwA2FWorkChain: Calculation of the Eliashberg spectral function
- EpwIsoWorkChain: Compute the superconductor critical temperature on the isotropic approximation
- EpwAnisoWorkChain: Compute the superconductor critical temperature on the anisotropic level with restriction on Fermi surface.

It follows the following steps:

.. code-block:: python

    spec.outline(
        cls.setup,
        cls.validate_inputs,
        if_(cls.should_run_b2w)(
            cls.run_b2w,
            cls.inspect_b2w,
        ),
        if_(cls.should_run_bands)(
            cls.run_bands,
            cls.inspect_bands,
        ),
        cls.prepare_intp,
        while_(cls.should_run_conv)(
            cls.run_conv,
            cls.inspect_conv,
        ),
        if_(cls.should_run_a2f)(
            cls.run_a2f,
            cls.inspect_a2f,
        ),
        if_(cls.should_run_iso)(
            cls.run_iso,
            cls.inspect_iso,
        ),
        if_(cls.should_run_aniso)(
            cls.run_aniso,
            cls.inspect_aniso,
        ),
        cls.results
    )

The first step is to compute the electron-phonon coupling on Wannier basis.




For the EpwB2WWorkChain, it will accept the following inputs:

.. code-block:: python

For the interpolation subprocesses, it will exclude the following inputs:

- parent_folder_chk
- parent_folder_ph
- parent_folder_nscf

Because the interpolation subprocesses are not allowed to restart from the previous ``EpwB2WWorkChain``, it only need to accept the parent epw folder.

For the ``EpwBandsWorkChain`` and ``EpwA2FWorkChain``, they restart from .epmatwp file. So they take ``remote_folder`` of ``EpwB2WWorkChain`` as the parent folder.

For the ``EpwIsoWorkChain`` and ``EpwAnisoWorkChain``, they share the same find grids settings. So they can restart from .ephmat file from ``EpwA2FWorkChain``.



--------------------------------
The Restart Mechanism
--------------------------------

``EpwSuperConWorkChain`` is a very complex vector of work chains. We might need to restart from some stop points.

To implement the restart mechanism, we implement the *get_builder_restart* functions.

It is a class method taking only the previous workchain *from_supercon_workchain* as the input.

.. code-block:: python

    @classmethod
    def get_builder_restart(
        cls,
        from_supercon_workchain: orm.WorkChainNode,
        ):
        ...

The *from_supercon_workchain* is the previous workchain of the current workchain.

get_builder_restart will take the advantage of the get_builder_restart functions of the subprocesses to make maximal use of codes.

get_builder_restart function will automatically parse the parent workchains and fill the inputs of a new builder according to the progress of the previous workchain.


To make the restart method work, we need to always make inputs like *parent_folder_epw*, *parent_folder_ph*... to be optional.

For example, if we run the ``EpwSuperConWorkChain`` from scratch, the inputs for ``EpwA2FWorkChain`` subprocess will take the remote_folder in ``EpwB2WWorkChain``'s outputs as the parent folder.

So the best is that we exclude the *parent_folder_epw* from the inputs of ``EpwA2FWorkChain`` and assign it upon the success of ``EpwB2WWorkChain``.

However, to make sure the workchain will also work for a restart from a previous workchain, we need to make the *parent_folder_epw* optional.








