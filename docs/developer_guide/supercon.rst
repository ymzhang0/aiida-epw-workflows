=======================
EpwSuperConWorkChain
=======================

This work chain is used to compute the superconductor critical temperature on different level of theories.

It is an aggregation of the following work chains:

- PwRelaxWorkChain: relaxation of the structure
- PwBandsWorkChain: Calculation of the quantum ESPRESSO electron bands
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
        if_(cls.should_run_pw_relax)(
            cls.run_pw_relax,
            cls.inspect_pw_relax,
        ),
        if_(cls.should_run_pw_bands)(
            cls.run_pw_bands,
            cls.inspect_pw_bands,
        ),
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

Firstly, the workchain may do structural relaxation and calculation of qe electron bands, if necessary.

The qe electron bands are to be used as an input for the Wannier90WorkChain but it's not implemented yet.

The next step is to compute the electron-phonon coupling on Wannier basis in the `EpwB2WWorkChain`.

For a detailed description of the `EpwB2WWorkChain`, please refer to the :ref:`EpwB2WWorkChain documentation <b2w>`.

Then the workchain will proceed to the interpolation of the electron and phonon bands using the `EpwBandsWorkChain`. It will check the phono stability of the EPW interpolated bands. For more information on the EpwBandsWorkChain, please refer to the :ref:`EpwBandsWorkChain documentation <bands>`.

Then the workchain will proceed to a convergence test of the Allen-Dynes Tc with respect to the fine grids of the electron-phonon coupling using the `EpwA2FWorkChain`. For more information on the EpwA2FWorkChain, please refer to the :ref:`EpwA2FWorkChain documentation <a2f>`.

On the converged fine grids, the workchain will proceed to the computation of the superconductor critical temperature on the isotropic and anisotropic levels using the `EpwIsoWorkChain` and `EpwAnisoWorkChain`. For more information on the EpwIsoWorkChain and EpwAnisoWorkChain, please refer to the :ref:`EpwIsoWorkChain documentation <iso>` and :ref:`EpwAnisoWorkChain documentation <aniso>`.

--------------------------------
The Restart logic
--------------------------------

`EpwSuperConWorkChain` is a very complex vector of work chains. We might need to restart from some stop points.

To implement the restart mechanism, we implement the `get_builder_restart` functions.

It is a class method taking only the previous workchain `from_supercon_workchain` as the input.

.. code-block:: python

    @classmethod
    def get_builder_restart(
        cls,
        from_supercon_workchain: orm.WorkChainNode,
        ):
        ...

The `from_supercon_workchain` is the previous workchain of the current workchain.

get_builder_restart will take the advantage of the get_builder_restart functions of the subprocesses to make maximal use of codes.

get_builder_restart function will automatically parse the parent workchains and fill the inputs of a new builder according to the progress of the previous workchain.

To make the restart method work, we need to always make inputs like `parent_folder_epw`, `parent_folder_ph`... to be optional. This is not desireable but a necessary compromise.

The other drawback of this kind of restart logic is one successful workchain is broken into several failed workchain fragments.

A better way to implement the restart logic is still under investigation.

--------------------------------
The Analyser
--------------------------------

The `EpwSuperConWorkChain` has a very complex structure. It is difficult to analyse the workchain manually.

To make the analysis of the results easier, we provide an analyser `EpwSuperConWorkChainAnalyser` which is included in aiida_epw_workflows.tools.analysers.supercon.

This analyser provide convenient methods to:

- Check the state of the workchain
- Get the results of the workchain
- Plot the bands, gap functions, etc.
- Clean the failed workchains

