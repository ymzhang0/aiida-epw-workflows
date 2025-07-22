************
Quick Start
************

This tutorial will guide you through running a complete `EpwSuperconWorkChain` to calculate the superconducting properties of a material.

Step 1: Setup your AiiDA environment
=======================================

First, make sure you are in a running `verdi shell` or have loaded the AiiDA profile in your Python script.

.. code-block:: python

   from aiida import orm, engine

   # Load all the necessary codes
   codes = {
       'pw': orm.load_code('pw-7.2@my_cluster'),
       'ph': orm.load_code('ph-7.2@my_cluster'),
       'epw': orm.load_code('epw-5.4@my_cluster'),
       'wannier90': orm.load_code('wannier90-3.1@my_cluster'),
       'pw2wannier90': orm.load_code('pw2wannier90-7.2@my_cluster'),
   }

.. note::
1. It is recommended to use PDWF to automate the wannierization. And one can find the installation here: https://github.com/qiaojunfeng/wannier90/tree/p2w_ham
2. If you want to use intermediate representation for an accelaration of low temperature calculation, you should add another 'epw_ir' code.

Important: please download the package from https://github.com/ymzhang0/aiida-epw-workflows.

Step 2: Prepare the input structure
====================================

Load the crystal structure you want to calculate.

For superconductivity calculation, lead is a good example.

.. code-block:: python

   # Load a structure from its PK or UUID
   structure = orm.load_node(123)

Step 3: Create the builder
==========================

We will use the `get_builder_from_protocol` factory method to easily set up the inputs. We will run a "fast" calculation from scratch.

.. code-block:: python

   from aiida_epw_workflows.workflows.supercon import EpwSuperconWorkChain
   from aiida_wannier90_workflows.common.types import WannierProjectionType

   builder = EpwSuperconWorkChain.get_builder_from_protocol(
       codes=codes,
       structure=structure,
       protocol='fast',  # Use the 'fast' protocol for a quick test
       # We can provide overrides for specific parameters if needed
       overrides={
           'b2w': {
               'w90_intp': {
                   'scf': {'pw': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}},
                   'nscf': {'pw': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}},
                   'wannier90': {'wannier90': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}},
                   'pw2wannier90': {'pw2wannier90': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}},
               },
               'ph_base': {
                   'ph': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}
               },
               'epw':{
                    'epw': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}
                },
           }
       },
       # Specify the wannierization scheme, here it is PDWF.
       wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
       # Specify the script to convert the wannier90 checkpoint file to the ukk format that is used for EPW.
       w90_chk_to_ukk_script = w90_script,
   )

   # You can modify the builder further if needed, e.g., for cleanup
   builder.clean_workdir = orm.Bool(True)


Step 4: Submit and run the calculation
=======================================

Use the AiiDA engine to run the workflow and get the results.

.. code-block:: python

   node, results = engine.run_get_node(builder)

Step 5: Inspect the results
===========================

Once the `EpwSuperconWorkChain` has finished successfully, you can inspect its outputs.

.. code-block:: python

   print(f"WorkChain finished with status: {node.process_state}")
   print(f"Available outputs: {results.keys()}")

   # You may plot the electron/phonon band structures, density of states, (accumulated) spectral functions using the tools provided by the package.
   from aiida_epw_workflows.tools.plot import (
        plot_epw_interpolated_bands,
        plot_a2f,
        plot_eldos,
    )

   import matplotlib.pyplot as plt
   import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[0, 1])

    descendants = EpwSuperConWorkChain.get_descendants_workchains(
        node,
        link_type=LinkType.CALL,
    )
    plot_epw_interpolated_bands(
        epw_workchain = descendants['bands'][0],
        axes=numpy.array([ax1, ax2]),
    )

    plot_a2f(
        a2f_workchain = descendants['a2f'][0],
        axis = ax3,
        show_data = True,
        )

    plot_eldos(
        a2f_workchain = descendants['a2f'][0],
        axis = ax4,
        )

   # Get the final Allen-Dynes Tc from the 'a2f' sub-process results

    tc = descendants['a2f'][0].outputs.output_parameters.get('Allen_Dynes_Tc')
    print(f"Calculated Allen-Dynes Tc = {tc:.2f} K")

    # You can also get the isotropic Tc from the 'iso' sub-process results
    tc = descendants['iso'][0].outputs.output_parameters.get('Allen_Dynes_Tc')
    print(f"Calculated Allen-Dynes Tc = {tc:.2f} K")

    # You can also get the anisotropic Tc from the 'aniso' sub-process results
    tc = descendants['aniso'][0].outputs.output_parameters.get('Allen_Dynes_Tc')
    print(f"Calculated Allen-Dynes Tc = {tc:.2f} K")

This concludes the quick start tutorial. For more advanced topics, such as restarting calculations or using the submission controller, please refer to the User Guide.