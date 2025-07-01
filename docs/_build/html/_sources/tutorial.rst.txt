************
Quick Start
************

This tutorial will guide you through running a complete `EpwSuperconWorkChain` to calculate the superconducting properties of a material from scratch.

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

Step 2: Prepare the input structure
====================================

Load the crystal structure you want to calculate.

.. code-block:: python

   # Load a structure from its PK or UUID
   structure = orm.load_node(123)

Step 3: Create the builder
==========================

We will use the `get_builder_from_protocol` factory method to easily set up the inputs. We will run a "fast" calculation from scratch.

.. code-block:: python

   from aiida_supercon.workflows.supercon import EpwSuperconWorkChain
   from aiida_wannier90_workflows.common.types import WannierProjectionType

   builder = EpwSuperconWorkChain.get_builder_from_protocol(
       codes=codes,
       structure=structure,
       protocol='fast',  # Use the 'fast' protocol for a quick test
       # We can provide overrides for specific parameters if needed
       overrides={
           'b2w': {
               'w90_intp': {
                   'scf': {'pw': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}}
               }
           }
       },
       # Specify other high-level options
       wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE
   )

   # You can modify the builder further if needed, e.g., for cleanup
   builder.clean_workdir = orm.Bool(True)


Step 4: Submit and run the calculation
=======================================

Use the AiiDA engine to run the workflow and get the results.

.. code-block:: python

   results, node = engine.run_get_node(builder)

Step 5: Inspect the results
===========================

Once the `EpwSuperconWorkChain` has finished successfully, you can inspect its outputs.

.. code-block:: python

   print(f"WorkChain finished with status: {node.process_state}")
   print(f"Available outputs: {results.keys()}")

   # Get the final Allen-Dynes Tc from the 'a2f' sub-process results
   if 'a2f' in node.outputs:
        a2f_results = node.outputs.a2f
        if 'tc_allen_dynes' in a2f_results:
            tc = a2f_results.tc_allen_dynes.value
            print(f"Calculated Allen-Dynes Tc = {tc:.2f} K")

This concludes the quick start tutorial. For more advanced topics, such as restarting calculations or using the submission controller, please refer to the User Guide.