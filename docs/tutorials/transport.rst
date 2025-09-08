************
EpwTransportWorkChain
************

This tutorial will guide you through running a complete `EpwTransportWorkChain` to calculate the transport properties of a material.

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

For transport calculation, you may choose silicon as an example. You can find the structures in the `examples/structures/` folder within the package.

.. code-block:: python

   # Load a structure from files
   structure =  read_structure_from_file(
        package / 'examples/structures/Si.xsf'
        )


Step 3: Create the builder for transport calculation
==========================

We will use the `get_builder_from_protocol` factory method to easily set up the inputs. We will run a "fast" calculation from scratch.

.. code-block:: python

   from aiida_epw_workflows.workflows.transport import EpwTransportWorkChain
   from aiida_wannier90_workflows.common.types import WannierProjectionType

   builder = EpwTransportWorkChain.get_builder_from_protocol(
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
               'epw_base':{
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

Please refer to the override.yaml inside the protocols folder for the structure of the overrides.

Step 4: Submit and run the calculation
=======================================

Use the AiiDA engine to run the workflow and get the results.

.. code-block:: python

   node, results = engine.run_get_node(builder)

Step 5: Inspect the results
===========================

Once the `EpwTransportWorkChain` has finished successfully, you can inspect its outputs.

This concludes the quick start tutorial. For more advanced topics, such as restarting calculations or using the submission controller, please refer to the User Guide.