from aiida import orm
import copy
from typing import Optional, Dict, Any

from aiida_quantumespresso.common.types import SpinType, ElectronicType
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain

from aiida_submission_controller import FromGroupSubmissionController

from .. import EpwSuperConWorkChain


class SuperConWorkChainController(FromGroupSubmissionController):
    """A complete SubmissionController for submitting `EpwSuperConWorkChain`s."""
    
    # Code configurations
    pw_code: str
    ph_code: str
    q2r_code: str
    matdyn_code: str
    
    # Workflow parameters
    protocol: str = "moderate"
    overrides: Optional[dict] = None
    electronic_type: ElectronicType = ElectronicType.METAL
    spin_type: SpinType = SpinType.NONE
    
    # Process class
    _process_class = EpwSuperConWorkChain
    
    # Convergence parameters
    convergence_threshold: Optional[float] = None
    interpolation_distances: Optional[list] = None
    always_run_final: bool = True
    
    # Resource configurations
    max_wallclock_seconds: int = 3600
    account: str = "project_465000106"
    queue_name: str = "debug"
    num_machines: int = 1
    num_mpiprocs_per_machine: int = 8
    num_cores_per_machine: int = 8

    @staticmethod
    def get_extra_unique_keys():
        """Return a tuple of the keys of the unique extras that will be used to uniquely identify your workchains."""
        return ("formula_hill", "number_of_sites", "source_db", "source_id")

    def get_inputs_and_processclass_from_extras(self, extras_values, dry_run=False):
        """Return inputs and process class for the submission of this specific process."""
        parent_node = self.get_parent_node_from_extras(extras_values)

        # Depending on the type of node in the parent class, grab the right inputs
        if isinstance(parent_node, orm.StructureData):
            structure = parent_node
        elif parent_node.process_class == PhBaseWorkChain:
            structure = parent_node.outputs.output_structure
        elif parent_node.process_class == PwRelaxWorkChain:
            structure = parent_node.outputs.output_structure
        else:
            raise TypeError(
                f"Node {parent_node} from parent group is of incorrect type: {type(parent_node)}."
            )

        # Prepare codes dictionary
        codes = {
            "pw": orm.load_code(self.pw_code),
            "ph": orm.load_code(self.ph_code),
            "q2r": orm.load_code(self.q2r_code),
            "matdyn": orm.load_code(self.matdyn_code),
        }

        # Prepare overrides
        overrides = copy.deepcopy(self.overrides) if self.overrides else {}
        
        # Add resource configurations to overrides
        self._add_resource_configs_to_overrides(overrides)

        # Build the workchain
        builder = self._process_class.get_builder_from_protocol(
            codes=codes,
            structure=structure,
            protocol=self.protocol,
            overrides=overrides,
            electronic_type=self.electronic_type,
            spin_type=self.spin_type,
        )
        
        # Add optional parameters
        if self.convergence_threshold is not None:
            builder.convergence_threshold = orm.Float(self.convergence_threshold)
        
        if self.interpolation_distances is not None:
            builder.interpolation_distances = orm.List(list=self.interpolation_distances)
        
        builder.always_run_final = orm.Bool(self.always_run_final)
        builder.clean_workdir = orm.Bool(True)

        return builder

    def _add_resource_configs_to_overrides(self, overrides: Dict[str, Any]):
        """Add resource configurations to the overrides dictionary."""
        options = {
            'max_wallclock_seconds': self.max_wallclock_seconds,
            'account': self.account,
            'queue_name': self.queue_name
        }
        
        resources = {
            'num_machines': self.num_machines,
            'num_mpiprocs_per_machine': self.num_mpiprocs_per_machine,
            'num_cores_per_machine': self.num_cores_per_machine,
        }
        
        # Add to all relevant sub-workchains
        sub_workchains = ['b2w', 'a2f', 'iso', 'aniso']
        
        for wc_name in sub_workchains:
            if wc_name not in overrides:
                overrides[wc_name] = {}
            
            # Add to all calculation jobs in the sub-workchain
            self._add_options_to_calc_jobs(overrides[wc_name], options, resources)

    def _add_options_to_calc_jobs(self, wc_overrides: Dict[str, Any], options: Dict[str, Any], resources: Dict[str, Any]):
        """Recursively add options to all calculation jobs in a workchain."""
        for key, value in wc_overrides.items():
            if isinstance(value, dict):
                if 'metadata' in value and 'options' in value['metadata']:
                    # This is a calculation job
                    value['metadata']['options'].update(options)
                    if 'resources' not in value['metadata']['options']:
                        value['metadata']['options']['resources'] = resources
                else:
                    # Recursively process nested workchains
                    self._add_options_to_calc_jobs(value, options, resources)

    def get_status_summary(self):
        """Get a summary of the current status."""
        return {
            'total_to_submit': len(self.get_all_extras_to_submit()),
            'already_submitted': self.num_already_run,
            'still_to_run': self.num_to_run,
            'max_concurrent': self.max_concurrent,
            'active_slots': self.num_active_slots,
            'available_slots': self.num_available_slots,
        }

    def submit_with_monitoring(self, verbose=True, sleep_interval=30):
        """Submit workchains with continuous monitoring."""
        import time
        
        print(f"Starting SuperCon workchain submission with max_concurrent={self.max_concurrent}")
        
        while self.num_to_run > 0:
            if verbose:
                summary = self.get_status_summary()
                print(f"\nStatus: {summary['active_slots']}/{summary['max_concurrent']} active, "
                      f"{summary['available_slots']} available, {summary['still_to_run']} remaining")
            
            # Submit new batch
            submitted = self.submit_new_batch(verbose=verbose)
            
            if submitted:
                print(f"Submitted {len(submitted)} new workchains")
            else:
                print("No new workchains submitted")
            
            # Wait before next iteration
            time.sleep(sleep_interval)
        
        print("All workchains have been submitted!")


# Convenience function for easy setup
def create_supercon_controller(
    parent_group_label: str,
    group_label: str,
    pw_code: str,
    ph_code: str,
    q2r_code: str,
    matdyn_code: str,
    max_concurrent: int = 5,
    protocol: str = "moderate",
    overrides: Optional[dict] = None,
    **kwargs
) -> SuperConWorkChainController:
    """Create a SuperCon workchain controller with default settings."""
    
    return SuperConWorkChainController(
        parent_group_label=parent_group_label,
        group_label=group_label,
        max_concurrent=max_concurrent,
        pw_code=pw_code,
        ph_code=ph_code,
        q2r_code=q2r_code,
        matdyn_code=matdyn_code,
        protocol=protocol,
        overrides=overrides,
        **kwargs
    ) 