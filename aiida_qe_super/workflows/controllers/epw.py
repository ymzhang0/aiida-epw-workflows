from aiida import orm
import copy
from typing import Optional

from aiida_quantumespresso.common.types import SpinType, ElectronicType
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida_submission_controller import FromGroupSubmissionController

from aiida_qe_super.workflows.epw import EpwWorkChain


class EpwWorkChainController(FromGroupSubmissionController):
    """A SubmissionController for submitting `ElectronPhononWorkChain`s."""
    pw_code: str
    ph_code: str
    projwfc_code: str
    pw2wannier90_code: str
    wannier90_code: str
    epw_code: str
    protocol = "moderate"
    overrides: Optional[dict] = None
    electronic_type: ElectronicType = ElectronicType.METAL,
    spin_type: SpinType = SpinType.NONE,

    _process_class = EpwWorkChain

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
        elif parent_node.process_class == PwRelaxWorkChain:
            structure = parent_node.outputs.output_structure
        else:
            raise TypeError(
                f"Node {parent_node} from parent group is of incorrect type: {type(parent_node)}."
            )

        process_class = self._process_class
        codes = {
            "pw": self.pw_code,
            "ph": self.ph_code,
            "projwfc": self.projwfc_code,
            "pw2wannier90": self.pw2wannier90_code,
            "wannier90": self.wannier90_code,
            "epw": self.epw_code
        }
        codes = {key: orm.load_code(code_label) for key, code_label in codes.items()}

        builder = process_class.get_builder_from_protocol(
            codes=codes,
            structure=structure,
            protocol=self.protocol,
            overrides=copy.deepcopy(self.overrides),
            electronic_type=self.electronic_type,
            spin_type=self.spin_type,
        )
        # HARDCODED CRAP FOR NOW
        options = {
            'max_wallclock_seconds': 1800,
            'account': 'mr32',
            # 'queue_name': 'debug'
        }
        pp_resources = {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 8,
            'num_cores_per_machine': 8,
        }
        builder.w90_bands.scf.pw.metadata.options.update(options)
        builder.w90_bands.scf.pw.parallelization = orm.Dict({'npool': 8})
        builder.w90_bands.nscf.pw.metadata.options.update(options)
        builder.w90_bands.nscf.pw.parallelization = orm.Dict({'npool': 8})
        # builder.open_grid.open_grid.metadata.options.update(options)
        builder.w90_bands.projwfc.projwfc.metadata.options.update(options)
        builder.w90_bands.projwfc.projwfc.metadata.options['resources'] = pp_resources
        builder.w90_bands.pw2wannier90.pw2wannier90.metadata.options.update(options)
        builder.w90_bands.pw2wannier90.pw2wannier90.metadata.options['resources'] = pp_resources
        builder.w90_bands.wannier90.wannier90.metadata.options.update(options)

        builder.clean_workdir = orm.Bool(False)
        builder.kpoints_factor = orm.Int(2)

        w90_script = orm.RemoteData(
            remote_path='/users/mbercx/code/epw_julia/chk2ukk.jl',
            computer=orm.load_computer('eiger')
        )
        builder.w90_chk_to_ukk_script = w90_script

        return builder
