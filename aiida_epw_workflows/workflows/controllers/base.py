from aiida import orm
import copy
from typing import Optional

from aiida_quantumespresso.common.types import SpinType, ElectronicType
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida_submission_controller import FromGroupSubmissionController

from aiida_epw_workflows.workflows.base import EpwBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida.plugins import WorkflowFactory

class EpwBaseWorkChainController(FromGroupSubmissionController):
    """A SubmissionController for submitting `ElectronPhononWorkChain`s."""
    codes = None
    protocol = None
    overrides: Optional[dict] = None
    electronic_type: ElectronicType = ElectronicType.METAL,
    spin_type: SpinType = SpinType.NONE,

    _process_class = EpwBaseWorkChain

    PhBaseWorkChain = WorkflowFactory('quantumespresso.ph.base')

    @staticmethod
    def get_extra_unique_keys():
        """Return a tuple of the keys of the unique extras that will be used to uniquely identify your workchains."""
        return ("formula_hill", "number_of_sites", "source_db", "source_id")

    @staticmethod
    def find_latest_successful_workchain(structure: orm.StructureData, process_class) -> orm.WorkChainNode:
        """
        Queries the database to find the latest successfully completed workchain of a given type
        that has the given structure as an input.

        :param structure: The structure to search for.
        :param process_class: The WorkChain class to search for (e.g., EpwB2WWorkChain).
        :return: The latest finished_ok node, or None if not found.
        """
        qb = orm.QueryBuilder()
        qb.append(orm.StructureData, filters={'id': structure.pk}, tag='structure')
        qb.append(
            process_class,
            with_incoming='structure', # Find workchains that have this structure as an input
            filters={'attributes.exit_status': 0}, # Filter for successfully completed ones
            tag='wc'
        )
        # Order by creation time to get the latest one first
        qb.order_by([{'wc': {'ctime': 'desc'}}])
        qb.project(['wc', '*']) # We want the full node object

        result = qb.first()

        if result:
            return result[1] # The query returns a list [tag, object], we want the object

        return None

    # This method will automatically generate the builder based on
    # the previous workchains that we have queried from the database

    def get_inputs_and_processclass_from_extras(
        self, extras_values, dry_run=False
        ):
        """Return inputs and process class for the submission of this specific process."""
        parent_node = self.get_parent_node_from_extras(extras_values)

        logger = self.get_logger()
        logger.report(f"Processing <{parent_node.pk}>...")

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

        # codes = {key: orm.load_code(code_label) for key, code_label in codes.items()}

        overrides = copy.deepcopy(self.overrides)

        w90_scf_overrides = overrides.get('w90_bands', {}).pop('scf', {})
        w90_nscf_overrides = overrides.get('w90_bands', {}).pop('nscf', {})

        builder = process_class.get_builder_from_protocol(
            codes=self.codes,
            structure=structure,
            protocol=self.protocol,
            overrides=overrides,
            electronic_type=self.electronic_type,
            spin_type=self.spin_type,
        )
        # HARDCODED CRAP FOR NOW
        options = {
            'max_wallclock_seconds': 1800,
            'account': 'project_465000106',
            'queue_name': 'debug'
        }
        pp_resources = {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 8,
            'num_cores_per_machine': 8,
        }
        # MORE CRAP SINCE JUNFENG's work chains don't work with overrides
        scf_params = builder.w90_bands.scf.pw.parameters.get_dict()
        scf_params = recursive_merge(
            scf_params,
            w90_scf_overrides.get('pw', {}).get('parameters')
        )
        builder.w90_bands.scf.pw.parameters = orm.Dict(scf_params)
        nscf_params = builder.w90_bands.nscf.pw.parameters.get_dict()
        nscf_params = recursive_merge(
            nscf_params,
            w90_nscf_overrides.get('pw', {}).get('parameters')
        )
        builder.w90_bands.nscf.pw.parameters = orm.Dict(nscf_params)

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

        builder.clean_workdir = orm.Bool(True)
        builder.kpoints_factor = orm.Int(2)

        # w90_script = orm.RemoteData(
        #     remote_path='/users/mbercx/code/epw_julia/chk2ukk.jl',
        #     computer=orm.load_computer('eiger')
        # )

        builder.w90_chk_to_ukk_script = self.codes['wannier90']

        return builder
