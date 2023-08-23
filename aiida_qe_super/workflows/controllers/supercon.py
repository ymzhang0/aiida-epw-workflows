from aiida import orm
import copy
from typing import Optional

from aiida_quantumespresso.common.types import SpinType, ElectronicType
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida_submission_controller import FromGroupSubmissionController

from aiida_qe_super.workflows.epw import EpwWorkChain
from aiida_qe_super.workflows.supercon import SuperConWorkChain


class SuperConWorkChainController(FromGroupSubmissionController):
    """A SubmissionController for submitting `ElectronPhononWorkChain`s."""
    epw_code: str
    protocol = "moderate"
    overrides: Optional[dict] = None

    _process_class = SuperConWorkChain

    @staticmethod
    def get_extra_unique_keys():
        """Return a tuple of the keys of the unique extras that will be used to uniquely identify your workchains."""
        return ("formula_hill", "number_of_sites", "source_db", "source_id")

    def get_inputs_and_processclass_from_extras(self, extras_values, dry_run=False):
        """Return inputs and process class for the submission of this specific process."""
        parent_node = self.get_parent_node_from_extras(extras_values)

        builder = self._process_class.get_builder_from_protocol(
            epw_code=orm.load_code(self.epw_code),
            parent_epw=parent_node,
            overrides=copy.deepcopy(self.overrides)
        )
        return builder
