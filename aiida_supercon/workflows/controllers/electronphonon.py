# -*- coding: utf-8 -*-
"""A SubmissionController for submitting `ElectronPhononWorkChain`s."""
from typing import Optional, Union
from aiida import orm

from aiida_quantumespresso.common.types import SpinType, ElectronicType
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida_submission_controller import FromGroupSubmissionController

from aiida_qe_super.workflows.electronphonon import ElectronPhononWorkChain


class ElectronPhononGroupSubmissionController(FromGroupSubmissionController):
    """A SubmissionController for submitting `ElectronPhononWorkChain`s."""
    pw_code: Union[int, str]
    ph_code: Union[int, str]
    q2r_code: Union[int, str]
    matdyn_code: Union[int, str]
    protocol = "moderate"
    overrides: Optional[dict] = None
    electronic_type: ElectronicType = ElectronicType.METAL,
    spin_type: SpinType = SpinType.NONE,

    _process_class = ElectronPhononWorkChain

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

        builder = process_class.get_builder_from_protocol(
            pw_code=orm.load_code(self.pw_code),
            ph_code=orm.load_code(self.ph_code),
            q2r_code=orm.load_code(self.q2r_code),
            matdyn_code=orm.load_code(self.matdyn_code),
            structure=structure,
            protocol=self.protocol,
            overrides=self.overrides,
            electronic_type=self.electronic_type,
            spin_type=self.spin_type,
        )

        return dict(builder), self._process_class
