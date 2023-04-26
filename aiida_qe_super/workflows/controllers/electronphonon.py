# -*- coding: utf-8 -*-
"""A SubmissionController for submitting `ElectronPhononWorkChain`s."""
from aiida.orm import StructureData

from aiida_quantumespresso.common.types import SpinType, ElectronicType
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida_submission_controller import FromGroupSubmissionController

from aiida_qe_super.workflows.electronphonon import ElectronPhononWorkChain


class ElectronPhononGroupSubmissionController(FromGroupSubmissionController):
    """A SubmissionController for submitting `ElectronPhononWorkChain`s."""

    @staticmethod
    def get_extra_unique_keys():
        """Return a tuple of the keys of the unique extras that will be used to uniquely identify your workchains."""
        return ("formula_hill", "number_of_sites", "source_db", "source_id")

    # pylint: disable=abstract-method
    def __init__(
        self,
        group_label,
        max_concurrent,
        parent_group_label,
        filters,
        pw_code,
        ph_code,
        q2r_code,
        matdyn_code,
        order_by=None,
        protocol="moderate",
        overrides=None,
        electronic_type=ElectronicType.METAL,
        spin_type=SpinType.NONE,
    ):
        """Create an instance of a ``ElectronPhononGroupSubmissionController``.

        :param group_label: label of the group used to store the work chains. Will be created at instantiation if not
            existing already.
        :param max_concurrent: maximum number of processes that can run concurrently.
        :param parent_group_label: a group label: the group will be used to decide which submissions to use. The group
            must already exist. Extras (in the method `get_all_extras_to_submit`) will be returned from all extras in
            that group (you need to make sure they are unique).
        :param order_by: operation to order the nodes in the ``parent_group`` by before submission.
        :param filters: filters to apply to the nodes in the ``parent_group``, if any.
        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param ph_code: the ``Code`` instance configured for the ``quantumespresso.ph`` plugin.
        :param q2r_code: the ``Code`` instance configured for the ``quantumespresso.a2r`` plugin.
        :param matdyn_code: the ``Code`` instance configured for the ``quantumespresso.matdyn`` plugin.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param electronic_type: indicate the electronic character of the system through ``ElectronicType`` instance.
        :param spin_type: indicate the spin polarization type to use through a ``SpinType`` instance.
        """
        super().__init__(
            parent_group_label=parent_group_label,
            group_label=group_label,
            max_concurrent=max_concurrent,
            order_by=order_by,
            filters=filters,
        )
        self._process_class = ElectronPhononWorkChain
        self._pw_code = pw_code
        self._ph_code = ph_code
        self._q2r_code = q2r_code
        self._matdyn_code = matdyn_code
        self._protocol = protocol
        self._overrides = overrides
        self._electronic_type = electronic_type
        self._spin_type = spin_type

    def get_inputs_and_processclass_from_extras(self, extras_values, dry_run):
        """Return inputs and process class for the submission of this specific process."""
        parent_node = self.get_parent_node_from_extras(extras_values)

        # Depending on the type of node in the parent class, grab the right inputs
        if isinstance(parent_node, StructureData):
            structure = parent_node
        elif parent_node.process_class == PwRelaxWorkChain:
            structure = parent_node.outputs.output_structure
        else:
            raise TypeError(
                f"Node {parent_node} from parent group is of incorrect type: {type(parent_node)}."
            )

        process_class = self._process_class

        builder = process_class.get_builder_from_protocol(
            pw_code=self._pw_code,
            ph_code=self._ph_code,
            q2r_code=self._q2r_code,
            matdyn_code=self._matdyn_code,
            structure=structure,
            protocol=self._protocol,
            overrides=self._overrides,
            electronic_type=self._electronic_type,
            spin_type=self._spin_type,
        )

        return dict(builder), self._process_class
