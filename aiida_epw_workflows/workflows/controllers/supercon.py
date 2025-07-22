from aiida import orm
from typing import Optional
from pydantic import ConfigDict

from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_epw_workflows.workflows import EpwB2WWorkChain, EpwSuperConWorkChain
from aiida_epw_workflows.tools.ph import get_negative_frequencies, get_phonon_wc_from_epw_wc

from aiida_submission_controller import FromGroupSubmissionController

from pydantic import ConfigDict


class EpwSuperConWorkChainController(FromGroupSubmissionController):
    """A SubmissionController for submitting `ElectronPhononWorkChain`s."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


    codes: dict = {}
    protocol: str = 'moderate'
    overrides: Optional[dict] = None
    wannier_projection_type: WannierProjectionType = WannierProjectionType.ATOMIC_PROJECTORS_QE
    reference_bands: orm.BandsData = None
    bands_kpoints: orm.KpointsData = None
    w90_chk_to_ukk_script: orm.RemoteData = None

    _process_class = EpwSuperConWorkChain

    def get_inputs_and_processclass_from_extras(self, extras_values, dry_run=False):
        """Return inputs and process class for the submission of this specific process."""
        parent_node = self.get_parent_node_from_extras(extras_values)
        process_class = self._process_class

        # Depending on the type of node in the parent class, grab the right inputs
        if isinstance(parent_node, orm.StructureData):
            structure = parent_node
            builder = process_class.get_builder_from_protocol(
                structure=structure,
                codes = self.codes,
                protocol=self.protocol,
                overrides=self.overrides,
                wannier_projection_type=self.wannier_projection_type,
                reference_bands=self.reference_bands,
                bands_kpoints=self.bands_kpoints,
                w90_chk_to_ukk_script=self.w90_chk_to_ukk_script,
            )
        elif parent_node.process_class == PhBaseWorkChain:

            is_stable, negative_freqs = get_negative_frequencies(parent_node)
            if not is_stable:
                raise ValueError(f"Phonon workchain {parent_node.pk} is unstable.")

            builder = process_class.get_builder_restart_from_ph_base(
                parent_node,
                codes = self.codes,
                protocol=self.protocol,
                overrides=self.overrides,
                wannier_projection_type=self.wannier_projection_type,
                reference_bands=self.reference_bands,
                bands_kpoints=self.bands_kpoints,
                w90_chk_to_ukk_script=self.w90_chk_to_ukk_script,
            )
        elif parent_node.process_label == 'EpwWorkChain':
            try:
                ph_wc = get_phonon_wc_from_epw_wc(parent_node)
                is_stable, negative_freqs = get_negative_frequencies(ph_wc)
                if not is_stable:
                    raise ValueError(f"Phonon workchain {ph_wc.pk} is unstable.")
            except Exception as e:
                raise ValueError(f"Failed to get phonon workchain from EpwWorkChain {parent_node.pk}: {e}")

            builder = process_class.get_builder_restart_from_ph_base(
                ph_wc,
                codes = self.codes,
                protocol=self.protocol,
                overrides=self.overrides,
                wannier_projection_type=self.wannier_projection_type,
                reference_bands=self.reference_bands,
                bands_kpoints=self.bands_kpoints,
                w90_chk_to_ukk_script=self.w90_chk_to_ukk_script,
            )

        elif parent_node.process_class == EpwB2WWorkChain:
            builder = process_class.get_builder_restart_from_b2w(
                parent_node,
                protocol=self.protocol,
                overrides=self.overrides,
            )
        elif parent_node.process_class == EpwSuperConWorkChain:
            builder = process_class.get_builder_restart(
                parent_node,
            )
        else:
            raise TypeError(
                f"Node {parent_node} from parent group is of incorrect type: {type(parent_node)}."
            )


        return builder


