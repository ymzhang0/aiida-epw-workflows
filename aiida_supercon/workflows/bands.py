# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm
from aiida.engine import if_


from .intp import EpwBaseIntpWorkChain

from ..common.restart import RestartState


class EpwBandsWorkChain(EpwBaseIntpWorkChain):
    """Work chain to interpolate the band structure using epw.x.
    """

    _INTP_NAMESPACE = 'bands'
    _ALL_NAMESPACES = [EpwBaseIntpWorkChain._B2W_NAMESPACE, _INTP_NAMESPACE]
    _RESTART_STATE = RestartState(_ALL_NAMESPACES)

    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]

    _frozen_bands_parameters = {
        'INPUTEPW': {
            'band_plot': True,
        }
    }
    @classmethod
    def validate_inputs(cls, inputs, ctx=None):
        """Validate the inputs."""
        return None

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)


        spec.inputs[cls._INTP_NAMESPACE].validator = cls.validate_inputs
        spec.inputs.validator = cls.validate_inputs

        spec.input(
            'bands_kpoints_distance', valid_type=orm.Float, required=False,
            help='The distance between the kpoints in the band structure.')

        spec.output(
            "seekpath_parameters",
            valid_type=orm.Dict,
            required=False,
            help="The parameters used in the SeeKpath call to normalize the input or relaxed structure.",
        )

        spec.exit_code(
            402, 'ERROR_SUB_PROCESS_BANDS',
            message='The `epw.x` workflow failed.'
            )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'

    @classmethod
    def get_builder_restart(
        cls,
        from_bands_workchain
        ):

        return super()._get_builder_restart(
            from_intp_workchain=from_bands_workchain,
            )

    @classmethod
    def get_builder_from_protocol(
            cls,
            codes,
            structure,
            protocol=None,
            overrides=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        builder = super().get_builder_from_protocol(
            codes,
            structure,
            protocol,
            overrides,
            **kwargs
            )

        return builder

    def setup(self):
        """Setup the work chain."""
        super().setup()

        from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis

        inputs = {
            "structure": self.inputs.structure,
            'metadata': {
                'call_link_label': 'seekpath'
            }
        }

        if 'bands_kpoints_distance' in self.inputs:
            inputs['reference_distance'] = self.inputs.bands_kpoints_distance

        result = seekpath_structure_analysis(**inputs)
        self.ctx.bands_kpoints = result['explicit_kpoints']

        self.out('seekpath_parameters', result['parameters'])

    def prepare_process(self):
        """Prepare the process."""
        super().prepare_process()

        self.ctx.inputs.pop('qfpoints_distance')
        self.ctx.inputs.pop('kfpoints_factor')

        self.ctx.inputs.qfpoints = self.ctx.bands_kpoints
        self.ctx.inputs.kfpoints = self.ctx.bands_kpoints

        parameters = self.ctx.inputs.epw.parameters.get_dict()

        for namespace, _parameters in self._frozen_bands_parameters.items():
            for keyword, value in _parameters.items():
                parameters[namespace][keyword] = value

        self.ctx.inputs.epw.parameters = orm.Dict(parameters)

    def inspect_process(self):
        """Verify that the epw.x workflow finished successfully."""
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_BANDS

