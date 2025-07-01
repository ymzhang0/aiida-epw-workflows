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

    @classmethod
    def validate_inputs(cls, inputs, ctx=None):
        """Validate the inputs."""
        return None

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        # spec.outline(
        #     cls.setup,
        #     if_(cls.should_run_b2w)(
        #         cls.run_b2w,
        #         cls.inspect_b2w,
        #     ),
        #     cls.prepare_process,
        #     cls.run_process,
        #     cls.inspect_process,
        #     cls.results
        # )

        spec.inputs[cls._INTP_NAMESPACE].validator = cls.validate_inputs
        spec.inputs.validator = cls.validate_inputs

        spec.output('el_band_structure', valid_type=orm.BandsData,
                    help='The electronic band structure.')
        spec.output('ph_band_structure', valid_type=orm.BandsData,
                    help='The phonon band structure.')

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
            'reference_distance': self.inputs.get('bands_kpoints_distance', None),
            'metadata': {
                'call_link_label': 'seekpath'
            }
        }
        result = seekpath_structure_analysis(self.inputs.structure, **inputs)
        self.ctx.bands_kpoints = result['explicit_kpoints']

        self.out('seekpath_parameters', result['parameters'])

    def prepare_process(self):
        """Prepare the process."""
        super().prepare_process()

        self.ctx.inputs.pop('qfpoints_distance')
        self.ctx.inputs.pop('kfpoints_factor')

        self.ctx.inputs.qfpoints = self.ctx.bands_kpoints
        self.ctx.inputs.kfpoints = self.ctx.bands_kpoints

    def inspect_process(self):
        """Verify that the epw.x workflow finished successfully."""
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_BANDS

    def results(self):
        """TODO"""

        super().results()

        self.out('el_band_structure', self.ctx.workchain_intp.outputs.el_band_structure)
        self.out('ph_band_structure', self.ctx.workchain_intp.outputs.ph_band_structure)
