# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm
from aiida.engine import if_


from .intp import EpwBaseIntpWorkChain

class EpwA2fWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the Allen-Dynes critical temperature."""

    _INTP_NAMESPACE = 'a2f'
    _ALL_NAMESPACES = [
        EpwBaseIntpWorkChain._B2W_NAMESPACE, _INTP_NAMESPACE]

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

        spec.outline(
            cls.setup,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            cls.prepare_process,
            cls.run_process,
            cls.inspect_process,
            cls.results
        )

        spec.inputs[cls._INTP_NAMESPACE].validator = cls.validate_inputs
        spec.inputs.validator = cls.validate_inputs

        # spec.output('a2f', valid_type=orm.XyData,
        #             help='The contents of the `.a2f` file.')
        # spec.output('Tc_allen_dynes', valid_type=orm.Float,
        #             help='The Allen-Dynes Tc interpolated from the a2f file.')

        spec.exit_code(
            402, 'ERROR_SUB_PROCESS_A2F',
            message='The `epw.x` workflow failed.'
            )
    # @classmethod
    # def get_protocol_filepath(cls):
    #     """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
    #     from importlib_resources import files
    #     from . import protocols
    #     return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'

    @classmethod
    def get_builder_restart(
        cls,
        from_a2f_workchain
        ):

        return super()._get_builder_restart(
            from_intp_workchain=from_a2f_workchain,
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

    def prepare_process(self):
        """Prepare the process for the current interpolation distance."""

        super().prepare_process()
        try:
            settings = self.ctx.inputs.epw.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = [
            'aiida.a2f',
            'aiida.a2f_proj',
            'out/aiida.dos',
            'aiida.phdos',
            'aiida.phdos_proj',
            'aiida.lambda_FS',
            'aiida.lambda_k_pairs'
            ]

        self.ctx.inputs.epw.settings = orm.Dict(settings)

    def inspect_process(self):
        """Verify that the epw.x workflow finished successfully."""
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_A2F

    def results(self):
        """TODO"""

        super().results()

        # self.out('a2f', self.ctx.workchain_intp.outputs.a2f)
