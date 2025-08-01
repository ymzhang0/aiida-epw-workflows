# -*- coding: utf-8 -*-
from aiida import orm
from aiida.engine import process_handler, ProcessHandlerReport

from .intp import EpwBaseIntpWorkChain


class EpwAnisoWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the superconductivity based on anisotropic Migdal-Eliashberg theory.
    This workchain aims to implement the IR representation which allows calculation on low temperatures.

    However, this version of epw.x made several changes that are not compatible with the previous versions.

    1.  It fix the typo in previous versions, that is, `eps_acustic` -> `eps_acoustic`.

    2.  The format of crystal.fmt file is changed. Previously it is:
            nat
            nmode
            nelec
            ...
        And now it is:
            nat
            nmode
            nelec   nbndskp
            ...

    4.  The `epw.x` from EPW 5.9 will try to read 'vmedata.fmt' even if
        I set vme = 'dipole'.

    5.  When I use the preinstalled version of EPW v5.9, it can't open the freq file generated by the old version.

    6.  When I use the latest version to generate .epmatwp file. It has problem with opening the .ukk file generated from Wannier90
        It turns out that the latest version change the definition of the first line of the file.
        (nbndstart, nbnd) -> (nbndep, nbndskip) and after the first line, it will write the number of bands.
        1
        2
        3
        ...
        nbndep

    7.  The latest version of EPW requires .bvec, .mmn file when using vme='dipole'.
        Remember to delete the comment lines in the .mmn file.
    8.  Now the code will output dmedata.fmt and vmedata.fmt files even if vme='dipole'.
        Remember to save them when using new version of EPW.
    9.  I get lambda ~ 0 even if I fix all the problems above.
    10. The latest version of EPW change the format of the prefix.ephmat/freq file.
        It no longer does iq = select(iqq). It write frequcies of all q points instead.
        Simply rollback to version 9d42e9b4530f58543e33dbfce6268f11705ff01d
        I would suggest to start only from the existing prefix.ephmat folder.
        But this requires another input code. I would temporarily mute use_ir.
    """

    _INTP_NAMESPACE = 'aniso'

    _ALL_NAMESPACES = [EpwBaseIntpWorkChain._B2W_NAMESPACE, _INTP_NAMESPACE]

    _forced_parameters =  EpwBaseIntpWorkChain._forced_parameters.copy()
    _forced_parameters['INPUTEPW']  = EpwBaseIntpWorkChain._forced_parameters['INPUTEPW'] | {
          'eliashberg': True,
          'ephwrite': True,
          'laniso': True,
          'limag': True,
          'lpade': True,
        }

    _frozen_restart_parameters = {
        'INPUTEPW': {
            'elph': False,
            'ep_coupling': False,
            'epwread': True,
            'epwwrite': False,
            'ephwrite': False,
            'restart': True,
        },
    }

    _frozen_plot_gap_function_parameters = {
        'INPUTEPW': {
            'iverbosity': 2,
        }
    }

    _frozen_fbw_parameters = {
        'INPUTEPW': {
            'fbw': True,
        }
    }

    _DEFAULT_FILIROBJ = "ir_nlambda6_ndigit8.dat"
    _frozen_ir_parameters = {
        'INPUTEPW': {
            'fbw': True,
            'muchem': True,
            'gridsamp': 2,
            'broyden_beta': -0.7,
            # 'filirobj': './' + _DEFAULT_FILIROBJ,
        }
    }

    _MIN_TEMP = 3.5
    _MIN_TEMP_IR = 1.0

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('plot_gap_function', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='Whether to plot the gap function.')
        spec.input('estimated_Tc_aniso', valid_type=orm.Float, default=lambda: orm.Float(40.0),
            help='The estimated Tc for the aniso calculation.')
        spec.input('fbw', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='Whether to use the full bandwidth.')
        spec.input('use_ir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='Whether to use the intermediate representation.')

        # spec.output(
        #     'Tc_aniso', valid_type=orm.Float,
        #     help='The anisotropic Tc from fitting the gap function.')

        spec.exit_code(402, 'ERROR_SUB_PROCESS_ANISO',
            message='The `aniso` sub process failed')
        spec.exit_code(403, 'ERROR_TEMPERATURE_OUT_OF_RANGE',
            message='The `aniso` calculation have less than two temperatures within aniso Tc ')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'

    @classmethod
    def validate_inputs(cls, value, port_namespace):  # pylint: disable=unused-argument
        """Validate the top level namespace."""

        if not ('parent_epw_folder' in port_namespace or 'epw' in port_namespace):
            return "Only one of `parent_epw_folder` or `epw` can be accepted."

        return None

    @classmethod
    def get_builder_restart(
        cls,
        from_aniso_workchain
        ):
        """Get a builder for a restart from an anisotropic calculation.

        :param from_aniso_workchain: The anisotropic workchain from which to restart.
        :type from_aniso_workchain: :class:`aiida.orm.Node`
        :return: A builder instance with the inputs prepopulated.
        :rtype: :class:`aiida.engine.ProcessBuilder`
        """
        return super()._get_builder_restart(
            from_intp_workchain=from_aniso_workchain,
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
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param codes: The codes to use for the calculations.
        :type codes: dict
        :param structure: The structure to use for the calculations.
        :type structure: :class:`aiida.orm.StructureData`
        :param protocol: The protocol to use for the calculations.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        use_ir = inputs.get('use_ir', False)
        builder = super().get_builder_from_protocol(
            codes,
            structure,
            protocol,
            overrides,
            **kwargs
        )

        builder.plot_gap_function = orm.Bool(inputs.get('plot_gap_function', True))
        builder.fbw = orm.Bool(inputs.get('fbw', False))
        builder.use_ir = orm.Bool(use_ir)

        return builder

    def prepare_process(self):
        """Prepare the process for the current interpolation distance.

        It will set the necessary inputs parameters for an anisotropic calculation.
        It will set the necessary retrieve items from an anisotropic calculation:
        - a2f
        - a2f_proj
        - dos
        - phdos
        - lambda_k_pairs
        - lambda_FS
        - imag_aniso_gap0_*
        - pade_aniso_gap0_*
        """
        super().prepare_process()

        parameters = self.ctx.inputs.epw.parameters.get_dict()

        min_temp = self._MIN_TEMP

        try:
            settings = self.ctx.inputs.epw.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = [
            'aiida.lambda_k_pairs', 'aiida.lambda_FS',
            ]

        if self.inputs.plot_gap_function.value:
            for namespace, _parameters in self._frozen_plot_gap_function_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value
            settings['ADDITIONAL_RETRIEVE_LIST'].extend([
                'aiida.imag_aniso_gap0_*.frmsf',
                'aiida.lambda.frmsf',
                ])

        if self.inputs.fbw.value:
            for namespace, _parameters in self._frozen_fbw_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value

        if self.inputs.use_ir.value:
            for namespace, _parameters in self._frozen_ir_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value

            filirobj = self.ctx.inputs.epw.code.filepath_executable.parent.parent / 'EPW' / 'irobjs' / self._DEFAULT_FILIROBJ

            parameters['INPUTEPW']['filirobj'] = str(filirobj)

            min_temp = self._MIN_TEMP_IR

            # EPW 5.9 (IR representation) can't recognize eps_acustic.

            # eps_acustic = parameters['INPUTEPW'].pop('eps_acustic', 0.1)
            # parameters['INPUTEPW']['eps_acoustic'] = eps_acustic

        temps = f'{min_temp} {self.inputs.estimated_Tc_aniso.value*1.5}'
        nstemp = max(5, int((self.inputs.estimated_Tc_aniso.value*1.5 - min_temp) // 2) + 1)

        parameters['INPUTEPW']['temps'] = temps
        parameters['INPUTEPW']['nstemp'] = nstemp

        self.ctx.inputs.epw.settings = orm.Dict(settings)
        self.ctx.inputs.epw.parameters = orm.Dict(parameters)

    def inspect_process(self):
        """Verify that the `EpwBaseWorkChain` finished successfully.
        It will calculate the anisotropic Tc from the `EpwBaseWorkChain` outputs.
        """
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ANISO

        if False:
            return self.handle_temperature_out_of_range(aniso)


    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report(f'Action taken: {action}')

    @process_handler(priority=403,)
    def handle_temperature_out_of_range(self, calculation):
        """Handle calculations with an exit status below 400 which are unrecoverable, so abort the work chain."""
        if calculation.exit_status == self.exit_codes.ERROR_TEMPERATURE_OUT_OF_RANGE:
            self.report_error_handled(calculation, 'unrecoverable error, aborting...')
            return ProcessHandlerReport(True, self.exit_codes.ERROR_TEMPERATURE_OUT_OF_RANGE)
