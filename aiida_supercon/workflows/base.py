# -*- coding: utf-8 -*-
from aiida import orm
from aiida.common import AttributeDict, NotExistent

from aiida.engine import BaseRestartWorkChain, ProcessHandlerReport, process_handler, while_
from aiida.plugins import CalculationFactory

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_quantumespresso.calculations.ph import PhCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation

from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain, Wannier90OptimizeWorkChain


from pathlib import Path

EpwCalculation = CalculationFactory('epw.epw')


class EpwBaseWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """BaseWorkchain to run a epw.x calculation."""

    # pylint: disable=too-many-public-methods, too-many-statements

    _process_class = EpwCalculation


    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)

        # EpwBaseWorkChain will take over the determination of coarse grid according to the parent folders.
        # It will automatically generate fine grid k/q points according to qpoints distance and kfpoints factor.
        spec.expose_inputs(
            EpwCalculation, namespace='epw',
            exclude=(
                'kpoints',
                'qpoints',
                'kfpoints',
                'qfpoints',
                'parent_folder_nscf',
                'parent_folder_ph',
                'parent_folder_epw',
            )
        )

        spec.input(
            'structure', valid_type=orm.StructureData,
            help='The structure to compute the epw for fine grid generation'
            )

        spec.input(
            'qfpoints', valid_type=orm.KpointsData, required=False,
            help='fine qpoint mesh'
            )

        spec.input(
            'qfpoints_distance', valid_type=orm.Float, required=False,
            help='fine qpoint mesh distance'
            )

        spec.input(
            'kfpoints', valid_type=orm.KpointsData, required=False,
            help='fine kpoint mesh'
            )

        spec.input(
            'kfpoints_factor', valid_type=orm.Int, required=False,
            help='fine kpoint mesh factor'
            )

        spec.input(
            'parent_folder_nscf', valid_type=orm.RemoteData, required=False,
            help='parent folder of the nscf calculation'
            )

        spec.input(
            'parent_folder_ph', valid_type=(orm.RemoteData, orm.RemoteStashFolderData), required=False,
            help='parent folder of the ph calculation'
            )

        spec.input(
            'parent_folder_epw', valid_type=(orm.RemoteData, orm.RemoteStashFolderData), required=False,
            help='parent folder of the epw calculation'
            )

        spec.input(
            'parent_folder_chk', valid_type=orm.RemoteData, required=False,
            help='parent folder of the chk file'
            )

        spec.input(
            'w90_chk_to_ukk_script', valid_type=orm.RemoteData, required=False,
            help='w90_chk_to_ukk_script'
            )

        spec.outline(
            cls.setup,
            cls.validate_parent_folders,
            cls.validate_parallelization,
            cls.validate_kpoints,
            while_(cls.should_run_process)(
                cls.prepare_process,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.expose_outputs(
            EpwCalculation,
            )


        spec.exit_code(300, 'ERROR_UNRECOVERABLE_FAILURE',
            message='The calculation failed with an unidentified unrecoverable error.')
        spec.exit_code(310, 'ERROR_KNOWN_UNRECOVERABLE_FAILURE',
            message='The calculation failed with a known unrecoverable error.')


    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols
        return files(protocols) / 'base.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        protocol=None,
        overrides=None,
        parent_folder_nscf=None,
        parent_folder_ph=None,
        parent_folder_chk=None,
        parent_folder_epw=None,
        w90_chk_to_ukk_script=None,
        **_
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.epw`` plugin.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param parent_folder_nscf: parent folder of the nscf calculation
        :param parent_folder_ph: parent folder of the ph calculation
        :param parent_folder_chk: parent folder of the chk calculation
        :param parent_folder_epw: parent folder of the epw calculation
        :param w90_chk_to_ukk_script: a julia script to convert the prefix.chk file (generated by wannier90.x) to a prefix.ukk file (to be used by epw.x)
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from .utils.overrides import recursive_copy

        if isinstance(code, str):
            code = orm.load_code(code)

        inputs = cls.get_protocol_inputs(protocol, overrides)

        # Update the parameters based on the protocol inputs
        parameters = inputs['epw']['parameters']

        # If overrides are provided, they are considered absolute
        if overrides:
            parameter_overrides = overrides.get('epw', {}).get('parameters', {})
            parameters = recursive_copy(parameters, parameter_overrides)

        metadata = inputs['epw']['metadata']


        # pylint: disable=no-member
        builder = cls.get_builder()
        builder.structure = structure
        builder.epw['code'] = code
        builder.epw['parameters'] = orm.Dict(parameters)
        builder.epw['metadata'] = metadata

        if parent_folder_nscf:
            builder.parent_folder_nscf = parent_folder_nscf
        if parent_folder_ph:
            builder.parent_folder_ph = parent_folder_ph
        if parent_folder_chk:
            builder.parent_folder_chk = parent_folder_chk
        if parent_folder_epw:
            builder.parent_folder_epw = parent_folder_epw
        if w90_chk_to_ukk_script:
            builder.w90_chk_to_ukk_script = w90_chk_to_ukk_script

        if 'settings' in inputs['epw']:
            builder.epw['settings'] = orm.Dict(inputs['epw']['settings'])
        if 'parallelization' in inputs['epw']:
            builder.epw['parallelization'] = orm.Dict(inputs['epw']['parallelization'])

        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        if 'qfpoints' in inputs:
            builder.qfpoints = inputs['qfpoints']
        else:
            builder.qfpoints_distance = orm.Float(inputs['qfpoints_distance'])

        builder.kfpoints_factor = orm.Int(inputs['kfpoints_factor'])

        builder.max_iterations = orm.Int(inputs['max_iterations'])
        # pylint: enable=no-member

        return builder

    def setup(self):
        """Call the ``setup`` of the ``BaseRestartWorkChain`` and create the inputs dictionary in ``self.ctx.inputs``.

        This ``self.ctx.inputs`` dictionary will be used by the ``EpwCalculation`` in the internal loop.

        """
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(EpwCalculation, 'epw'))

    def _get_kpoints_from_nscf_folder(self, nscf_folder):
        """
        This method tries different strategies to find the k-point mesh from a parent nscf folder.

        :param nscf_folder: A RemoteData node from a PwCalculation (nscf).
        :return: A KpointsData node that has mesh information.
        :raises ValueError: If the mesh cannot be found through any strategy.
        """
        pw_calc = nscf_folder.creator

        if not pw_calc.process_class is PwCalculation:
            raise ValueError('Parent folder of nscf calculation is not a valid nscf calculation.')

        w90_workchain = pw_calc.caller.caller
        if not w90_workchain:
            # If there's no caller, it must be a standalone PwBaseWorkChain run.
            # We can either get the kpoints mesh from the inputs.
            # Or deduce the mesh from the coordinates if the input KpointsData is not a mesh but a list of explicit k-points.
            try:
                kpoints = pw_calc.inputs.kpoints
                mesh, _ = kpoints.get_kpoints_mesh()
                return kpoints
            except (AttributeError, ValueError) as e:
                raise ValueError(f"Could not deduce mesh from coordinates. Reason: {e}") from e

        # Check if the caller is a Wannier90 workchain and look for `mp_grid`.
        if w90_workchain.process_class is Wannier90OptimizeWorkChain:
            if hasattr(w90_workchain.inputs, 'optimize_disproj') and w90_workchain.inputs.optimize_disproj:
                wannier_params = w90_workchain.inputs.wannier90_optimal.wannier90.get('parameters')
            else:
                wannier_params = w90_workchain.inputs.wannier90.wannier90.get('parameters')
        elif w90_workchain.process_class is Wannier90BandsWorkChain:
            wannier_params = w90_workchain.inputs.wannier.parameters.get_dict()
        else:
            # If the caller is not a Wannier90 workchain, we use the fallback strategy.
            return self._deduce_mesh_from_explicit_kpoints(pw_calc.inputs.kpoints)

        if 'mp_grid' in wannier_params:
            mp_grid = wannier_params['mp_grid']
            kpoints = orm.KpointsData()
            kpoints.set_kpoints_mesh(mp_grid)
            return kpoints
        else:
            return self._deduce_mesh_from_explicit_kpoints(pw_calc.inputs.kpoints)

    def _deduce_mesh_from_explicit_kpoints(self, kpoints_node):
        """The fallback strategy: deduce mesh from a list of k-points."""
        try:
            import numpy as np
            explicit_kpoints = kpoints_node.get_kpoints()
            nk1 = len(np.unique(explicit_kpoints[:, 0].round(decimals=6)))
            nk2 = len(np.unique(explicit_kpoints[:, 1].round(decimals=6)))
            nk3 = len(np.unique(explicit_kpoints[:, 2].round(decimals=6)))
            if len(explicit_kpoints) != nk1 * nk2 * nk3:
                raise ValueError("Product of deduced dimensions does not match k-point count.")
            mesh = [nk1, nk2, nk3]
            kpoints = orm.KpointsData()
            kpoints.set_kpoints_mesh(mesh)
            return kpoints
        except (AttributeError, ValueError) as e:
            raise ValueError(f"Could not deduce mesh from coordinates. Reason: {e}") from e

    def _get_qpoints_from_ph_folder(self, ph_folder):
        """
        This method tries to find the q-point mesh from a parent ph folder.
        It assumes that the q-points is a required input for `PhCalculation`.

        :param ph_folder: A RemoteData node from a PhCalculation.
        :return: A KpointsData node that has mesh information.
        :raises ValueError: If the mesh cannot be found through any strategy.
        """

        ph_calc = ph_folder.creator
        if not ph_calc.process_class is PhCalculation:
            raise ValueError('Parent folder of ph calculation is not a valid ph calculation.')

        qpoints = ph_calc.inputs.qpoints
        return qpoints

    def _get_coarse_grid_from_epw_folder(self, epw_folder):
        """
        Get the coarse grid from the epw folder. Usually used for restart purposes.
        It assumes that the k-points and q-points are required inputs for `EpwCalculation`.
        """
        epw_calc = epw_folder.creator
        # if not epw_calc.process_class is EpwBaseWorkChain._process_class:
        if not epw_calc.process_label == 'EpwCalculation':
            raise ValueError('Parent folder of epw calculation is not a valid epw calculation.')

        kpoints = epw_calc.inputs.kpoints
        qpoints = epw_calc.inputs.qpoints

        return kpoints, qpoints

    def validate_parent_folders(self):
        """Validate the parent folders.
        If the parent_folder_epw is provided, use it to get the coarse grid. And no other parent folders should be provided.
        If the parent_folder_epw is not provided, load the other parent folders, check if they are cleaned and get the coarse grid from them.
        """

        # Check if parent_folder_epw is provided
        has_parent_folder_epw = 'parent_folder_epw' in self.inputs

        # Check which of the other three folders are provided
        has_parent_folder_nscf = 'parent_folder_nscf' in self.inputs
        has_parent_folder_ph = 'parent_folder_ph' in self.inputs
        has_parent_folder_chk = 'parent_folder_chk' in self.inputs

        # Rule: Cannot provide `parent_folder_epw` AND any of the other three.
        if has_parent_folder_epw and any([
            has_parent_folder_nscf,
            has_parent_folder_ph,
            has_parent_folder_chk
            ]
        ):
            raise ValueError("You cannot provide `parent_folder_epw` at the same time as the `nscf/ph/chk` folders.")

        if 'parent_folder_epw' in self.inputs:
            kpoints, qpoints = self._get_coarse_grid_from_epw_folder(self.inputs.parent_folder_epw)
            self.ctx.inputs.kpoints = kpoints
            self.ctx.inputs.qpoints = qpoints
            self.ctx.inputs.parent_folder_epw = self.inputs.parent_folder_epw
            return

        else:
            if self.inputs.parent_folder_nscf.is_cleaned:
                raise ValueError('Parent folder of nscf calculation is cleaned. Skipping k-point mesh search.')

            try:
                kpoints = self._get_kpoints_from_nscf_folder(self.inputs.parent_folder_nscf)
                self.ctx.inputs.kpoints = kpoints
                self.report(f"Successfully determined k-point mesh: {kpoints.get_kpoints_mesh()[0]}")
            except (ValueError, NotExistent) as e:
                raise ValueError(f"Could not determine the k-points mesh: {e}")

            self.ctx.inputs.parent_folder_nscf = self.inputs.parent_folder_nscf

            if self.inputs.parent_folder_ph.is_cleaned:
                raise ValueError("Parent folder of ph calculation is cleaned.")

            try:
                qpoints = self._get_qpoints_from_ph_folder(self.inputs.parent_folder_ph)
                self.ctx.inputs.qpoints = qpoints
                self.report(f"Successfully determined q-point mesh: {qpoints.get_kpoints_mesh()[0]}")
            except (ValueError, NotExistent) as e:
                raise ValueError(f"Could not determine the q-points mesh: {e}")

            self.ctx.inputs.parent_folder_ph = self.inputs.parent_folder_ph

        # if hasattr(self.inputs, 'parent_folder_chk'):
            if self.inputs.parent_folder_chk.is_cleaned:
                raise ValueError('Parent folder of chk calculation is cleaned')

            if not self.inputs.w90_chk_to_ukk_script:
                self.report("w90_chk_to_ukk_script is not provided. Skipping wannierization.")
                return self.exit_codes.ERROR_MISSING_W90_CHK_TO_UKK_SCRIPT

    def validate_parallelization(self):
        """Validate the parallelization settings. `epw.x` requires npool == nprocs.
        """
        try:
            resources = self.ctx.inputs.metadata.options['resources']
            num_machines = resources['num_machines']
            num_mpiprocs_per_machine = resources['num_mpiprocs_per_machine']
            total_procs = num_machines * num_mpiprocs_per_machine
        except KeyError as e:
            # If resource options are not defined, we cannot perform the check.
            # This is unlikely as AiiDA requires them, but it is a safe fallback.
            self.report(f'Could not determine total MPI processes from metadata.options.resources: {e}. Skipping check.')
            return

        # 2. Extract the `npool` value from either `parallelization` or `settings.CMDLINE`
        npool = None
        parallelization = self.ctx.inputs.get('parallelization', {})
        settings = self.ctx.inputs.get('settings', {})
        cmdline = settings.get('cmdline', [])

        if '-npool' in parallelization:
            try:
                npool = int(parallelization['-npool'])
            except (ValueError, TypeError):
                npool = None # Treat non-integer value as not set

        elif '-npool' in cmdline:
            try:
                # Find the index of '-npool' and get the next item in the list
                idx = cmdline.index('-npool')
                npool = int(cmdline[idx + 1])
            except (ValueError, IndexError, TypeError):
                # This handles cases like: ['-npool'] (at the end) or ['-npool', 'four']
                npool = None # Treat malformed cmdline as not set

        # 3. Perform the validation checks
        if npool is None:
            parallelization['-npool'] = total_procs
            self.ctx.inputs.parallelization = orm.Dict(parallelization)
        if npool != total_procs:
            raise ValueError(
                f'Validation failed: `npool` value ({npool}) is not equal to the '
                f'total number of MPI processes ({total_procs}).'
            )

    def validate_kpoints(self):
        """
        Validate the inputs related to k-points.
        `epw.x` requires coarse k-points and q-points to be compatible, which means the kpoints should be multiple of qpoints.
        e.g. if qpoints are [2,2,2], kpoints should be [2*l,2*m,2*n] for integer l,m,n.
        We firstly construct qpoints. Either an explicit `KpointsData` with given mesh/path, or a desired qpoints distance should be specified.
        In the case of the latter, the `KpointsData` will be constructed for the input `StructureData` using the `create_kpoints_from_distance` calculation function.
        Then we construct kpoints by multiplying the qpoints mesh by the `kpoints_factor`.
        """
        from ..tools.kpoints import is_compatible

        if not is_compatible(self.ctx.inputs.kpoints, self.ctx.inputs.qpoints):
            raise ValueError('The coarse kpoints and qpoints are not compatible.')

        if all(key not in self.inputs for key in ['qfpoints', 'qfpoints_distance']):
            raise ValueError('qfpoints or qfpoints_distance are required')

        if all(key not in self.inputs for key in ['kfpoints', 'kfpoints_factor']):
            raise ValueError('kfpoints or kfpoints_factor are required')

        if 'qfpoints' in self.inputs:
            qfpoints = self.inputs.qfpoints
        else:
            inputs = {
                'structure': self.inputs.structure,
                'distance': self.inputs.qfpoints_distance,
                'force_parity': self.inputs.get('qfpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_qfpoints_from_distance'
                }
            }
            qfpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        if 'kfpoints' in self.inputs:
            kfpoints = self.inputs.kfpoints
        else:
            qfpoints_mesh = qfpoints.get_kpoints_mesh()[0]
            kfpoints = orm.KpointsData()
            kfpoints.set_kpoints_mesh([v * self.inputs.kfpoints_factor.value for v in qfpoints_mesh])

        self.ctx.inputs.qfpoints = qfpoints
        self.ctx.inputs.kfpoints = kfpoints

    def prepare_process(self):
        """
        Prepare inputs for the next calculation.

        Currently, no modifications to `self.ctx.inputs` are needed before
        submission. We rely on the parent `run_process` to create the builder.
        """
        parameters = self.ctx.inputs.parameters.get_dict()

        wannierize = parameters['INPUTEPW'].get('wannierize', False)

        if wannierize:
            if 'parent_folder_epw' in self.inputs:
                raise ValueError("Should not have a parent_folder_epw if wannierize is True")

            elif 'parent_folder_chk' in self.inputs:
                raise ValueError("Should not have a chk folder if wannierize is True")

        else:
            if 'parent_folder_chk' in self.inputs:
                w90_calcjob = self.inputs.parent_folder_chk.creator
                w90_params = w90_calcjob.inputs.parameters.get_dict()
                exclude_bands = w90_params.get('exclude_bands', None) #TODO check this!

                if exclude_bands:
                    parameters['INPUTEPW']['bands_skipped'] = f'exclude_bands = {exclude_bands[0]}:{exclude_bands[-1]}'

                parameters['INPUTEPW']['nbndsub'] = w90_params['num_wann']

                wannier_chk_path = Path(self.inputs.parent_folder_chk.get_remote_path(), 'aiida.chk')
                nscf_xml_path = Path(self.inputs.parent_folder_nscf.get_remote_path(), 'out/aiida.xml')

                prepend_text = self.ctx.inputs.metadata.options.get('prepend_text', '')
                prepend_text += f'\n{self.inputs.w90_chk_to_ukk_script.get_remote_path()} {wannier_chk_path} {nscf_xml_path} aiida.ukk'

                self.ctx.inputs.metadata.options.prepend_text = prepend_text

        self.ctx.inputs.parameters = orm.Dict(parameters)

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report(f'Action taken: {action}')


    @process_handler(priority=600)
    def handle_unrecoverable_failure(self, calculation):
        """Handle calculations with an exit status below 400 which are unrecoverable, so abort the work chain."""
        if calculation.is_failed and calculation.exit_status < 400:
            self.report_error_handled(calculation, 'unrecoverable error, aborting...')
            return ProcessHandlerReport(True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)

    # @process_handler(priority=590, exit_codes=[])
    # def handle_known_unrecoverable_failure(self, calculation):
    #     """Handle calculations with an exit status that correspond to a known failure mode that are unrecoverable.

    #     These failures may always be unrecoverable or at some point a handler may be devised.
    #     """
    #     self.report_error_handled(calculation, 'known unrecoverable failure detected, aborting...')
    #     return ProcessHandlerReport(True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE)

    # @process_handler(priority=413, exit_codes=[
    #     EpwCalculation.exit_codes.ERROR_CONVERGENCE_TC_LINEAR_NOT_REACHED,
    # ])
    # def handle_convergence_tc_linear_not_reached(self, calculation):
    #     """Handle `ERROR_CONVERGENCE_TC_LINEAR_NOT_REACHED`: consider finished."""
    #     self.ctx.is_finished = True
    #     action = 'Convergence (tc_linear) was not reached. But it is acceptable if Tc can be estimated.'
    #     self.report_error_handled(calculation, action)
    #     self.results()  # Call the results method to attach the output nodes
    #     return ProcessHandlerReport(True, self.exit_codes.ERROR_CONVERGENCE_TC_LINEAR_NOT_REACHED)
