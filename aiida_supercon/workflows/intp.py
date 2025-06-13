# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_, ExitCode

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida.engine import calcfunction

from ..common.restart import RestartType

from .base import EpwBaseWorkChain
from .b2w import EpwB2WWorkChain

class EpwBaseIntpWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute the Allen-Dynes critical temperature."""
    
    # --- Child classes should override these placeholders ---
    _B2W_NAMESPACE = 'b2w'  # e.g., 'intp' for A2fWorkChain
    _INTP_NAMESPACE = 'intp'  # e.g., 'a2f' for A2fWorkChain
    # ---------------------------------------------------------

    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        
        spec.input_namespace(
            'restart', required=True
            )
        
        spec.input(
            'restart.restart_mode', required=True, 
            valid_type=orm.EnumData, default=lambda: orm.EnumData(RestartType.FROM_SCRATCH),
            help='The restart mode for the `EpwBaseWorkChain`.'
            )
        
        spec.input_namespace(
            'restart.overrides', required=False, dynamic=True,
            help='The overrides for the `EpwBaseWorkChain`.'
            )

        spec.expose_inputs(
            EpwB2WWorkChain, namespace=cls._B2W_NAMESPACE, exclude=(
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the interpolation `EpwCalculation`s.'
            }
        )
        spec.expose_inputs(
            EpwBaseWorkChain, namespace=cls._INTP_NAMESPACE, exclude=(
                'parent_folder_epw',
                'parent_folder_nscf',
                'parent_folder_ph',
                'parent_folder_chk',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the a2f `EpwBaseWorkChain`.'
            }
        )
        spec.outline(
            cls.setup,
            cls.validate_parent_folders,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            cls.run_intp,
            cls.inspect_intp,
            cls.results
        )
        spec.output('parameters', valid_type=orm.Dict,
                    help='The `output_parameters` output node of the final EPW calculation.')
        spec.output('remote_folder', valid_type=orm.RemoteData,
                    help='The remote folder of the final EPW calculation.')
        spec.exit_code(401, 'ERROR_SUB_PROCESS_EPW',
            message='The `epw` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_EPW_A2F',
            message='The `a2f` sub process failed')
        spec.exit_code(403, 'ERROR_INVALID_INPUT_PARENT_FOLDER_EPW',
            message='The parent folder of the EPW calculation is not valid.')
        spec.exit_code(404, 'ERROR_INVALID_INPUT_A2F_NAMESPACE',
            message='The a2f namespace is not valid.')


    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        self.ctx.degaussq = None
        inputs = self.exposed_inputs(EpwBaseWorkChain, namespace=self._INTP_NAMESPACE)
        self.ctx.inputs = inputs
        
    def validate_parent_folders(self):
        """Validate the parent folders."""
        
        if hasattr(self.inputs, 'parent_folder_epw'):
            parent_epw_wc = self.inputs.parent_folder_epw.creator.caller
            if parent_epw_wc.process_class != EpwB2WWorkChain:
                return self.exit_codes.ERROR_INVALID_INPUT_PARENT_FOLDER_EPW


    def should_run_b2w(self):
        """Check if the epw loop should continue or not."""
        
        return self.inputs.restart.restart_mode is RestartType.FROM_SCRATCH
    
    def run_b2w(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""

        inputs = self.exposed_inputs(EpwB2WWorkChain, namespace=self._B2W_NAMESPACE)
        
        
        inputs.metadata.call_link_label = self._B2W_NAMESPACE
        workchain_node = self.submit(EpwB2WWorkChain, **inputs)

        self.report(f'launching `{self._B2W_NAMESPACE}` with PK {workchain_node.pk}')

        return ToContext(b2w=workchain_node)

    def inspect_b2w(self):
        """Verify that the epw.x workflow finished successfully."""
        b2w_workchain = self.ctx.b2w

        if not b2w_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {b2w_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W

    def run_intp(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        
        inputs = self.ctx.inputs

        if self.inputs.restart.restart_mode == RestartType.FROM_SCRATCH:
            intp_workchain = self.ctx.b2w
        
            parameters = inputs.parameters.get_dict()
            intp_parameters = intp_workchain.inputs.epw.parameters.get_dict()
            
            for namespace, _parameters in self._defaults_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value
            for namespace, keyword in self._blocked_keywords:
                if keyword in intp_parameters[namespace]:
                    parameters[namespace][keyword] = intp_parameters[namespace][keyword]
            
            inputs.parameters = orm.Dict(parameters)

            inputs.parent_folder_epw = intp_workchain.outputs.epw.remote_folder

        elif self.ctx.restart.restart_mode == RestartType.RESTART_A2F:
            inputs.parent_folder_epw = self.inputs.restart.overrides.parent_folder_epw
            parameters = inputs.parameters.get_dict()
            
            for namespace, _parameters in self._defaults_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value
        try:
            settings = inputs.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = ['aiida.a2f']
        inputs.settings = orm.Dict(settings)

        inputs.metadata.call_link_label = self._INTP_NAMESPACE
        workchain_node = self.submit(EpwBaseWorkChain, **inputs)

        self.report(f'launching `{self._INTP_NAMESPACE}` with PK {workchain_node.pk}')

        return ToContext(intp=workchain_node)

    def inspect_intp(self):
        """Verify that the epw.x workflow finished successfully."""
        intp = self.ctx.intp

        if not intp.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_EPW_INTP


    def results(self):
        """TODO"""
        
        # self.out('Tc_a2f', self.ctx.Tc_a2f)
        self.out('parameters', self.ctx.intp.outputs.output_parameters)
        self.out('remote_folder', self.ctx.intp.outputs.remote_folder)
        

    def on_terminated(self):
        """Clean up the work chain."""
        super().on_terminated()
        if self.inputs.clean_workdir.value:
            self.report('cleaning remote folders')
            if hasattr(self.ctx, 'b2w'):
                self.ctx.b2w.outputs.remote_folder._clean()
            if hasattr(self.ctx, 'intp'):
                self.ctx.intp.outputs.remote_folder._clean()