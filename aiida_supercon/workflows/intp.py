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
                'clean_workdir',
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
            cls.prepare_process,
            cls.run_process,
            cls.inspect_process,
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

    @classmethod
    def get_builder_from_protocol(
            cls, 
            codes, 
            structure, 
            protocol=None, 
            from_workchain=None,
            overrides=None,
            restart_intp = None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        inputs = cls.get_protocol_inputs(protocol, overrides)
        builder = cls.get_builder()
        builder.structure = structure

        ## NOTE: It's user's obligation to provide the 
        ##       finished EpwIntpWorkChain as intp
        if from_workchain:
            if from_workchain.process_class == EpwB2WWorkChain:
                if from_workchain.is_finished and from_workchain.process_class is EpwB2WWorkChain:
                    builder.restart.restart_mode = orm.EnumData(RestartType.RESTART_A2F)
                    builder.pop(cls._B2W_NAMESPACE)
                    builder.restart.overrides.parent_folder_epw = from_workchain.outputs.epw.remote_folder
                else:
                    raise ValueError("The `epw` must be a finished `EpwB2WWorkChain` or `EpwBaseWorkChain`.")
            elif (
                issubclass(from_workchain.process_class, EpwBaseIntpWorkChain) and
                from_workchain.is_finished
            ):
                b2w = from_workchain.base.links.get_outgoing(link_label_filter=cls._B2W_NAMESPACE).first().node
                if not b2w.is_finished:
                    raise ValueError("The `b2w` must be a finished `EpwB2WWorkChain`.")
                builder.restart.restart_mode = orm.EnumData(restart_intp)
                builder.pop(cls._B2W_NAMESPACE)
                builder.restart.overrides.parent_folder_epw = b2w.outputs.epw.remote_folder
            else:
                raise ValueError("The `epw` must be a finished `EpwB2WWorkChain` or `EpwBaseWorkChain`.")
        else:
            builder.restart.restart_mode = orm.EnumData(RestartType.FROM_SCRATCH)
            b2w_builder = EpwB2WWorkChain.get_builder_from_protocol(
                codes=codes,
                structure=structure,
                protocol=protocol,
                overrides=inputs.get(cls._B2W_NAMESPACE, None),
                wannier_projection_type=kwargs.get('wannier_projection_type', None),
                w90_chk_to_ukk_script = kwargs.get('w90_chk_to_ukk_script', None),
            )
            
            b2w_builder.w90_intp.pop('open_grid')
            b2w_builder.w90_intp.pop('projwfc')
            
            builder[cls._B2W_NAMESPACE] = b2w_builder
            
        builder[cls._INTP_NAMESPACE] = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._INTP_NAMESPACE, None),
            **kwargs
        )


        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder
    
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
        return self.inputs.restart.restart_mode == RestartType.FROM_SCRATCH
    
    def run_b2w(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""

        self.report(f'Running B2W...')
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
        
    def run_process(self):
        """Prepare the process for the current interpolation distance."""
        
        inputs = self.ctx.inputs
        parameters = inputs.parameters.get_dict()
        
        if self.inputs.restart.restart_mode == RestartType.FROM_SCRATCH:
            b2w_workchain = self.ctx.b2w

            b2w_parameters = b2w_workchain.inputs.epw.parameters.get_dict()
            
            parent_folder_epw = b2w_workchain.outputs.epw.remote_folder

        else:
            parent_folder_epw = self.inputs.restart.overrides.parent_folder_epw
            b2w_parameters = parent_folder_epw.creator.inputs.parameters.get_dict()
        
        for namespace, keyword in self._blocked_keywords:
            if keyword in b2w_parameters[namespace]:
                parameters[namespace][keyword] = b2w_parameters[namespace][keyword]
        
        inputs.parent_folder_epw = parent_folder_epw
        inputs.epw.parameters = orm.Dict(parameters)
        
        inputs.metadata.call_link_label = self._INTP_NAMESPACE
        workchain_node = self.submit(EpwBaseWorkChain, **inputs)

        self.report(f'launching `{self._INTP_NAMESPACE}` with PK {workchain_node.pk}')

        return ToContext(intp=workchain_node)


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
            if hasattr(self.ctx, self.b2w):
                self.ctx.b2w.outputs.remote_folder._clean()
            if hasattr(self.ctx, self.intp):
                self.ctx.intp.outputs.remote_folder._clean()