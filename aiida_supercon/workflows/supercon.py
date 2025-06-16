# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_, calcfunction

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from .b2w import EpwB2WWorkChain
from .a2f import EpwA2fWorkChain
from .iso import EpwIsoWorkChain
from .aniso import EpwAnisoWorkChain

from ..common.restart import RestartType

@calcfunction
def split_list(list_node: orm.List) -> dict:
    return {f'el_{no}': orm.Float(el) for no, el in enumerate(list_node.get_list())}

class EpwSuperConWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute the electron-phonon coupling."""
    

    _INTP_NAMESPACE = 'supercon'
    
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    _restart_from_ephmat = {
        'INPUTEPW': (
            ('elph',        False),
            ('ep_coupling', False),
            ('ephwrite',    False),
            ('restart',     True),
        )
    }
    _excluded_intp_inputs = (
        'clean_workdir', 
    )
    
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(True))
        spec.input('interpolation_distances', required=False, valid_type=orm.List)
        spec.input('convergence_threshold', required=False, valid_type=orm.Float)
        spec.input('always_run_final', required=False, valid_type=orm.Bool, default=lambda: orm.Bool(True))    
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
            EpwB2WWorkChain, namespace='b2w',
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the anisotropic `EpwCalculation`.'
            }
        )
        spec.expose_inputs(
            EpwA2fWorkChain, namespace='a2f', exclude=cls._excluded_intp_inputs,
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the anisotropic `EpwCalculation`.'
            }
        )
        spec.expose_inputs(
            EpwIsoWorkChain, namespace='iso', exclude=cls._excluded_intp_inputs,
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the isotropic `EpwCalculation`.'
            }
        )
        spec.expose_inputs(
            EpwAnisoWorkChain, namespace='aniso', exclude=cls._excluded_intp_inputs,
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the anisotropic `EpwCalculation`.'
            }
        )
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            cls.prepare_intp,
            while_(cls.should_run_conv)(
                cls.run_conv,
                cls.inspect_conv,
            ),
            if_(cls.should_run_a2f)(
                cls.run_a2f,
                cls.inspect_a2f,
            ),
            cls.run_iso,
            cls.inspect_iso,
            cls.run_aniso,
            cls.inspect_aniso,
            cls.results
        )
        # spec.output('parameters', valid_type=orm.Dict,
        #             help='The `output_parameters` output node of the final EPW calculation.')
        # spec.output('Tc_allen_dynes', valid_type=orm.Float, required=False,
        #             help='The Allen-Dynes Tc interpolated from the a2f file.')
        # spec.output('Tc_iso', valid_type=orm.Float, required=False,
        #             help='The isotropic linearised Eliashberg Tc interpolated from the max eigenvalue curve.')
        # spec.output('Tc_aniso', valid_type=orm.Float, required=False,
        #             help='The anisotropic Eliashberg Tc interpolated from the max eigenvalue curve.')

        spec.exit_code(401, 'ERROR_SUB_PROCESS_EPW',
            message='The `epw` sub process failed')
        spec.exit_code(402, 'ERROR_CONVERGENCE_NOT_REACHED',
            message='The convergence is not reached in current interpolation list.')
        spec.exit_code(403, 'ERROR_INVALID_INTERPOLATION_DISTANCE',
            message='The interpolation distance is not valid.')
        spec.exit_code(404, 'ERROR_CONVERGENCE_THRESHOLD_NOT_SPECIFIED',
            message='The convergence threshold is not specified.')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_A2F',
            message='The `a2f` sub process failed')
        spec.exit_code(406, 'ERROR_SUB_PROCESS_ISO',
            message='The `isotropic` sub process failed')
        spec.exit_code(407, 'ERROR_SUB_PROCESS_ANISO',
            message='The `aniso` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'

    @classmethod
    def get_builder_from_protocol(
            cls, 
            codes, 
            structure, 
            protocol=None, 
            from_workchain=None,
            overrides=None, 
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :TODO:
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        args = (codes, structure, protocol)
        
        builder = cls.get_builder()
        
        ## NOTE: It is user's obligation to ensure that it restart `from_workchain` of a finished `EpwB2WWorkChain`
        if from_workchain:
            if from_workchain.process_class is EpwB2WWorkChain:
                if from_workchain.is_finished:
                    builder.restart.restart_mode = orm.EnumData(RestartType.RESTART_A2F)
                    builder.pop(EpwB2WWorkChain._B2W_NAMESPACE)
                    builder.restart.overrides.parent_folder_epw = from_workchain.outputs.epw.remote_folder
                else:
                    raise ValueError("The `epw` must be a finished `EpwB2WWorkChain` or `EpwBaseWorkChain`.")
            else:
                raise ValueError("The `epw` must be a finished `EpwB2WWorkChain`.")
        else:
            builder.restart.restart_mode = orm.EnumData(RestartType.FROM_SCRATCH)
            b2w_builder = EpwB2WWorkChain.get_builder_from_protocol(
                *args,
                overrides=inputs.get(EpwB2WWorkChain._B2W_NAMESPACE, None),
                wannier_projection_type=kwargs.get('wannier_projection_type', None),
                w90_chk_to_ukk_script = kwargs.get('w90_chk_to_ukk_script', None),
            )
            
            b2w_builder.w90_intp.pop('open_grid')
            b2w_builder.w90_intp.pop('projwfc')
            
            builder.b2w = b2w_builder
            
        for (epw_namespace, epw_workchain_class) in (
            ('a2f', EpwA2fWorkChain),
            ('iso', EpwIsoWorkChain),
            ('aniso', EpwAnisoWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder_from_protocol(
                *args,
                overrides=inputs.get(epw_namespace, None),
                from_workchain=from_workchain,
                )
            if EpwB2WWorkChain._B2W_NAMESPACE in epw_builder:
                epw_builder.pop(EpwB2WWorkChain._B2W_NAMESPACE)
            
            # epw_builder.pop('restart')
            # if not from_workchain:
            #     epw_builder.restart.restart_mode = orm.EnumData(epw_workchain_class._INTP_NAMESPACE)
            builder[epw_namespace] = epw_builder

        builder.interpolation_distances = orm.List(inputs.get('interpolation_distances', None))
        builder.convergence_threshold = orm.Float(inputs['convergence_threshold'])
        builder.always_run_final = orm.Bool(inputs.get('always_run_final', True))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def validate_inputs(self):
        """Validate the inputs."""
        if self.inputs.interpolation_distances:
            if isinstance(self.inputs.interpolation_distances, orm.List) and len(self.inputs.interpolation_distances) > 1:
                if not self.inputs.convergence_threshold:
                    return self.exit_codes.ERROR_CONVERGENCE_THRESHOLD_NOT_SPECIFIED
            else:
                return self.exit_codes.ERROR_INVALID_INTERPOLATION_DISTANCE
        else:
            if self.inputs.convergence_threshold:
                return self.exit_codes.ERROR_INVALID_INTERPOLATION_DISTANCE

    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        
        self.ctx.inputs_a2f = AttributeDict(self.exposed_inputs(EpwA2fWorkChain, namespace=EpwA2fWorkChain._INTP_NAMESPACE))
        self.ctx.inputs_iso = AttributeDict(self.exposed_inputs(EpwIsoWorkChain, namespace=EpwIsoWorkChain._INTP_NAMESPACE))
        self.ctx.inputs_aniso = AttributeDict(self.exposed_inputs(EpwAnisoWorkChain, namespace=EpwAnisoWorkChain._INTP_NAMESPACE))
        
        if hasattr(self.inputs, 'interpolation_distances'):
            self.report("Will check convergence")
            self.ctx.interpolation_distances = self.inputs.get('interpolation_distances').get_list()
            self.ctx.interpolation_distances.sort()
            self.ctx.do_conv = True
            self.ctx.final_interp = None
            self.ctx.allen_dynes_values = []
            self.ctx.is_converged = False
        else:
            self.ctx.do_conv = False

    def should_run_b2w(self):
        """Check if the b2w workflow should continue or not."""
        if not self.inputs.restart.restart_mode == RestartType.FROM_SCRATCH:
            self.report('Restarting from previous `EpwB2WWorkChain`')
    
        return self.inputs.restart.restart_mode == RestartType.FROM_SCRATCH
    
    def run_b2w(self):
        """Run the b2w workflow."""
        inputs = AttributeDict(self.exposed_inputs(EpwB2WWorkChain))
        inputs.metadata.call_link_label = 'b2w'
        workchain_node = self.submit(EpwB2WWorkChain, **inputs)

        self.report(f'launching `b2w` with PK {workchain_node.pk}')

        return ToContext(b2w=workchain_node)    

    def inspect_b2w(self):
        """Inspect the b2w workflow."""
        b2w_workchain = self.ctx.b2w

        if not b2w_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {b2w_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W
    
    def prepare_intp(self):
        """Prepare the inputs for the interpolation workflow."""
        
        
        self.ctx.inputs_a2f.restart.restart_mode = EpwA2fWorkChain._RESTART_INTP
        
        parameters = self.ctx.inputs_a2f.a2f.epw.parameters.get_dict()
        
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
        
        self.ctx.inputs_a2f.restart.overrides.parent_folder_epw = parent_folder_epw
        self.ctx.inputs_a2f.a2f.epw.parameters = orm.Dict(parameters)
        
    def should_run_conv(self):
        """Check if the conv loop should continue or not."""
        if not self.ctx.do_conv:
            return False

        try:
            prev_allen_dynes = self.ctx.epw_conv[-2].outputs.output_parameters['Allen_Dynes_Tc']
            new_allen_dynes = self.ctx.epw_conv[-1].outputs.output_parameters['Allen_Dynes_Tc']
            self.ctx.is_converged = (
                abs(prev_allen_dynes - new_allen_dynes) / new_allen_dynes
                < self.inputs.convergence_threshold.value
            )
            self.report(f'Checking convergence: old {prev_allen_dynes}; new {new_allen_dynes} -> Converged = {self.ctx.is_converged.value}')
            
        except (AttributeError, IndexError, KeyError):
            self.report('Not enough data to check convergence.')

        if len(self.ctx.interpolation_distances) == 0 and not self.ctx.is_converged:
            if self.inputs.always_run_final.value:
                self.report('Allen-Dynes Tc is not converged, but will run the following workchains!.')
                return False
            else:   
                return self.exit_codes.ERROR_CONVERGENCE_NOT_REACHED

        return not self.ctx.is_converged
    
    def run_conv(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        
        inputs = self.ctx.inputs_a2f
        
        inputs.a2f.qfpoints_distance = self.ctx.interpolation_distances.pop()
        

        inputs.metadata.call_link_label = 'epw_conv'
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)

        self.report(f'launching interpolation `epw` with PK {workchain_node.pk} [qfpoints_distance = {inputs.a2f.qfpoints_distance}]')

        return ToContext(epw_conv=append_(workchain_node))

    def inspect_conv(self):
        """Verify that the conv workflow finished successfully."""
        epw_workchain = self.ctx.epw_conv[-1]
        
        if not epw_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {epw_workchain.exit_status}')
            self.ctx.epw_conv.pop()
        else:
            ## TODO: Use better way to get the mesh
            epw_calculation = epw_workchain.called_descendants[-1]
            mesh = 'x'.join(str(i) for i in epw_calculation.inputs.qfpoints.get_kpoints_mesh()[0])

            self.ctx.final_interp = epw_workchain
            try:
                self.report(f"Allen-Dynes: {epw_workchain.outputs.output_parameters['Allen_Dynes_Tc']} at {mesh}")
            except KeyError:
                self.report(f"Could not find Allen-Dynes temperature in parsed output parameters!")

    def should_run_a2f(self):
        """Check if the a2f workflow should continue or not."""
        return not self.ctx.do_conv
    
    def run_a2f(self):
        """Run the a2f workflow."""
        inputs = self.ctx.inputs_a2f
        
        inputs.metadata.call_link_label = 'a2f'
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)
        return ToContext(a2f=workchain_node)

    def inspect_a2f(self):
        """Inspect the a2f workflow."""
        a2f_workchain = self.ctx.a2f
        if not a2f_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {a2f_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_A2F
        
        self.ctx.final_interp = a2f_workchain
        
    def run_iso(self):
        """Run the iso workflow."""
        inputs = self.ctx.inputs_iso
        
        a2f_parameters = self.ctx.inputs_a2f.a2f.epw.parameters.get_dict()
        parameters = inputs.iso.epw.parameters.get_dict()
        
        for namespace, keyword in self._blocked_keywords:
            if keyword in a2f_parameters[namespace]:
                parameters[namespace][keyword] = a2f_parameters[namespace][keyword]
        
        for namespace, _params in self._restart_from_ephmat.items():
            for key, value in _params:
                parameters[namespace][key] = value
        
        inputs.restart.restart_mode = EpwIsoWorkChain._RESTART_INTP
        inputs.restart.overrides.parent_folder_epw = self.ctx.final_interp.outputs.remote_folder
        inputs.iso.epw.parameters = orm.Dict(parameters)
        
        inputs.iso.qfpoints_distance = self.ctx.final_interp.inputs.a2f.qfpoints_distance
        inputs.metadata.call_link_label = 'iso'
        workchain_node = self.submit(EpwIsoWorkChain, **inputs)
        return ToContext(iso=workchain_node)

    def inspect_iso(self):
        """Inspect the iso workflow."""
        iso_workchain = self.ctx.iso
        if not iso_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {iso_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ISO

    def run_aniso(self):
        """Run the aniso workflow."""
        inputs = self.ctx.inputs_aniso
        inputs.metadata.call_link_label = 'aniso'
        
        a2f_parameters = self.ctx.inputs_a2f.a2f.epw.parameters.get_dict()
        parameters = inputs.aniso.epw.parameters.get_dict()
        
        for namespace, keyword in self._blocked_keywords:
            if keyword in a2f_parameters[namespace]:
                parameters[namespace][keyword] = a2f_parameters[namespace][keyword] 
        
        for namespace, _params in self._restart_from_ephmat.items():
            for key, value in _params:
                parameters[namespace][key] = value
        
        inputs.aniso.epw.parameters = orm.Dict(parameters)
        
        inputs.restart.restart_mode = EpwAnisoWorkChain._RESTART_INTP
        inputs.restart.overrides.parent_folder_epw = self.ctx.final_interp.outputs.remote_folder
        inputs.aniso.qfpoints_distance = self.ctx.final_interp.inputs.a2f.qfpoints_distance
        workchain_node = self.submit(EpwAnisoWorkChain, **inputs)
        return ToContext(aniso=workchain_node)

    def inspect_aniso(self):
        """Inspect the aniso workflow."""
        aniso_workchain = self.ctx.aniso
        if not aniso_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {aniso_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ANISO
        
    def results(self):
        """TODO"""
        

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")

