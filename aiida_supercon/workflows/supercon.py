# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_, calcfunction

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.common.types import SpinType, ElectronicType
from aiida_wannier90_workflows.common.types import WannierProjectionType

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
    

    _NAMESPACE = 'supercon'
    
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
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(True))
        spec.input('interpolation_distances', required=False, valid_type=orm.List)
        spec.input('convergence_threshold', required=False, valid_type=orm.Float)
        spec.input('always_run_final', required=False, valid_type=orm.Bool, default=lambda: orm.Bool(True))    
        
        spec.expose_inputs(
            EpwB2WWorkChain, 
            namespace=EpwA2fWorkChain._B2W_NAMESPACE, 
            exclude=('clean_workdir'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the anisotropic `EpwCalculation`.'
            }
        )
        spec.expose_inputs(
            EpwA2fWorkChain, 
            namespace=EpwA2fWorkChain._INTP_NAMESPACE, 
            exclude=(
                'clean_workdir', 
                f'{EpwA2fWorkChain._INTP_NAMESPACE}.parent_folder_nscf',
                f'{EpwA2fWorkChain._INTP_NAMESPACE}.parent_folder_chk',
                f'{EpwA2fWorkChain._INTP_NAMESPACE}.parent_folder_ph',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the anisotropic `EpwCalculation`.'
            }
        )
        spec.expose_inputs(
            EpwIsoWorkChain, 
            namespace=EpwIsoWorkChain._INTP_NAMESPACE, 
            exclude=(
                'clean_workdir', 
                f"{EpwIsoWorkChain._INTP_NAMESPACE}.parent_folder_nscf",
                f"{EpwIsoWorkChain._INTP_NAMESPACE}.parent_folder_chk",
                f"{EpwIsoWorkChain._INTP_NAMESPACE}.parent_folder_ph",
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the isotropic `EpwCalculation`.'
            }
        )
        spec.expose_inputs(
            EpwAnisoWorkChain, 
            namespace=EpwAnisoWorkChain._INTP_NAMESPACE, 
            exclude=(
                'clean_workdir', 
                f"{EpwAnisoWorkChain._INTP_NAMESPACE}.parent_folder_nscf",
                f"{EpwAnisoWorkChain._INTP_NAMESPACE}.parent_folder_chk",
                f"{EpwAnisoWorkChain._INTP_NAMESPACE}.parent_folder_ph",
            ),
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

        spec.expose_outputs(
            EpwB2WWorkChain,
            namespace=EpwA2fWorkChain._B2W_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the anisotropic `EpwCalculation`.'
            }
        )
        
        spec.expose_outputs(
            EpwA2fWorkChain,
            namespace=EpwA2fWorkChain._INTP_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the anisotropic `EpwCalculation`.'
            }
        )
        
        spec.expose_outputs(
            EpwIsoWorkChain,
            namespace=EpwIsoWorkChain._INTP_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the isotropic `EpwCalculation`.'
            }
        )
        
        spec.expose_outputs(
            EpwAnisoWorkChain,
            namespace=EpwAnisoWorkChain._INTP_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the anisotropic `EpwCalculation`.'
            }
        )

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
        return files(protocols) / f'{cls._NAMESPACE}.yaml'

    @classmethod
    def get_builder_restart_from_b2w(
        cls,
        from_b2w_workchain: orm.WorkChainNode,
        protocol=None,
        overrides=None,
        **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        inputs = cls.get_protocol_inputs(protocol, overrides)
        
        builder = cls.get_builder()
        
        if not from_b2w_workchain or not from_b2w_workchain.process_class == EpwB2WWorkChain:
            raise ValueError('Currently we only accept `EpwB2WWorkChain`')
    
        # b2w_parameters = from_b2w_workchain.inputs.epw.parameters.get_dict()
        
        # parameters = builder.epw.parameters.get_dict()
        
        # for namespace, keyword in cls._blocked_keywords:
        #     if keyword in b2w_parameters[namespace]:
        #         parameters[namespace][keyword] = b2w_parameters[namespace][keyword]
        if from_b2w_workchain.is_finished_ok:
            builder.pop(EpwA2fWorkChain._B2W_NAMESPACE)
            parent_folder_epw = from_b2w_workchain.outputs.epw.remote_folder
        else:
            b2w_builder = EpwB2WWorkChain.get_builder_restart(
                from_b2w_workchain=from_b2w_workchain,
                protocol=protocol,
                overrides=overrides.get(EpwA2fWorkChain._B2W_NAMESPACE, None),
                **kwargs
                )
            
            # Actually there is no exclusion of EpwB2WWorkChain namespace
            # So we need to set the _data manually
            builder[EpwA2fWorkChain._B2W_NAMESPACE]._data = b2w_builder._data
        
        for (epw_namespace, epw_workchain_class) in (
            (EpwA2fWorkChain._INTP_NAMESPACE, EpwA2fWorkChain),
            (EpwIsoWorkChain._INTP_NAMESPACE, EpwIsoWorkChain),
            (EpwAnisoWorkChain._INTP_NAMESPACE, EpwAnisoWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder_restart_from_b2w(
                from_b2w_workchain=from_b2w_workchain,
                protocol=protocol,
                overrides=overrides.get(epw_namespace, None),
                **kwargs
                )
            
            if epw_workchain_class._B2W_NAMESPACE in epw_builder:
                epw_builder.pop(epw_workchain_class._B2W_NAMESPACE)
                
            builder[epw_namespace]._data = epw_builder._data
        
        if parent_folder_epw:
            builder[EpwA2fWorkChain._INTP_NAMESPACE].parent_folder_epw = parent_folder_epw
        
        builder.interpolation_distances = orm.List(inputs.get('interpolation_distances', None))
        builder.convergence_threshold = orm.Float(inputs['convergence_threshold'])
        builder.always_run_final = orm.Bool(inputs.get('always_run_final', True))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        
        return builder
    
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

        :TODO:
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        args = (codes, structure, protocol)
        
        builder = cls.get_builder()
        
        b2w_builder = EpwB2WWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get(EpwB2WWorkChain._B2W_NAMESPACE, None),
            wannier_projection_type=kwargs.get('wannier_projection_type', WannierProjectionType.ATOMIC_PROJECTORS_QE),
            w90_chk_to_ukk_script = kwargs.get('w90_chk_to_ukk_script', None),
            reference_bands = kwargs.get('reference_bands', None),
            bands_kpoints = kwargs.get('bands_kpoints', None),
        )

        b2w_builder.w90_intp.pop('open_grid')
        b2w_builder.w90_intp.pop('projwfc')

        builder.b2w = b2w_builder
            
        for (epw_namespace, epw_workchain_class) in (
            ('a2f', EpwA2fWorkChain),
            ('iso', EpwIsoWorkChain),
            ('aniso', EpwAnisoWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder()
            
            epw_builder.pop(epw_workchain_class._B2W_NAMESPACE)
            
            builder[epw_namespace]._data = epw_builder._data
            
            builder.structure = structure
            # builder.qfpoints_distance = orm.Float(inputs.get('qfpoints_distance', None))
            # builder.kfpoints_factor = orm.Float(inputs.get('kfpoints_factor', None))
            
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

        return 'b2w' in self.inputs
    def run_b2w(self):
        """Run the b2w workflow."""
        inputs = AttributeDict(self.exposed_inputs(EpwB2WWorkChain, namespace=EpwB2WWorkChain._B2W_NAMESPACE))
        inputs.metadata.call_link_label = EpwA2fWorkChain._B2W_NAMESPACE
        workchain_node = self.submit(EpwB2WWorkChain, **inputs)

        self.report(f'launching `b2w` with PK {workchain_node.pk}')

        return ToContext(workchain_b2w=workchain_node)    

    def inspect_b2w(self):
        """Inspect the b2w workflow."""
        b2w_workchain = self.ctx.workchain_b2w

        if not b2w_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {b2w_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W
    
    def prepare_intp(self):
        """Prepare the inputs for the interpolation workflow."""
        
                
        parameters = self.ctx.inputs_a2f[EpwA2fWorkChain._INTP_NAMESPACE].epw.parameters.get_dict()
        
        if self.should_run_a2f():
            b2w_workchain = self.ctx.workchain_b2w

            b2w_parameters = b2w_workchain.inputs[EpwA2fWorkChain._B2W_NAMESPACE].epw.parameters.get_dict()
            
            parent_folder_epw = b2w_workchain.outputs.epw.remote_folder

            for namespace, keyword in self._blocked_keywords:
                if keyword in b2w_parameters[namespace]:
                    parameters[namespace][keyword] = b2w_parameters[namespace][keyword]
            
            self.ctx.inputs_a2f[EpwA2fWorkChain._INTP_NAMESPACE].parent_folder_epw = parent_folder_epw
            self.ctx.inputs_a2f[EpwA2fWorkChain._INTP_NAMESPACE].epw.parameters = orm.Dict(parameters)
            
    def should_run_conv(self):
        """Check if the conv loop should continue or not."""
        if not self.ctx.do_conv:
            return False

        try:
            prev_allen_dynes = self.ctx.a2f_conv[-2].outputs.output_parameters['Allen_Dynes_Tc']
            new_allen_dynes = self.ctx.a2f_conv[-1].outputs.output_parameters['Allen_Dynes_Tc']
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
        

        inputs.metadata.call_link_label = 'a2f_conv'
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)

        self.report(f'launching interpolation `epw` with PK {workchain_node.pk} [qfpoints_distance = {inputs.a2f.qfpoints_distance}]')

        return ToContext(a2f_conv=append_(workchain_node))

    def inspect_conv(self):
        """Verify that the conv workflow finished successfully."""
        a2f_workchain = self.ctx.a2f_conv[-1]
        
        if not a2f_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {a2f_workchain.exit_status}')
            self.ctx.a2f_conv.pop()
        else:
            ## TODO: Use better way to get the mesh
            a2f_calculation = a2f_workchain.called_descendants[-1]
            mesh = 'x'.join(str(i) for i in a2f_calculation.inputs.qfpoints.get_kpoints_mesh()[0])

            self.ctx.final_interp = a2f_workchain
            try:
                self.report(f"Allen-Dynes: {a2f_workchain.outputs.output_parameters['Allen_Dynes_Tc']} at {mesh}")
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
        return ToContext(workchain_a2f=workchain_node)

    def inspect_a2f(self):
        """Inspect the a2f workflow."""
        a2f_workchain = self.ctx.workchain_a2f
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
        return ToContext(workchain_iso=workchain_node)

    def inspect_iso(self):
        """Inspect the iso workflow."""
        iso_workchain = self.ctx.workchain_iso
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
        return ToContext(workchain_aniso=workchain_node)

    def inspect_aniso(self):
        """Inspect the aniso workflow."""
        aniso_workchain = self.ctx.workchain_aniso
        if not aniso_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {aniso_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ANISO
        
    def results(self):
        """TODO"""
        
        if 'workchain_b2w' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_b2w, 
                    EpwB2WWorkChain,
                    namespace=EpwB2WWorkChain._B2W_NAMESPACE
                )
            )
        
        # Either expose the last convergence
        
        if 'a2f_conv' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.a2f_conv[-1], 
                    EpwA2fWorkChain,
                    namespace=EpwA2fWorkChain._INTP_NAMESPACE
                )
            )
        
        # Or expose the A2F workchain since they are exclusive
        if 'workchain_a2f' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_a2f, 
                    EpwA2fWorkChain,
                    namespace=EpwA2fWorkChain._INTP_NAMESPACE
                )
            )
            
        if 'workchain_iso' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_iso, 
                    EpwIsoWorkChain,
                    namespace=EpwIsoWorkChain._INTP_NAMESPACE
                )
            )
            
        if 'workchain_aniso' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_aniso, 
                    EpwAnisoWorkChain,
                    namespace=EpwAnisoWorkChain._INTP_NAMESPACE
                )
            )

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

