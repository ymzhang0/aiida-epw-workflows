# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from .base import EpwBaseWorkChain
from .a2f import EpwA2fWorkChain
from .iso import EpwIsoWorkChain
from .aniso import EpwAnisoWorkChain
from aiida.engine import calcfunction

from .intp import EpwBaseIntpWorkChain

@calcfunction
def split_list(list_node: orm.List) -> dict:
    return {f'el_{no}': orm.Float(el) for no, el in enumerate(list_node.get_list())}

class EpwSuperConWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the electron-phonon coupling."""
    
    _INTP_NAMESPACE = 'supercon'
    
    _blocked_keywords = [
        ('epw', 'wannierize'),
        ('epw', 'epwread'),
        ('epw', 'epwwrite'),
        ('epw', 'epwread'),
    ]
    
    _excluded_a2f_inputs = (
        'structure', 'kfpoints_factor', 'parent_folder_epw', 'clean_workdir',
    )
    
    _excluded_iso_inputs = (
        'structure', 'kfpoints_factor', 'parent_folder_epw', 'clean_workdir',
    )
    
    _excluded_aniso_inputs = (
        'structure', 'kfpoints_factor', 'parent_folder_epw', 'clean_workdir',
    )
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('interpolation_distance', required=False, valid_type=(orm.Float, orm.List))
        spec.input('convergence_threshold', required=False, valid_type=orm.Float)


        spec.expose_inputs(
            EpwA2fWorkChain, namespace='a2f', exclude=cls._excluded_a2f_inputs,
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the anisotropic `EpwCalculation`.'
            }
        )
        spec.expose_inputs(
            EpwIsoWorkChain, namespace='iso', exclude=cls._excluded_iso_inputs,
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the isotropic `EpwCalculation`.'
            }
        )
        spec.expose_inputs(
            EpwAnisoWorkChain, namespace='aniso', exclude=cls._excluded_aniso_inputs,
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
            if_(cls.should_run_conv)(
                cls.run_conv,
                cls.inspect_conv,
            ),
            if_(lambda self: not cls.should_run_conv(self))(
                cls.run_a2f,
                cls.inspect_a2f,
            ),
            if_(cls.should_run_iso)(
                cls.run_iso,
                cls.inspect_iso,
            ),
            if_(cls.should_run_aniso)(
                cls.run_aniso,
                cls.inspect_aniso,
            ),
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
        spec.exit_code(403, 'ERROR_SUB_PROCESS_A2F',
            message='The `a2f` sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_ISO',
            message='The `isotropic` sub process failed')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_ANISO',
            message='The `aniso` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / 'supercon.yaml'

    @classmethod
    def get_builder_from_protocol(
            cls, 
            codes, 
            structure, 
            protocol=None, 
            overrides=None, 
            b2w=None,
            interpolation_distance=None,
            convergence_threshold=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :TODO:
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        args = (codes, structure, protocol)
        
        builder = cls.get_builder()
        builder.structure = structure
        
        for (epw_namespace, epw_workchain_class) in (
            ('a2f', EpwA2fWorkChain),
            ('iso', EpwIsoWorkChain),
            ('aniso', EpwAnisoWorkChain),
        ):
            epw_inputs = inputs.get(epw_namespace, None)
            epw_builder = epw_workchain_class.get_builder_from_protocol(
                codes=codes,
                structure=structure,
                protocol='fast',
                overrides=inputs.get(epw_namespace, None),
                b2w=b2w,
                wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
                w90_chk_to_ukk_script = w90_script,
                # reference_bands = bands_wc.outputs.band_structure,
                # bands_kpoints = bands_wc.outputs.band_structure.creator.inputs.kpoints,
                )
            epw_builder.pop('epw')
            epw_builder[epw_namespace]['code'] = codes['epw']
            epw_builder[epw_namespace]['metadata'] = epw_inputs[epw_namespace]['metadata']
            if 'settings' in epw_inputs[epw_namespace]:
                epw_builder[epw_namespace]['settings'] = orm.Dict(epw_inputs[epw_namespace]['settings'])

            epw_builder[epw_namespace]['parameters'] = orm.Dict(epw_inputs[epw_namespace].get('parameters', {}))
            builder[epw_namespace] = epw_builder

            
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def validate_inputs(self):
        """Validate the inputs."""
        if hasattr(self.inputs, 'parent_folder_epw') and hasattr(self.inputs, 'epw'):
            raise ValueError("Only one of `parent_folder_epw` or `epw` can be accepted.")

    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        super().setup()
        
        self.ctx.inputs = AttributeDict(self.inputs)
        
        if hasattr(self.inputs, 'interpolation_distance'):
            interpolation_distance = self.inputs.get('interpolation_distance')
            if isinstance(interpolation_distance, orm.List):
                self.ctx.interpolation_list = list(split_list(interpolation_distance).values())
            else:
                self.ctx.interpolation_list = [interpolation_distance]

            self.ctx.interpolation_list.sort()
            self.ctx.final_interp = None
            self.ctx.allen_dynes_values = []
            self.ctx.is_converged = False

        self.ctx.degaussq = None

    def should_run_conv(self):
        """Check if the conv loop should continue or not."""
        if not hasattr(self.inputs, 'interpolation_distance'):
            self.ctx.qfpoints_distance = self.inputs.qfpoints_distance
            self.ctx.kfpoints_factor = self.inputs.kfpoints_factor
            return False
        if 'convergence_threshold' in self.inputs:
            self.report('Will check convergence')
            try:
                prev_allen_dynes = self.ctx.epw_interp[-2].outputs.output_parameters['allen_dynes']
                new_allen_dynes = self.ctx.epw_interp[-1].outputs.output_parameters['allen_dynes']
                self.ctx.is_converged = (
                    abs(prev_allen_dynes - new_allen_dynes)
                    < self.inputs.convergence_threshold
                )
                self.report(f'Checking convergence: old {prev_allen_dynes}; new {new_allen_dynes} -> Converged = {self.ctx.is_converged.value}')
            except (AttributeError, IndexError, KeyError):
                self.report('Not enough data to check convergence.')

        else:
            self.report('No `convergence_threshold` input was provided, convergence automatically achieved.')
            self.ctx.qfpoints_distance = self.inputs.interpolation_distance[0]
            self.ctx.kfpoints_factor = self.inputs.kfpoints_factor
            self.ctx.is_converged = True

        if len(self.ctx.interpolation_list) == 0 and not self.ctx.is_converged:
            self.exit_codes.ERROR_CONVERGENCE_NOT_REACHED
            
        return len(self.ctx.interpolation_list) > 0 and not self.ctx.is_converged
    
    def run_conv(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='a2f'))
        
        inputs.structure = self.inputs.structure
        inputs.parent_folder_epw = self.ctx.epw.outputs.epw_folder
        inputs.qfpoints_distance = self.ctx.interpolation_list.pop()
        inputs.kfpoints_factor = self.inputs.kfpoints_factor
        
        try:
            settings = inputs.settings.get_dict()
        except AttributeError:
            settings = {}
            
        inputs.settings = orm.Dict(settings)

        if self.ctx.degaussq:
            parameters = inputs.parameters.get_dict()
            parameters['INPUTEPW']['degaussq'] = self.ctx.degaussq
            inputs.parameters = orm.Dict(parameters)

        inputs.metadata.call_link_label = 'epw_interp'
        calcjob_node = self.submit(EpwCalculation, **inputs)
        mesh = 'x'.join(str(i) for i in self.ctx.qfpoints.get_kpoints_mesh()[0])
        self.report(f'launching interpolation `epw` with PK {calcjob_node.pk} and interpolation mesh {mesh}')

        return ToContext(epw_interp=append_(calcjob_node))

    def inspect_conv(self):
        """Verify that the conv workflow finished successfully."""
        epw_calculation = self.ctx.epw_interp[-1]

        if not epw_calculation.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {epw_calculation.exit_status}')
            self.ctx.epw_interp.pop()
            # return self.exit_codes.ERROR_SUB_PROCESS_EPW_INTERP
        else:
            self.ctx.final_interp = self.ctx.inter_points
            try:
                self.report(f"Allen-Dynes: {epw_calculation.outputs.output_parameters['allen_dynes']}")
            except KeyError:
                self.report(f"Could not find Allen-Dynes temperature in parsed output parameters!")

            if self.ctx.degaussq is None:
                frequency = epw_calculation.outputs.a2f.get_array('frequency')
                self.ctx.degaussq = frequency[-1] / 100

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

