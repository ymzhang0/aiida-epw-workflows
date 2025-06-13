# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_, ExitCode

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from .base import EpwBaseWorkChain
from .intp import EpwBaseIntpWorkChain
from aiida.engine import calcfunction

from scipy.interpolate import interp1d
import numpy
from ..common.restart import RestartType
from .utils.dict import get_recursive_input_ports

from .b2w import EpwB2WWorkChain

@calcfunction
def calculate_tc(max_eigenvalue: orm.XyData) -> orm.Float:
    me_array = max_eigenvalue.get_array('max_eigenvalue')
    if me_array[:, 1].max() < 1.0:
        return orm.Float(0.0)
    else:
        return orm.Float(float(interp1d(me_array[:, 1], me_array[:, 0])(1.0)))

@calcfunction
def calculate_Allen_Dynes_tc(a2f: orm.ArrayData, mustar = 0.13) -> orm.Float:
    w        = a2f.get_array('frequency')
    # Here we preassume that there are 10 smearing values for a2f calculation
    spectral = a2f.get_array('a2f')[:, 9]   
    mev2K    = 11.604525006157

    _lambda  = 2*numpy.trapz(numpy.divide(spectral, w), x=w)

    # wlog =  np.exp(np.average(np.divide(alpha, w), weights=np.log(w)))
    wlog     =  numpy.exp(2/_lambda*numpy.trapz(numpy.multiply(numpy.divide(spectral, w), numpy.log(w)), x=w))

    Tc = wlog/1.2*numpy.exp(-1.04*(1+_lambda)/(_lambda-mustar*(1+0.62*_lambda))) * mev2K


    return orm.Float(Tc)

class EpwA2fWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the Allen-Dynes critical temperature."""
    
    _INTP_NAMESPACE = 'a2f'
    
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    _defaults_parameters = {
        'INPUTEPW': {
            'degaussw'   : 0.01,
            'eps_acustic' : 0.1,
            'iverbosity' : 1,
            'fsthick'    : 0.8,
            'temps'      : 1,
        }
    }
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)


        spec.outline(
            cls.setup,
            cls.validate_parent_folders,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            cls.run_a2f,
            cls.inspect_a2f,
            cls.results
        )
        spec.output('a2f', valid_type=orm.XyData,
                    help='The contents of the `.a2f` file.')
        spec.output('Tc_allen_dynes', valid_type=orm.Float,
                    help='The Allen-Dynes Tc interpolated from the a2f file.')

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
            b2w=None,
            overrides=None, 
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        inputs = cls.get_protocol_inputs(protocol, overrides)
        builder = cls.get_builder()
        builder.structure = structure

        ## NOTE: It's user's obligation to provide the 
        ##       finished EpwIntpWorkChain as intp
        if b2w:
            if b2w.is_finished and b2w.process_class in (
                EpwB2WWorkChain, EpwBaseWorkChain
            ):
                builder.restart.restart_mode = orm.EnumData(RestartType.RESTART_A2F)
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

    def run_a2f(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        
        inputs = self.ctx.inputs

        if self.inputs.restart.restart_mode == RestartType.FROM_SCRATCH:
            b2w_workchain = self.ctx.b2w
        
            parameters = inputs.epw.parameters.get_dict()
            b2w_parameters = b2w_workchain.inputs.epw.parameters.get_dict()
            
            for namespace, _parameters in self._defaults_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value
            for namespace, keyword in self._blocked_keywords:
                if keyword in b2w_parameters[namespace]:
                    parameters[namespace][keyword] = b2w_parameters[namespace][keyword]
            
            inputs.epw.parameters = orm.Dict(parameters)

            inputs.parent_folder_epw = b2w_workchain.outputs.epw.remote_folder

        elif self.inputs.restart.restart_mode == RestartType.RESTART_A2F:
            inputs.parent_folder_epw = self.inputs.restart.overrides.parent_folder_epw
            parameters = inputs.epw.parameters.get_dict()
            
            for namespace, _parameters in self._defaults_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value
        try:
            settings = inputs.epw.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = ['aiida.a2f']
        inputs.epw.settings = orm.Dict(settings)

        inputs.metadata.call_link_label = self._INTP_NAMESPACE
        workchain_node = self.submit(EpwBaseWorkChain, **inputs)

        self.report(f'launching `{self._INTP_NAMESPACE}` with PK {workchain_node.pk}')

        return ToContext(a2f=workchain_node)

    def inspect_a2f(self):
        """Verify that the epw.x workflow finished successfully."""
        a2f = self.ctx.a2f

        if not a2f.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {a2f.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_EPW_A2F


    def results(self):
        """TODO"""
        
        # self.out('Tc_a2f', self.ctx.Tc_a2f)
        self.out('parameters', self.ctx.a2f.outputs.output_parameters)
        self.out('a2f', self.ctx.a2f.outputs.a2f)
        self.out('remote_folder', self.ctx.a2f.outputs.remote_folder)
        
        # Calculate Tc using Allen-Dynes formula
        tc = calculate_Allen_Dynes_tc(self.ctx.a2f.outputs.a2f)
        self.out('Tc_allen_dynes', tc)

    def on_terminated(self):
        """Clean up the work chain."""
        super().on_terminated()
        if self.inputs.clean_workdir.value:
            self.report('cleaning remote folders')
            if hasattr(self.ctx, 'epw'):
                self.ctx.epw.outputs.remote_folder._clean()
            if hasattr(self.ctx, 'a2f'):
                self.ctx.a2f.outputs.remote_folder._clean()