# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from aiida.engine import calcfunction

from scipy.interpolate import interp1d
import numpy
import warnings

from .intp import EpwBaseIntpWorkChain
from .b2w import EpwB2WWorkChain
from .base import EpwBaseWorkChain

from ..common.restart import RestartType

@calcfunction
def calculate_iso_tc(max_eigenvalue: orm.XyData) -> orm.Float:
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

class EpwIsoWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the Allen-Dynes critical temperature."""
    
    _INTP_NAMESPACE = 'iso'
    _RESTART_INTP = RestartType.RESTART_ISO
    
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    _min_temp = 1.0
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('linearized_Eliashberg', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='Whether to use the linearized Eliashberg function.')
        
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
        spec.output('parameters', valid_type=orm.Dict,
                    help='The `output_parameters` output node of the final EPW calculation.')
        spec.output('a2f', valid_type=orm.XyData,
                    help='The contents of the `.a2f` file.')
        spec.output('max_eigenvalue', valid_type=orm.XyData,
            help='The temperature dependence of the max eigenvalue for the final EPW.')

        spec.output('Tc_iso', valid_type=orm.Float,
                    help='The isotropic Tc interpolated from the a2f file.')
        spec.exit_code(401, 'ERROR_SUB_PROCESS_EPW',
            message='The `epw` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_ISO',
            message='The `iso` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'
    
    @classmethod
    def get_builder_restart(
        cls, from_iso_workchain=None, **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        builder = cls.get_builder()
        parent_builder = from_iso_workchain.get_builder_restart()
        
        b2w = from_iso_workchain.base.links.get_outgoing(
            link_label_filter=cls._B2W_NAMESPACE
            ).first()
        
        if not b2w or b2w.node.is_finished_ok:
            builder.pop(cls._B2W_NAMESPACE)
        else:
            b2w_builder = EpwB2WWorkChain.get_builder_restart(from_b2w_workchain=b2w.node)
            builder[cls._B2W_NAMESPACE]._data = b2w_builder._data
        
        a2f = from_iso_workchain.base.links.get_outgoing(
            link_label_filter=cls._INTP_NAMESPACE
            ).first()

        
        if a2f and a2f.node.is_finished_ok:
            warnings.warn(
                f"The ISO workchain <{from_iso_workchain.pk}> is already finished.",
                stacklevel=2
                )
            return
        else:
            builder[cls._INTP_NAMESPACE]._data = parent_builder[cls._INTP_NAMESPACE]._data
        
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
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        builder = super().get_builder_from_protocol(
            codes, 
            structure, 
            protocol, 
            overrides,
            restart_intp=cls._RESTART_INTP,
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

        settings['ADDITIONAL_RETRIEVE_LIST'] = ['aiida.a2f']
        
        self.ctx.inputs.epw.settings = orm.Dict(settings)
                
    def inspect_process(self):
        """Verify that the epw.x workflow finished successfully."""
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ISO

        inputs = {
                'max_eigenvalue':intp_workchain.outputs.max_eigenvalue,
                'metadata': {
                    'call_link_label': 'calculate_iso_tc'
                }
            }
        Tc_iso = calculate_iso_tc(**inputs)
        self.ctx.Tc_iso = Tc_iso

    def results(self):
        """TODO"""
        
        super().results()
        
        # self.out('a2f', self.ctx.intp.outputs.a2f)
        # self.out('max_eigenvalue', self.ctx.intp.outputs.max_eigenvalue)
        # self.out('Tc_iso', self.ctx.Tc_iso)
