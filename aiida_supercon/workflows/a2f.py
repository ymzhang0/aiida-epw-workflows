# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm
from aiida.engine import calcfunction, ToContext, if_


from .base import EpwBaseWorkChain
from .b2w import EpwB2WWorkChain
from .intp import EpwBaseIntpWorkChain

from scipy.interpolate import interp1d
import numpy
from ..common.restart import RestartType


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
    _RESTART_INTP = RestartType.RESTART_A2F
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
        spec.output('a2f', valid_type=orm.XyData,
                    help='The contents of the `.a2f` file.')
        # spec.output('Tc_allen_dynes', valid_type=orm.Float,
        #             help='The Allen-Dynes Tc interpolated from the a2f file.')

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
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        builder = super().get_builder_from_protocol(
            codes, 
            structure, 
            protocol, 
            from_workchain, 
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
        intp = self.ctx.intp

        if not intp.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_EPW_A2F


    def results(self):
        """TODO"""

        super().results()
        
        self.out('a2f', self.ctx.intp.outputs.a2f)

    def on_terminated(self):
        """Clean up the work chain."""
        super().on_terminated()
        if self.inputs.clean_workdir.value:
            self.report('cleaning remote folders')
            if hasattr(self.ctx, 'b2w'):
                self.ctx.b2w.outputs.remote_folder._clean()
            if hasattr(self.ctx, 'a2f'):
                self.ctx.a2f.outputs.remote_folder._clean()