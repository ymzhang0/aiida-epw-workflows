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
                builder.restart.restart_mode = orm.EnumData(RestartType.RESTART_ISO)
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

    def prepare_process(self):
        """Prepare the process for the current interpolation distance."""
        
        inputs = self.ctx.inputs

        
    def inspect_process(self):
        """Verify that the epw.x workflow finished successfully."""
        intp = self.ctx.intp

        if not intp.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ISO

        inputs = {
                'max_eigenvalue': self.ctx.iso.outputs.max_eigenvalue,
                'metadata': {
                    'call_link_label': 'calculate_iso_tc'
                }
            }
        Tc_iso = calculate_iso_tc(**inputs)
        self.ctx.Tc_iso = Tc_iso

    def results(self):
        """TODO"""
        
        self.out('parameters', self.ctx.intp.outputs.output_parameters)
        self.out('a2f', self.ctx.intp.outputs.a2f)
        self.out('max_eigenvalue', self.ctx.intp.outputs.max_eigenvalue)
        self.out('Tc_iso', self.ctx.Tc_iso)
        self.out('remote_folder', self.ctx.intp.outputs.remote_folder)
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

