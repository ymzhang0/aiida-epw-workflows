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
    __KPOINTS_GAMMA = [1, 1, 1]
    
    _INTP_NAMESPACE = 'iso'
    
    _frozen_io_parameters = {
        'INPUTEPW': {
            'elph': False,
            'ep_coupling': False,
            'epwread': True,
            'epwwrite': False,
            'ephwrite': False,
            'restart': True,
        },
    }
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    _defaults_parameters = {
        'INPUTEPW': {
            'degaussq'   : 0.05,
            'degaussw'   : 0.01,
            'eps_acustic' : 1,
            'iverbosity' : 1,
            'fsthick'    : 0.8,
            'nqstep'     : 500,
            'nsiter'     : 500,
            'wscut'      : 0.5,
            'broyden_beta' : 0.4,
            'conv_thr_iaxis' : 1e-2,
            'nstemp'     : 40,
            'temps'      : '1 40',
        }
    }
    
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
            cls.run_iso,
            cls.inspect_iso,
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
        return files(protocols) / 'iso.yaml'
    
    @classmethod
    def get_builder_from_protocol(
            cls, 
            codes, 
            structure, 
            parent_folder_epw=None,
            protocol=None, 
            overrides=None, 
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :TODO:
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()
        builder.structure = structure


        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def inspect_epw(self):
        """Verify that the epw.x workflow finished successfully."""
        epw_workchain = self.ctx.epw

        if not epw_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {epw_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_EPW

    def run_iso(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        
        inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='iso'))
        parameters = inputs.parameters.get_dict()
        
        if self.ctx.epw.process_label == 'EpwBaseWorkChain':
            epw_calcjob = self.ctx.epw.base.links.get_outgoing(link_label_filter='epw').first().node
            parent_folder = self.ctx.epw.outputs.epw_folder
            create_qpoints_from_distance = self.ctx.epw.base.links.get_outgoing(link_label_filter='create_qpoints_from_distance').first().node

            qpoints = create_qpoints_from_distance.outputs.result
            kpoints = orm.KpointsData()
            kpoints.set_kpoints_mesh([
                v * self.ctx.epw.inputs.kpoints_factor_nscf.value 
                for v in qpoints.get_kpoints_mesh()[0]
                ])

            inputs.kpoints = kpoints
            inputs.qpoints = qpoints
        elif self.ctx.epw.process_label == 'EpwA2fWorkChain':
            epw_calcjob = self.ctx.epw.base.links.get_outgoing(link_label_filter='a2f').first().node
            parent_folder = epw_calcjob.outputs.remote_folder
            for namespace, _parameters in self._frozen_io_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value
            
            inputs.kpoints = epw_calcjob.inputs.kpoints
            inputs.qpoints = epw_calcjob.inputs.qpoints
        else:
            raise ValueError("`epw` or `a2f` workchain not found in `EpwWorkChain`")
        
        epw_parameters = epw_calcjob.inputs.parameters.get_dict()
        
        for namespace, _parameters in self._defaults_parameters.items():
            for keyword, value in _parameters.items():
                parameters[namespace][keyword] = value
        for namespace, keyword in self._blocked_keywords:
            if keyword in epw_parameters[namespace]:
                parameters[namespace][keyword] = epw_parameters[namespace][keyword]
        

        inputs.parameters = orm.Dict(parameters)
        inputs.parent_folder_epw = parent_folder

        inputs.kfpoints = self.ctx.kfpoints
        inputs.qfpoints = self.ctx.qfpoints

        try:
            settings = inputs.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = ['out/aiida.dos', 'aiida.a2f*', 'aiida.phdos*']
        inputs.settings = orm.Dict(settings)

        inputs.metadata.call_link_label = self._INTP_NAMESPACE
        calcjob_node = self.submit(EpwCalculation, **inputs)

        self.report(f'launching `iso` with PK {calcjob_node.pk}')

        return ToContext(iso=calcjob_node)

    def inspect_iso(self):
        """Verify that the epw.x workflow finished successfully."""
        iso = self.ctx.iso

        if not iso.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {iso.exit_status}')
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
        
        # self.out('Tc_a2f', self.ctx.Tc_a2f)
        self.out('parameters', self.ctx.iso.outputs.output_parameters)
        self.out('a2f', self.ctx.iso.outputs.a2f)
        self.out('max_eigenvalue', self.ctx.iso.outputs.max_eigenvalue)
        self.out('Tc_iso', self.ctx.Tc_iso)
        self.out('remote_folder', self.ctx.iso.outputs.remote_folder)
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

