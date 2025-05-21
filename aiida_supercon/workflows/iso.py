# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from .epw import EpwWorkChain
from .a2f import A2fWorkChain
from aiida.engine import calcfunction

from scipy.interpolate import interp1d
import numpy

load_profile()


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

class IsoWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute the Allen-Dynes critical temperature."""
    __KPOINTS_GAMMA = orm.KpointsData()
    __KPOINTS_GAMMA.set_kpoints_mesh([1, 1, 1])
    
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

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('qfpoints', required=False, valid_type=orm.KpointsData)
        spec.input('qfpoints_distance', required=False, valid_type=orm.Float)
        spec.input('kfpoints_factor', valid_type=orm.Int)
        spec.input('parent_epw_folder', required=False, valid_type=(orm.RemoteData, orm.RemoteStashFolderData))
        spec.expose_inputs(
            EpwWorkChain, namespace='epw', exclude=(
                'clean_workdir', 'structure'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the interpolation `EpwCalculation`s.'
            }
        )
        spec.expose_inputs(
            EpwCalculation, namespace='iso', exclude=(
                'kpoints',
                'qpoints',
                'kfpoints', 
                'qfpoints',
                'parent_folder_ph', 
                'parent_folder_nscf',
                'parent_folder_epw',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the isotropic `EpwCalculation`.'
            }
        )
        spec.inputs.validator = cls.validate_inputs
        spec.outline(
            cls.setup,
            if_(cls.should_run_epw)(
                cls.run_epw,
                cls.inspect_epw,
            ),
            cls.generate_reciprocal_points,
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
    def validate_inputs(cls, value, port_namespace):  # pylint: disable=unused-argument
        """Validate the top level namespace."""

        if not ('qfpoints_distance' in port_namespace or 'qfpoints' in port_namespace):
            return "Neither `qfpoints` nor `qfpoints_distance` were specified."

        if not ('parent_epw_folder' in port_namespace or 'epw' in port_namespace):
            return "Only one of `parent_epw_folder` or `epw` can be accepted."

    @classmethod
    def get_builder_from_protocol(
            cls, 
            codes, 
            structure, 
            parent_epw_folder=None,
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

        if not parent_epw_folder:
            builder.epw = EpwWorkChain.get_builder_from_protocol(
                codes=codes,
                structure=structure,
                protocol=protocol,
                overrides=inputs.get('epw', None),
                **kwargs
            )
            
        else:
            # TODO: Add cheeck to make sure epw_folder is on same computer as epw_code
            builder.parent_epw_folder = parent_epw_folder

        builder.qfpoints_distance = orm.Float(inputs['qfpoints_distance'])
        builder.kfpoints_factor = orm.Int(inputs['kfpoints_factor'])
        
        epw_inputs = inputs.get('iso', None) 
        
        epw_builder = EpwCalculation.get_builder()
        epw_builder.code = codes['epw']
        epw_builder.metadata = epw_inputs['metadata']
        if 'settings' in epw_inputs:
            epw_builder.settings = orm.Dict(epw_inputs['settings'])

        epw_builder.parameters = orm.Dict(epw_inputs.get('parameters', {}))
        
        builder.iso  = epw_builder

        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        self.ctx.degaussq = None

    @staticmethod
    def get_kpoints_from_inputs(
        inputs, 
        structure, 
        call_link_label = 'create_kpoints_from_distance',
        kpoints_arg = 'kpoints',
        kpoints_distance_arg = 'kpoints_distance'
    ):
        if hasattr(inputs, kpoints_arg):
            kpoints = inputs.get(kpoints_arg)
        elif hasattr(inputs, kpoints_distance_arg):
            _inputs = {
                'structure': structure,
                'distance': inputs.get(kpoints_distance_arg),
                'force_parity': orm.Bool(False),
                'metadata': {
                    'call_link_label': call_link_label
                }
            }
            kpoints = create_kpoints_from_distance(**_inputs)
        else:
            raise ValueError("Kpoints no specified in `parent_epw_folder`")
        
        return kpoints

    def generate_reciprocal_points(self):
        """Generate the qpoints and kpoints meshes for the interpolation."""

        if self.ctx.epw.process_label == 'EpwWorkChain':
            qfpoints = self.get_kpoints_from_inputs(
                self.inputs, 
                self.inputs.structure, 
                'create_qfpoints_from_distance',
                kpoints_arg = 'qfpoints',
                kpoints_distance_arg = 'qfpoints_distance'
                )

            kfpoints = orm.KpointsData()
            kfpoints.set_kpoints_mesh([
                v * self.inputs.kfpoints_factor.value 
                for v in qfpoints.get_kpoints_mesh()[0]
                ])
        elif self.ctx.epw.process_label == 'A2fWorkChain':
            create_qfpoints_from_distance = self.ctx.epw.base.links.get_outgoing(link_label_filter='create_qfpoints_from_distance').first().node
            qfpoints = create_qfpoints_from_distance.outputs.result
            kfpoints = orm.KpointsData()
            kfpoints.set_kpoints_mesh([
                v * self.ctx.epw.inputs.kfpoints_factor.value 
                for v in qfpoints.get_kpoints_mesh()[0]
                ])
        else:
            raise ValueError("`epw` workchain not found in `EpwWorkChain`")

        self.ctx.qfpoints = qfpoints
        self.ctx.kfpoints = kfpoints

    def should_run_epw(self):
        """Check if the epw loop should continue or not."""
        
        if hasattr(self.inputs, 'epw'):
            return True
        elif hasattr(self.inputs, 'parent_epw_folder'):
            parent_wc = self.inputs.parent_epw_folder.creator.caller
            if parent_wc.process_class in (EpwWorkChain, A2fWorkChain):
                self.report(f'Reading from parent epw folder')
                self.ctx.epw = parent_wc            
            else:
                raise ValueError("`parent_epw_folder` must be a `RemoteData` node from an `EpwWorkChain` or `A2FWorkChain`.")
            
            return False
        else:
            raise ValueError("No `epw` or `parent_epw_folder` specified in inputs")

    def run_epw(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        inputs = AttributeDict(self.exposed_inputs(EpwWorkChain, namespace='epw'))

        inputs.structure = self.inputs.structure
        
        inputs.metadata.call_link_label = 'epw'
        workchain_node = self.submit(EpwWorkChain, **inputs)

        self.report(f'launching `epw` with PK {workchain_node.pk}')

        return ToContext(epw=workchain_node)

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
        
        if self.ctx.epw.process_label == 'EpwWorkChain':
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
        elif self.ctx.epw.process_label == 'A2fWorkChain':
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

        inputs.metadata.call_link_label = 'iso'
        calcjob_node = self.submit(EpwCalculation, **inputs)

        self.report(f'launching iso `epw` with PK {calcjob_node.pk}')

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

