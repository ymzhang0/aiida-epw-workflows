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

load_profile()

@calcfunction
def split_list(list_node: orm.List) -> dict:
    return {f'el_{no}': orm.Float(el) for no, el in enumerate(list_node.get_list())}

class EpwSuperConWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute the electron-phonon coupling."""
    __KPOINTS_GAMMA = [1, 1, 1]
    
    _excluded_epw_inputs = (
        'structure', 'clean_workdir', 'w90_chk_to_ukk_script')
    
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

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('qfpoints_distance', valid_type=orm.Float)
        spec.input('kfpoints_factor', valid_type=orm.Int)
        spec.input('parent_folder_epw', required=False, valid_type=(orm.RemoteData, orm.RemoteStashFolderData))
        spec.input('interpolation_distance', required=False, valid_type=(orm.Float, orm.List))
        spec.input('convergence_threshold', required=False, valid_type=orm.Float)

        spec.expose_inputs(
            EpwBaseWorkChain, namespace='epw', exclude=cls._excluded_epw_inputs,
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the interpolation `EpwCalculation`s.'
            }
        )
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
            cls.validate_inputs,
            cls.setup,
            if_(cls.should_run_epw)(
                cls.run_epw,
                cls.inspect_epw,
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
            parent_folder_epw=None,
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
        builder.qfpoints_distance = orm.Float(inputs['qfpoints_distance'])
        builder.kfpoints_factor = orm.Int(inputs['kfpoints_factor'])
        
        
        for (epw_namespace, epw_workchain_class) in (
            ('a2f', EpwA2fWorkChain),
            ('iso', EpwIsoWorkChain),
            ('aniso', EpwAnisoWorkChain),
        ):
            epw_inputs = inputs.get(epw_namespace, None)
            epw_builder = epw_workchain_class.get_builder()
            epw_builder.pop('epw')
            epw_builder[epw_namespace]['code'] = codes['epw']
            epw_builder[epw_namespace]['metadata'] = epw_inputs[epw_namespace]['metadata']
            if 'settings' in epw_inputs[epw_namespace]:
                epw_builder[epw_namespace]['settings'] = orm.Dict(epw_inputs[epw_namespace]['settings'])

            epw_builder[epw_namespace]['parameters'] = orm.Dict(epw_inputs[epw_namespace].get('parameters', {}))

            builder[epw_namespace] = epw_builder

        if not parent_folder_epw:
            builder.epw = EpwBaseWorkChain.get_builder_from_protocol(
                *args,
                overrides=inputs.get('epw', None),
                **kwargs
            )
            
        else:
            # TODO: Add check to make sure epw_folder is on same computer as epw_code
            builder.parent_folder_epw = parent_folder_epw

            
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def validate_inputs(self):
        """Validate the inputs."""
        if hasattr(self.inputs, 'parent_folder_epw') and hasattr(self.inputs, 'epw'):
            raise ValueError("Only one of `parent_folder_epw` or `epw` can be accepted.")

    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        
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

    @staticmethod
    def get_kpoints_from_workchain(workchain):
        """Get the kpoints from the workchain."""
        if workchain.process_class is EpwWorkChain:
            epw_calcjob = workchain.base.links.get_outgoing(link_label_filter='epw').first().node
            create_qpoints_from_distance = workchain.base.links.get_outgoing(link_label_filter='create_qpoints_from_distance').first().node

            qpoints = create_qpoints_from_distance.outputs.result
            kpoints = orm.KpointsData()
            kpoints.set_kpoints_mesh([
                v * workchain.inputs.kpoints_factor_nscf.value 
                for v in qpoints.get_kpoints_mesh()[0]
                ])

        elif workchain.process_class in (A2fWorkChain, IsoWorkChain):
            epw_calcjob = workchain.base.links.get_outgoing(link_label_filter='a2f').first().node
            kpoints = epw_calcjob.inputs.kpoints
            qpoints = epw_calcjob.inputs.qpoints
        else:
            raise ValueError("`workchain` must be an `EpwWorkChain` or `A2fWorkChain` or `IsoWorkChain`.")

        return kpoints, qpoints
    
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

    def should_run_epw(self):
        """Check if the epw loop should continue or not."""
        
        if hasattr(self.inputs, 'epw'):
            self.report('Will run `epw` workchain from scratch')
            return True
        elif hasattr(self.inputs, 'parent_folder_epw'):
            parent_epw_wc = self.inputs.parent_folder_epw.creator.caller
            self.report(f'Will restart from previous `epw` workchain<{parent_epw_wc.pk}>')
            if parent_epw_wc.process_class not in (EpwBaseWorkChain, EpwA2fWorkChain, EpwIsoWorkChain, EpwAnisoWorkChain):
                raise ValueError("`parent_folder_epw` must be a `RemoteData` node from an `EpwWorkChain`.")
            
            self.ctx.epw = parent_epw_wc            
            
            return False
        else:
            raise ValueError("No `epw` or `parent_epw_folder` specified in inputs")

    def run_epw(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        inputs = AttributeDict(self.exposed_inputs(EpwBaseWorkChain, namespace='epw'))

        inputs.structure = self.inputs.structure
        
        inputs.metadata.call_link_label = 'epw'
        calcjob_node = self.submit(EpwBaseWorkChain, **inputs)

        self.report(f'launching `epw` with PK {calcjob_node.pk}')

        return ToContext(epw=calcjob_node)

    def inspect_epw(self):
        """Verify that the epw.x workflow finished successfully."""
        epw_calculation = self.ctx.epw

        if not epw_calculation.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {epw_calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_EPW

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

    def run_a2f(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        inputs = AttributeDict(self.exposed_inputs(EpwA2fWorkChain, namespace='a2f'))

        inputs.structure = self.inputs.structure
        inputs.parent_folder_epw = self.ctx.epw.outputs.epw_folder
        inputs.qfpoints_distance = self.ctx.qfpoints_distance
        inputs.kfpoints_factor = self.inputs.kfpoints_factor
        inputs.metadata.call_link_label = 'a2f'
        
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)

        self.report(f'launching `a2f` with PK {workchain_node.pk}')

        return ToContext(a2f=workchain_node)

    def inspect_a2f(self):
        """Verify that the a2f workflow finished successfully."""
        a2f_workchain = self.ctx.a2f

        if not a2f_workchain.is_finished_ok:
            self.report(f'`a2f` failed with exit status {a2f_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_A2F
        
    def should_run_iso(self):
        """Check if the isotropic loop should continue or not."""
        if hasattr(self.inputs, 'iso'):
            self.report('Will run `iso` workchain from scratch')
            return True
        else:
            return False

    def run_iso(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""

        inputs = AttributeDict(self.exposed_inputs(EpwIsoWorkChain, namespace='iso'))

        inputs.structure = self.inputs.structure
        inputs.parent_folder_epw = self.ctx.a2f.outputs.remote_folder
        inputs.qfpoints_distance = self.ctx.qfpoints_distance
        inputs.kfpoints_factor = self.inputs.kfpoints_factor

        inputs.metadata.call_link_label = 'iso'
        
        workchain_node = self.submit(EpwIsoWorkChain, **inputs)

        self.report(f'launching isotropic `epw` with PK {workchain_node.pk}')

        return ToContext(iso=workchain_node)

    def inspect_iso(self):
        """Verify that the epw.x workflow finished successfully."""
        iso = self.ctx.iso

        if not iso.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {iso.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_EPW_ISO

    def should_run_aniso(self):
        """Check if the anisotropic loop should continue or not."""
        
        Tc_iso = self.ctx.iso.outputs.Tc_iso
        if hasattr(self.inputs, 'aniso') and Tc_iso > 5.0:
            self.report('Will run `aniso` workchain from scratch')
            return True
        else:
            return False
        
    def run_aniso(self):
        """Run the aniso ``epw.x`` calculation."""
        
        inputs = AttributeDict(self.exposed_inputs(EpwAnisoWorkChain, namespace='aniso'))

        inputs.structure = self.inputs.structure
        inputs.parent_folder_epw = self.ctx.iso.outputs.remote_folder
        inputs.qfpoints_distance = self.ctx.qfpoints_distance
        inputs.kfpoints_factor = self.inputs.kfpoints_factor

        inputs.metadata.call_link_label = 'aniso'

        workchain_node = self.submit(EpwAnisoWorkChain, **inputs)
        self.report(f'launching anisotropic `epw` {workchain_node.pk}')

        return ToContext(aniso=workchain_node)
    
    def inspect_aniso(self):
        """Verify that the aniso epw.x workflow finished successfully."""
        aniso_workchain = self.ctx.aniso

        if not aniso_workchain.is_finished_ok:
            self.report(f'Anisotropic `epw.x` failed with exit status {aniso_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_EPW_ANISO
        
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

