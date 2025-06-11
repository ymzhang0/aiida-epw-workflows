# -*- coding: utf-8 -*-
"""Work chain to compute the electron-phonon coupling.

This work chain is refined to accept the parent folders of the Wannier90 and Phonon work chains.

Firstly it will validate the parent folders.

- If the phonon parent folders are valid, it will use the qpoints from the phonon parent folders.

Then it will check if the wannier90 parent folders are valid. 

- If so, it will check if the kpoints of the wannier90 work chain are compatible with the qpoints of the phonon work chain. 

- If they are not compatible, it will re-generate the kpoints of the wannier90 work chain based on the qpoints of the phonon work chain. Then it will re-run the wannier90 work chain.

- If the wannier90 parent folders are not valid, it will run the wannier90 work chain from scratch.

If none of the parent folders are provided or are not valid, it will run the Wannier90 and Phonon work chains from scratch.

The first step is to compute the electron-phonon (dvscf) on a coarse grid. I think 0.3A^-1 q-point grid should be good but you can use the same k-point grid (as opposed to twice) so it should be cheaper. This step should be done only once

For this step you need to run 

pw.x < scf.in
ph.x < ph.in

Once the calculation is done, you need to rename and gather the results in a "save" folder
This is done with QE/EPW/bin/pp.py script
You just run it as python3 QE/EPW/bin/pp.py
you need to provide the prefix name to the script
What is very important is that the "save" folder is saved in the AiiDA framework
This will be the biggest thing to save and is typically ~ 500 mb to 1 Gb

We can discuss this if we want to keep it or not but if possible, I would say yes
Ok then the EPW step starts

2. Find initial wannier projection

pw.x < scf.in
pw.x < nscf.in
projwfc.x
+ wannier steps from Junfeng workflow?
-> What we need here is the block of inputs provided to epw.in linked to the wannierization 
(wdata(1) = "<wannier inputs>" input variable; is basically a vector of input variables that are directly passed to wannier.x)

e.g.

 wdata(1) = 'bands_plot = .true.'
 wdata(2) = 'begin kpoint_path'
 wdata(3) = 'G 0.00 0.00 0.00 M 0.50 0.00 0.00'
 wdata(4) = 'M 0.50 0.00 0.00 K 0.333333333333 0.333333333333 0.00'
 wdata(5) = 'K 0.333333333333 0.333333333333 0.00 G 0.0 0.0 0.00'
 wdata(6) = 'end kpoint_path'
 wdata(7) = 'bands_plot_format = gnuplot'
 wdata(8) = 'dis_num_iter      = 5000'
 wdata(9) = 'num_print_cycles  = 10'
 wdata(10) = 'dis_mix_ratio     = 1.0'
 wdata(11) = 'conv_tol = 1E-12'
 wdata(12) = 'conv_window = 4'


3. EPW step (in a different folder). You just need to do a soft link to the "save" folder from the above step

You need to do
pw.x <scf.in   (Could potentially be the same as in step 2)
pw.x <nscf.in -> JQ: we can simply use these from the wannier workflow

epw.x < epw1.in 

At the end of this step, you have the electron-phonon matrix element in real space
This needs to be stored for sure
The most important and only big file is "PREFIX..epmatwp"
For TiO, this is 872 Mb ...
From this file you can interpolate to any fine grid density

I mean use for next calculation but you may want to keep it in order to do more convergence later
for example if you find that a 40x40x40 grid is not enough
you dont want to redo step 1 and 2 to get 60x60x60

Files to save:

ln -s ../epw8-conv1/crystal.fmt
ln -s ../epw8-conv1/epwdata.fmt
ln -s ../epw8-conv1/<prefix>.bvec
ln -s ../epw8-conv1/<prefix>.chk
ln -s ../epw8-conv1/<prefix>.kgmap
ln -s ../epw8-conv1/<prefix>.kmap
ln -s ../epw8-conv1/<prefix>.mmn
ln -s ../epw8-conv1/<prefix>.nnkp
ln -s ../epw8-conv1/<prefix>.ukk
ln -s ../epw8-conv1/<prefix>.epmatwp (Note: quite big file!)
ln -s ../epw8-conv1/vmedata.fmt
ln -s ../epw8-conv1/dmedata.fmt
ln -s ../epw8-conv1/save (Is basically the save folder from step 1)

4. EPW interpolation to get Eliashberg Tc

epw.x < epw2.in
epw2.in

and basically here you can change the fine grid in epw2.in to converge things
This run can be done in a different folder but you need to soft link a number of files from the previous calculation 2.

"""
from pathlib import Path

from aiida import orm
from aiida.common import AttributeDict

import logging


from aiida.engine import PortNamespace, ProcessBuilder, WorkChain, ToContext, if_, while_
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from aiida_wannier90_workflows.workflows import Wannier90BaseWorkChain, Wannier90BandsWorkChain, Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.workflows.optimize import validate_inputs as validate_inputs_w90_intp
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints
from aiida_wannier90_workflows.common.types import WannierProjectionType

from .utils.overrides import get_overrides_from_w90_workchain, update_epw_from_w90_overrides, update_epw_from_w90_intp, get_overrides_from_ph_base, update_epw_from_ph_overrides, update_epw_from_ph_base
from .utils.kpoints import is_compatible

def validate_inputs(inputs, ctx=None):  # pylint: disable=unused-argument
    """Validate the inputs of the entire input namespace of `Wannier90OptimizeWorkChain`."""

    if hasattr(inputs, 'w90_intp'):
        print('validate_inputs_w90_intp')
        validate_inputs_w90_intp(inputs)
    else:
        print('validate nothing')
        
    return None
        
class EpwIntpWorkChain(ProtocolMixin, WorkChain):
    """Main work chain to start calculating properties using EPW.

    Has support for both the selected columns of the density matrix (SCDM) and
    (projectability-disentangled Wannier function) PDWF projection types.
    """
        
    _KPOINTS_GAMMA = [1, 1, 1]
    
    SOURCE_LIST = {
        'ph_base':[
            'DYN_MAT/dynamical-matrix-*',
            'out/_ph0/aiida.dvscf1',
            'out/_ph0/aiida.q_*/aiida.dvscf1', 
            ],
        'epw': [
            'crystal.fmt', 
            'dmedata.fmt', 
            'epwdata.fmt', 
            'selecq.fmt', 
            'dmedata.fmt', 
            'aiida.kgmap', 
            'aiida.kmap', 
            'aiida.ukk', 
            'out/aiida.epmatwp', 
            'save'
            ]
        }
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('w90_chk_to_ukk_script', valid_type=(orm.RemoteData, orm.SinglefileData))
        spec.input('kpoints_factor_nscf', valid_type=orm.Int)
        spec.input('qpoints_distance', valid_type=orm.Float, required=False)
        
        spec.input('qpoints', valid_type=orm.KpointsData, required=False)
        
        # w90_ns = PortNamespace(non_db=True)
        # w90_ns['kpoints_nscf'] = Port(valid_type=KpointsData, required=False)
        # w90_ns['parent_folder_scf'] = Port(valid_type=RemoteData, required=False)

        # spec.input('w90_overrides', w90_ns)

        spec.input_namespace(
            'w90_overrides',
            required=False,
            help='Overrides extracted from the previous Wannier WorkChain'
        )
        spec.input('w90_overrides.kpoints_nscf', valid_type=orm.KpointsData, required=False)
        spec.input('w90_overrides.parent_folder_scf', valid_type=orm.RemoteData, required=False)
        spec.input('w90_overrides.parent_folder_nscf', valid_type=orm.RemoteData, required=False)
        spec.input('w90_overrides.parent_folder_chk', valid_type=orm.RemoteData, required=False)
        spec.input('w90_overrides.parameters', valid_type=orm.Dict, required=False)

        spec.input_namespace(
            'ph_base_overrides',
            required=False,
            help='Overrides extracted from the previous PhBase WorkChain'
        )
        spec.input('ph_base_overrides.parent_folder_ph', valid_type=(orm.RemoteData, orm.RemoteStashFolderData), required=False)
        spec.input('ph_base_overrides.qpoints', valid_type=orm.KpointsData, required=False)

        spec.inputs.validator = validate_inputs

        spec.expose_inputs(
            Wannier90OptimizeWorkChain, 
            namespace='w90_intp', 
            exclude=(
                'clean_workdir', 'nscf.kpoints', 'nscf.kpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `Wannier90OptimizeWorkChain/Wannier90BandsWorkChain`.'
            }
        )
                
        spec.expose_inputs(
            PhBaseWorkChain, 
            namespace='ph_base',
            exclude=(
                'clean_workdir', 'ph.parent_folder', 'qpoints', 'qpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` that does the `ph.x` calculation.'
            }
        )
        spec.expose_inputs(
            EpwCalculation, 
            namespace='epw', 
            exclude=(
                'parent_folder_ph', 'parent_folder_nscf', 'parent_folder_chk', 'kpoints', 'qpoints', 'kfpoints', 'qfpoints'
            ),
            namespace_options={
                'help': 'Inputs for the `EpwCalculation`.'
            }
        )
        
        spec.output('retrieved', valid_type=orm.FolderData)
        spec.output('epw_folder', valid_type=orm.RemoteStashFolderData)

        spec.outline(
            cls.generate_reciprocal_points,
            if_(cls.should_run_wannier90)(
                cls.run_wannier90,
                cls.inspect_wannier90,
            ),
            if_(cls.should_run_ph)(
                cls.run_ph,
                cls.inspect_ph,
            ),
            cls.run_epw,
            cls.inspect_epw,
            cls.results,
        )
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_PHONON',
            message='The electron-phonon `PhBaseWorkChain` sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_WANNIER90',
            message='The `Wannier90BandsWorkChain` sub process failed')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_FAILED_EPW',
            message='The `EpwWorkChain` sub process failed')
        spec.exit_code(406, 'ERROR_SCF_PARENT_FOLDER_NOT_FOUND',
            message='The `scf` parent folder was not found')
        spec.exit_code(407, 'ERROR_KPOINTS_QPOINTS_NOT_COMPATIBLE',
            message='The kpoints and qpoints are not compatible')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / 'intp.yaml'

        
    @staticmethod
    def set_target_base(
        inputs,
        target_base_prefix,
        ):
        
        if 'stash' not in inputs.metadata['options']:
            computer = inputs.code.computer
            if computer.transport_type == 'core.local':
                target_basepath = Path(computer.get_workdir(), f'{target_base_prefix}-stash').as_posix()
            elif computer.transport_type == 'core.ssh':
                target_basepath = Path(
                    computer.get_workdir().format(username=computer.get_configuration()['username']), f'{target_base_prefix}-stash'
                ).as_posix()
                
            inputs.metadata['options']['stash'] = {
                'target_base': target_basepath,
                'source_list': EpwIntpWorkChain.SOURCE_LIST[target_base_prefix]
                }
            
    @classmethod
    def get_builder_from_protocol(
        cls, codes, structure, protocol=None, overrides=None,
        wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        reference_bands=None, bands_kpoints=None,
        w90_intp=None, ph_base=None, epw=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        
        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()
        
        epw_builder = EpwCalculation.get_builder()
        
        epw_builder.code = codes['epw']
        epw_inputs = inputs.get('epw', None)
        
        epw_builder.parameters = orm.Dict(epw_inputs['parameters'])
        epw_builder.metadata = epw_inputs['metadata']
        epw_builder.settings = orm.Dict(epw_inputs['settings'])

        
        # Here we check the previous wannier90 workchain instead of checking
        # it inside the workchain because we can't pass the `WorkChain` as an
        # input.
        
        if w90_intp:
            w90_overrides = get_overrides_from_w90_workchain(w90_intp)
            builder.w90_overrides.update(w90_overrides)
            builder.pop('w90_intp')
        else:
            # Only create w90_intp inputs if not provided
            w90_intp_inputs = inputs.get('w90_intp', {})
            pseudo_family = w90_intp_inputs.pop('pseudo_family', None)

            w90_intp = Wannier90OptimizeWorkChain.get_builder_from_protocol(
                codes=codes,
                structure=structure,
                overrides=w90_intp_inputs,
                pseudo_family=pseudo_family,
                projection_type=wannier_projection_type,
                reference_bands=reference_bands,
                bands_kpoints=bands_kpoints,
            )
            ## TODO: It's ugly but if we don't pop these two, the validation will fail
            w90_intp.pop('projwfc', None)
            w90_intp.pop('open_grid', None)
            
            builder.w90_intp = w90_intp
            
            # TODO: Only for testing, will remove later
            builder.w90_intp.optimize_disproj = orm.Bool(False)

            ## TODO: It's ugly but if we don't pop these two, these namespaces will be 
            # regenerated
            builder.w90_intp.pop('projwfc', None)
            builder.w90_intp.pop('open_grid', None)
                        
        # TODO: Check we can restart phonon from parent_ph_folder.
        if ph_base:
            ph_base_overrides = get_overrides_from_ph_base(ph_base)
            builder.ph_base_overrides.update(ph_base_overrides)
            builder.pop('ph_base')
        else:
            ph_base_inputs = inputs.get('ph_base', {})
            ph_base = PhBaseWorkChain.get_builder_from_protocol(
                codes['ph'],
                protocol=protocol, 
                overrides=ph_base_inputs, 
                **kwargs
                )

            builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])
            builder.ph_base = ph_base
        # Set kpoints and qpoints before checking compatibility


        builder.epw = epw_builder
        builder.structure = structure
        builder.kpoints_factor_nscf = orm.Int(inputs['kpoints_factor_nscf'])
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder._inputs(prune=True)

        return builder

    def generate_reciprocal_points(self):
        """Generate the reciprocal points."""

        self.report('Generating q-points and k-points')
        if hasattr(self.inputs, 'ph_base'):
            inputs = {
                'structure': self.inputs.structure,
                'distance': self.inputs.qpoints_distance,
                'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_qpoints_from_distance'
                }
            }
            
            qpoints = create_kpoints_from_distance(**inputs)
        else:
            qpoints = self.inputs.ph_base_overrides.get('qpoints')
            
        if hasattr(self.inputs, 'w90_intp'):
            qpoints_mesh = qpoints.get_kpoints_mesh()[0]
            kpoints_nscf = orm.KpointsData()
            kpoints_nscf.set_kpoints_mesh([v * self.inputs.kpoints_factor_nscf.value for v in qpoints_mesh])
        else:
            kpoints_nscf = self.inputs.w90_overrides['kpoints_nscf']
        
        if not is_compatible(kpoints_nscf, qpoints):
            self.exit_codes.ERROR_KPOINTS_QPOINTS_NOT_COMPATIBLE
        
        self.report(f'qpoints: {qpoints.get_kpoints_mesh()}')
        self.report(f'kpoints_nscf: {kpoints_nscf.get_kpoints_mesh()}')
        self.ctx.qpoints = qpoints
        self.ctx.kpoints_nscf = kpoints_nscf

    def should_run_wannier90(self):
        """Check if the wannier90 workflow should be run."""
        # The only reason that we run the wannier90 workflow is 
        # if w90_intp is in the inputs port.
        
        return 'w90_intp' in self.inputs

    def run_wannier90(self):
        """Run the wannier90 workflow."""
        
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace='w90_intp')
        )
        
        inputs.metadata.call_link_label = 'w90_intp'
        
        set_kpoints(inputs, self.ctx.kpoints_nscf, Wannier90OptimizeWorkChain)

        workchain_node = self.submit(Wannier90OptimizeWorkChain, **inputs)
        self.report(f'launching wannier90 work chain {workchain_node.pk}')

        return ToContext(workchain_w90_intp=workchain_node)

    def inspect_wannier90(self):
        """Verify that the wannier90 workflow finished successfully."""
        workchain = self.ctx.workchain_w90_intp

        if not workchain.is_finished_ok:
            self.report(f'`Wannier90BandsWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90
    
    def should_run_ph(self):
        """Check if the phonon workflow should be run."""
        # The only reason that we run the phonon workflow is 
        # if ph_base is in the inputs port.
        return 'ph_base' in self.inputs

    def run_ph(self):
        """Run the `PhBaseWorkChain`."""
        inputs = AttributeDict(self.exposed_inputs(PhBaseWorkChain, namespace='ph_base'))
        if hasattr(self.inputs, 'w90_overrides'):
            inputs.ph.parent_folder = self.inputs.w90_overrides.get('parent_folder_scf', None)
        elif hasattr(self.ctx, 'workchain_w90_intp'):
            inputs.ph.parent_folder = self.ctx.workchain_w90_intp.outputs.scf.remote_folder
        else:
            self.exit_codes.ERROR_SCF_PARENT_FOLDER_NOT_FOUND
            
        inputs.qpoints = self.ctx.qpoints
        inputs.metadata.call_link_label = 'ph_base'
        
        self.set_target_base(inputs.ph, inputs.metadata.call_link_label)

        workchain_node = self.submit(PhBaseWorkChain, **inputs)
        
        self.report(f'launching `ph` {workchain_node.pk}')

        return ToContext(workchain_ph=workchain_node)

    def inspect_ph(self):
        """Verify that the `PhBaseWorkChain` finished successfully."""
        workchain = self.ctx.workchain_ph

        if not workchain.is_finished_ok:
            self.report(f'Electron-phonon PhBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PHONON

    def run_epw(self):
        """Run the `epw.x` calculation."""       
        
        inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='epw'))

        inputs.kpoints = self.ctx.kpoints_nscf
        inputs.qpoints = self.ctx.qpoints
        
        fpoints = orm.KpointsData()
        fpoints.set_kpoints_mesh(self._KPOINTS_GAMMA)
        inputs.kfpoints = fpoints
        inputs.qfpoints = fpoints

        if hasattr(self.inputs, 'ph_base_overrides'):
            update_epw_from_ph_overrides(inputs, self.inputs.ph_base_overrides)
        else:
            update_epw_from_ph_base(inputs, self.ctx.workchain_ph)
            
        if hasattr(self.inputs, 'w90_overrides'):
            update_epw_from_w90_overrides(inputs, self.inputs.w90_chk_to_ukk_script, self.inputs.w90_overrides)
        else:
            update_epw_from_w90_intp(inputs, self.inputs.w90_chk_to_ukk_script, self.ctx.workchain_w90_intp)
            
        inputs.metadata.call_link_label = 'epw'
        self.set_target_base(inputs, 'epw')
        
        calcjob_node = self.submit(EpwCalculation, **inputs)
        self.report(f'launching `epw` {calcjob_node.pk}')

        return ToContext(calcjob_epw=calcjob_node)

    def inspect_epw(self):
        """Verify that the `epw.x` calculation finished successfully."""
        calcjob = self.ctx.calcjob_epw

        if not calcjob.is_finished_ok:
            self.report(f'`EpwCalculation` failed with exit status {calcjob.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EPW

    def results(self):
        """Add the most important results to the outputs of the work chain."""
        self.out('retrieved', self.ctx.calcjob_epw.outputs.retrieved)
        self.out('epw_folder', self.ctx.calcjob_epw.outputs.remote_stash)

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

