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

from aiida.engine import WorkChain, ToContext, if_, while_
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from aiida_wannier90_workflows.workflows import Wannier90BaseWorkChain, Wannier90BandsWorkChain, Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.workflows.optimize import validate_inputs as validate_inputs_bands
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints
from aiida_wannier90_workflows.common.types import WannierProjectionType


class EpwWorkChain(ProtocolMixin, WorkChain):
    """Main work chain to start calculating properties using EPW.

    Has support for both the selected columns of the density matrix (SCDM) and
    (projectability-disentangled Wannier function) PDWF projection types.
    """

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('qpoints_distance', valid_type=orm.Float, default=lambda: orm.Float(0.5))
        spec.input('kpoints_distance_scf', valid_type=orm.Float, default=lambda: orm.Float(0.15))
        spec.input('kpoints_factor_nscf', valid_type=orm.Int, default=lambda: orm.Int(2))
        spec.input('w90_chk_to_ukk_script', valid_type=(orm.RemoteData, orm.SinglefileData))

        spec.input('parent_ph_folder', valid_type=(orm.RemoteStashFolderData, orm.RemoteData), required=False,
            help='PhBaseWorkChain, if provided, will be skipped')

        spec.input('parent_w90_folder', valid_type=(orm.RemoteStashFolderData, orm.RemoteData), required=False,
            help='Wannier90BandsWorkChain or Wannier90OptimizeWorkChain, if provided, will be skipped')

        spec.expose_inputs(
            Wannier90OptimizeWorkChain, namespace='w90_bands', exclude=(
                'structure', 'clean_workdir',
            ),
            namespace_options={
                'help': 'Inputs for the `Wannier90OptimizeWorkChain/Wannier90BandsWorkChain`.'
            }
        )
        spec.inputs['w90_bands'].validator = validate_inputs_bands
        spec.expose_inputs(
            PhBaseWorkChain, namespace='ph_base', exclude=(
                'clean_workdir', 'ph.parent_folder', 'qpoints', 'qpoints_distance'
            ),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` that does the `ph.x` calculation.'
            }
        )
        spec.expose_inputs(
            EpwCalculation, namespace='epw', exclude=(
                'parent_folder_ph', 'parent_folder_nscf', 'kpoints', 'qpoints', 'kfpoints', 'qfpoints'
            ),
            namespace_options={
                'help': 'Inputs for the `EpwCalculation`.'
            }
        )
        spec.output('retrieved', valid_type=orm.FolderData)
        spec.output('epw_folder', valid_type=orm.RemoteStashFolderData)

        spec.outline(
            cls.validate_parent_folders,
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

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / 'epw.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, codes, structure, protocol=None, overrides=None,
        wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        reference_bands=None, bands_kpoints=None,
        parent_w90_folder=None, parent_ph_folder=None,
        **kwargs
        ):
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
        builder.structure = structure
        
        # Check if the wannier90 workflow should be run from scratch
        
        if not parent_w90_folder:
            w90_bands_inputs = inputs.get('w90_bands', {})
            pseudo_family = w90_bands_inputs.pop('pseudo_family', None)
                
            w90_bands = Wannier90OptimizeWorkChain.get_builder_from_protocol(
                structure=structure,
                codes=codes,
                pseudo_family=pseudo_family,
                overrides=w90_bands_inputs,
                projection_type=wannier_projection_type,
                reference_bands=reference_bands,
                bands_kpoints=bands_kpoints,
            )
            w90_bands.pop('structure', None)
            w90_bands.pop('open_grid', None)
            
            if wannier_projection_type == WannierProjectionType.ATOMIC_PROJECTORS_QE:
                w90_bands.pop('projwfc', None)

            builder.w90_bands = w90_bands
            builder.kpoints_distance_scf = orm.Float(inputs['kpoints_distance_scf'])
            builder.kpoints_factor_nscf = orm.Int(inputs['kpoints_factor_nscf'])
            
        else:
            builder.parent_w90_folder = parent_w90_folder

        # Check if the phonon workflow should be run from scratch
        if not parent_ph_folder:
            ph_base_inputs = inputs.get('ph_base', {})
            ph_base = PhBaseWorkChain.get_builder_from_protocol(
                codes['ph'],
                None, 
                protocol, 
                overrides=ph_base_inputs, 
                **kwargs)
            
            ph_base.pop('clean_workdir', None)
            ph_base.pop('qpoints_distance')
            builder.ph_base = ph_base
            builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])
        else:
            builder.parent_ph_folder = parent_ph_folder

        epw_builder = EpwCalculation.get_builder()

        epw_builder.code = codes['epw']
        epw_inputs = inputs.get('epw', None)

        epw_builder.parameters = orm.Dict(epw_inputs['parameters'])

        if 'target_base' not in epw_builder.metadata['options']['stash']:
            epw_computer = codes['epw'].computer
            if epw_computer.transport_type == 'core.local':
                target_basepath = Path(epw_computer.get_workdir(), 'stash').as_posix()
            elif epw_computer.transport_type == 'core.ssh':
                target_basepath = Path(
                    epw_computer.get_workdir().format(username=epw_computer.get_configuration()['username']), 'stash'
                ).as_posix()
            epw_inputs['metadata']['options']['stash']['target_base'] = target_basepath

        epw_builder.metadata = epw_inputs['metadata']
        epw_builder.settings = orm.Dict(epw_inputs['settings'])

        builder.epw = epw_builder
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder


    def validate_parent_folders(self):
        """Validate the parent folders."""
        
        ## TODO: Check if the parent folders are CLEANED
        
        is_w90_parent_folder_valid = False
        is_ph_parent_folder_valid = False

        parent_ph_folder = self.inputs.get('parent_ph_folder', None)
        if (
            parent_ph_folder and
            (not parent_ph_folder.is_cleaned) and
            parent_ph_folder.creator.caller.is_finished_ok and
            parent_ph_folder.creator.caller.process_class is PhBaseWorkChain
        ):
            self.report(f'PhBaseWorkChain[{parent_ph_folder.creator.caller.pk}] finished successfully.')
            self.ctx.workchain_ph = parent_ph_folder.creator.caller
            is_ph_parent_folder_valid = True

        else:
            self.report('PhBaseWorkChain not provided or failed, will run it again.')
            is_ph_parent_folder_valid = False

        parent_w90_folder = self.inputs.get('parent_w90_folder', None)
        
        if (
            parent_w90_folder and
            (not parent_w90_folder.is_cleaned) and
            parent_w90_folder.creator.caller.is_finished_ok and 
            parent_w90_folder.creator.caller.process_class is Wannier90BaseWorkChain
        ):
            self.report(f'Wannier90BaseWorkChain[{parent_w90_folder.creator.caller.pk}] finished successfully.')
            is_w90_parent_folder_valid = True
            self.ctx.workchain_w90_bands = parent_w90_folder.creator.caller.caller
        else:
            self.report('Wannier90BaseWorkChain not provided or failed, will run it again.')
            is_w90_parent_folder_valid = False
            
        ## TODO: what if there w90 provided but ph not?
        
        if is_ph_parent_folder_valid and is_w90_parent_folder_valid:
            compatibility, _ = check_kq_compatibility(self.ctx.kpoints_nscf, self.ctx.qpoints)
            if compatibility:
                self.report('Kpoints of the Wannier90BaseWorkChain is a multiple of the qpoints of the PhBaseWorkChain.')
                is_w90_parent_folder_valid = True
            else:
                self.report('Kpoints of the Wannier90BaseWorkChain is not a multiple of the qpoints of the PhBaseWorkChain.')
                self.report("Rerun Wannier90WorkChain with kpoints based on given PhBaseWorkChain qpoints.")
                is_w90_parent_folder_valid = False

        self.ctx.is_w90_parent_folder_valid = is_w90_parent_folder_valid
        self.ctx.is_ph_parent_folder_valid = is_ph_parent_folder_valid

        self.report(f'is_w90_parent_folder_valid: {self.ctx.is_w90_parent_folder_valid}')
        self.report(f'is_ph_parent_folder_valid: {self.ctx.is_ph_parent_folder_valid}')
        
    def generate_reciprocal_points(self):
        """Generate the qpoints and kpoints meshes for the `ph.x` and `pw.x` calculations."""

        if self.ctx.is_ph_parent_folder_valid:
            if hasattr(self.ctx.workchain_ph.inputs, 'qpoints'):
                qpoints = self.ctx.workchain_ph.inputs.qpoints
            elif hasattr(self.ctx.workchain_ph.inputs, 'qpoints_distance'):
                inputs = {
                    'structure': self.inputs.structure,
                    'distance': self.ctx.workchain_ph.inputs.qpoints_distance,
                    'force_parity': self.ctx.workchain_ph.inputs.get('kpoints_force_parity', orm.Bool(False)),
                    'metadata': {
                        'call_link_label': 're-create_qpoints_from_distance'
                    }
                }
                qpoints = create_kpoints_from_distance(**inputs) 
            else:
                raise ValueError('No qpoints or qpoints_distance provided!')
        else:
            if hasattr(self.inputs, 'qpoints'):
                qpoints = self.inputs.qpoints
            elif hasattr(self.inputs, 'qpoints_distance'):
                inputs = {
                    'structure': self.inputs.structure,
                    'distance': self.inputs.qpoints_distance,
                    'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
                    'metadata': {
                        'call_link_label': 'create_qpoints_from_distance'
                    }
                }
                qpoints = create_kpoints_from_distance(**inputs) 

        if not self.ctx.is_w90_parent_folder_valid:
            
            ## TODO: Check if kpoints is provided in the inputs.
            ## If not, create kpoints from distance.
            ## If yes, use the provided kpoints.
            
            inputs = {
                'structure': self.inputs.structure,
                'distance': self.inputs.kpoints_distance_scf,
                'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_kpoints_scf_from_distance'
                }
            }
            
            kpoints_scf = create_kpoints_from_distance(**inputs)

            qpoints_mesh = qpoints.get_kpoints_mesh()[0]
            kpoints_nscf = orm.KpointsData()
            kpoints_nscf.set_kpoints_mesh([v * self.inputs.kpoints_factor_nscf.value for v in qpoints_mesh])

        else:
            if hasattr(self.ctx.workchain_w90_bands.inputs, 'kpoints_scf'):
                kpoints_scf = self.ctx.workchain_w90_bands.inputs.kpoints_scf
            elif hasattr(self.ctx.workchain_w90_bands.inputs, 'kpoints_distance_scf'):
                inputs = {
                    'structure': self.inputs.structure,
                    'distance': self.ctx.workchain_w90_bands.inputs.kpoints_distance_scf,
                    'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
                    'metadata': {
                    'call_link_label': 're-create_kpoints_scf_from_distance'
                    }
                }
                kpoints_scf = create_kpoints_from_distance(**inputs)
            
            mp_grid = self.ctx.workchain_w90_bands.inputs.parameters['mp_grid']
            kpoints_nscf = orm.KpointsData()
            kpoints_nscf.set_kpoints_mesh(mp_grid)

        self.report(f'kpoints_scf: {kpoints_scf}')
        self.report(f'kpoints_nscf: {kpoints_nscf}')
        
        self.ctx.qpoints = qpoints
        self.ctx.kpoints_scf = kpoints_scf
        self.ctx.kpoints_nscf = kpoints_nscf

    def should_run_wannier90(self):
        """Check if the wannier90 workflow should be run."""
        ## The parent_w90_folder is created by Wannier90Calculation, which is a child process
        ## of Wannier90BaseWorkChain. Wannier90BaseWorkChain is also a child process.
        # If One use 'SCDM' projection, its caller is a Wannier90BandsWorkChain 
        # If One use 'PDWF' projection, its caller is a Wannier90OptimizeWorkChain
        
        return not self.ctx.is_w90_parent_folder_valid

    def run_wannier90(self):
        """Run the wannier90 workflow."""
        
        if 'projwfc' in self.inputs.w90_bands:
            self.report('Running a Wannier90BandsWorkChain.')
            w90_class = Wannier90BandsWorkChain
        else:
            self.report('Running a Wannier90OptimizeWorkChain.')
            w90_class = Wannier90OptimizeWorkChain

        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace='w90_bands')
        )
        inputs.metadata.call_link_label = 'w90_bands'
        inputs.structure = self.inputs.structure
        
        inputs['scf']['kpoints'] = self.ctx.kpoints_scf
        set_kpoints(inputs, self.ctx.kpoints_nscf, w90_class)

        workchain_node = self.submit(w90_class, **inputs)
        self.report(f'launching wannier90 work chain {workchain_node.pk}')

        return ToContext(workchain_w90_bands=workchain_node)

    def inspect_wannier90(self):
        """Verify that the wannier90 workflow finished successfully."""
        workchain = self.ctx.workchain_w90_bands

        if not workchain.is_finished_ok:
            self.report(f'`Wannier90BandsWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90
    
    def should_run_ph(self):
        """Check if the phonon workflow should be run."""
        
        return not self.ctx.is_ph_parent_folder_valid

    def run_ph(self):
        """Run the `PhBaseWorkChain`."""
        inputs = AttributeDict(self.exposed_inputs(PhBaseWorkChain, namespace='ph_base'))

        scf_base_wc = self.ctx.workchain_w90_bands.base.links.get_outgoing(link_label_filter='scf').first().node
        inputs.ph.parent_folder = scf_base_wc.outputs.remote_folder

        inputs.qpoints = self.ctx.qpoints

        inputs.metadata.call_link_label = 'ph_base'
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

        inputs.parent_folder_ph = self.ctx.workchain_ph.outputs.remote_folder

        nscf_base_wc =  self.ctx.workchain_w90_bands.base.links.get_outgoing(link_label_filter='nscf').first().node
        inputs.parent_folder_nscf = nscf_base_wc.outputs.remote_folder

        fine_points = orm.KpointsData()
        fine_points.set_kpoints_mesh([1, 1, 1])

        inputs.kpoints = self.ctx.kpoints_nscf
        inputs.kfpoints = fine_points
        inputs.qpoints = self.ctx.qpoints
        inputs.qfpoints = fine_points      

        parameters = inputs.parameters.get_dict()

        wannier_params = self.ctx.workchain_w90_bands.inputs.wannier90.wannier90.parameters.get_dict()
        exclude_bands = wannier_params.get('exclude_bands') #TODO check this!
        if exclude_bands:
            parameters['INPUTEPW']['bands_skipped'] = f'exclude_bands = {exclude_bands[0]}:{exclude_bands[-1]}'

        parameters['INPUTEPW']['nbndsub'] = wannier_params['num_wann']
        inputs.parameters = orm.Dict(parameters)
        # if 'projwfc' in self.inputs.w90_bands:
        #     w90_remote_data = self.ctx.workchain_w90_bands.outputs.wannier90__remote_folder
        # else:
        #     w90_remote_data = self.ctx.workchain_w90_bands.outputs.wannier90_optimal__remote_folder
        if (
            hasattr(self.ctx.workchain_w90_bands.inputs, 'optimize_disproj') 
            and 
            self.ctx.workchain_w90_bands.inputs.optimize_disproj
            ):
            w90_remote_data = self.ctx.workchain_w90_bands.outputs.wannier90_optimal.remote_folder
        else:
            w90_remote_data = self.ctx.workchain_w90_bands.outputs.wannier90.remote_folder

        wannier_chk_path = Path(w90_remote_data.get_remote_path(), 'aiida.chk')
        nscf_xml_path = Path(self.ctx.workchain_w90_bands.outputs.nscf.remote_folder.get_remote_path(), 'out/aiida.xml')

        prepend_text = inputs.metadata.options.get('prepend_text', '')
        prepend_text += f'\n{self.inputs.w90_chk_to_ukk_script.get_remote_path()} {wannier_chk_path} {nscf_xml_path} aiida.ukk'
        inputs.metadata.options.prepend_text = prepend_text

        inputs.metadata.call_link_label = 'epw'

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

def check_kq_compatibility(
    kpoints: orm.KpointsData,
    qpoints: orm.KpointsData,
) -> (bool, list):
    """
    Check if the kpoints and qpoints are compatible.
    """
    kpoints_mesh = kpoints.get_kpoints_mesh()[0]
    kpoints_shift = kpoints.get_kpoints_mesh()[1]
    qpoints_mesh = qpoints.get_kpoints_mesh()[0]
    qpoints_shift = qpoints.get_kpoints_mesh()[1]
    
    compatibility = None
    multiplicities = []
    remainder = []
    
    for k, q in zip(kpoints_mesh, qpoints_mesh):
        multiplicities.append(k // q)
        remainder.append(k % q)


    if kpoints_shift != [0.0, 0.0, 0.0] or qpoints_shift != [0.0, 0.0, 0.0]:
        compatibility = False
    else:
        if remainder == [0, 0, 0]:
            compatibility = True
        else:
            compatibility = False
    
    return compatibility, multiplicities
