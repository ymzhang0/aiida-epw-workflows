from email import message
from re import S
from socket import NI_NOFQDN
from pathlib import Path

from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
import numpy
from ..workchains import clean_workdir
from enum import Enum
from aiida.tools import delete_nodes
from .b2w import EpwB2WWorkChainAnalyser
from ..plot import (
    plot_epw_interpolated_bands,
    plot_a2f,
    plot_eldos,
    plot_gap_function,
    plot_bands_comparison,
    plot_bands,
    plot_aniso,
    plot_phdos
)


class EpwSuperConWorkChainState(Enum):
    FINISHED_OK = 0
    WAITING = 1
    RUNNING = 2
    EXCEPTED = 3
    KILLED = 4
    PW_RELAX_FAILED = 4001
    PW_BANDS_FAILED = 4002
    B2W_W90_INTP_SCF_FAILED = 4004
    B2W_W90_INTP_NSCF_FAILED = 4005
    B2W_W90_INTP_PW2WAN_FAILED = 4006
    B2W_W90_INTP_WANNIER_FAILED = 4007
    B2W_PH_BASE_S_MATRIX_NOT_POSITIVE_DEFINITE = 4008
    B2W_PH_BASE_NODE_FAILURE = 4009
    B2W_PH_BASE_FAILED = 4008
    B2W_PH_BASE_UNSTABLE = 4009
    B2W_MATDYN_BASE_UNSTABLE = 4010
    B2W_EPW_BASE_FAILED = 4011
    BANDS_FAILED = 4012
    BANDS_UNSTABLE = 4013
    CONVERGENCE_NOT_REACHED = 4014
    A2F_FAILED = 4015
    A2F_CONV_FAILED = 4016
    ALLEN_DYNES_TC_TOO_LOW = 4017
    ISO_FAILED = 4018
    ISO_TC_TOO_LOW = 4019
    ANISO_FAILED = 4020
    PW_RELAX_EXCEPTED = 901
    PW_BANDS_EXCEPTED = 902
    B2W_EXCEPTED = 903
    B2W_W90_INTP_EXCEPTED = 904
    B2W_PH_BASE_EXCEPTED = 905
    B2W_EPW_BASE_EXCEPTED = 906
    BANDS_EXCEPTED = 907
    A2F_EXCEPTED = 908
    ISO_EXCEPTED = 909
    ANISO_EXCEPTED = 910
    PW_RELAX_FINISHED_OK = 1001
    PW_BANDS_FINISHED_OK = 1002
    B2W_W90_INTP_SCF_FINISHED_OK = 1003
    B2W_W90_INTP_NSCF_FINISHED_OK = 1004
    B2W_W90_INTP_PW2WAN_FINISHED_OK = 1005
    B2W_W90_INTP_WANNIER_FINISHED_OK = 1006
    B2W_PH_BASE_FINISHED_OK = 1007
    B2W_EPW_BASE_FINISHED_OK = 1008
    B2W_FINISHED_OK = 1009
    BANDS_FINISHED_OK = 1010
    A2F_FINISHED_OK = 1011
    ISO_FINISHED_OK = 1012
    ANISO_FINISHED_OK = 1013
    UNKNOWN = 999

from collections import OrderedDict

class EpwSuperConWorkChainAnalyser:
    """
    Analyser for the EpwSuperConWorkChain.
    """
    _all_descendants = OrderedDict([
        ('pw_relax', None),
        ('pw_bands', None),
        ('b2w',      None),
        ('bands',    None),
        ('a2f_conv', None),
        ('a2f',      None),
        ('iso',      None),
        ('aniso',    None),
    ])

    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain
        self.state = EpwSuperConWorkChainState.UNKNOWN
        self.descendants = {}
        for link_label, _ in self._all_descendants.items():
            descendants = workchain.base.links.get_outgoing(link_label_filter=link_label).all_nodes()
            if descendants != []:
                self.descendants[link_label] = descendants

    @staticmethod
    def base_check(
        workchain: orm.WorkChainNode,
        excepted_state: EpwSuperConWorkChainState,
        failed_state: EpwSuperConWorkChainState,
        finished_ok_state: EpwSuperConWorkChainState,
        namespace: str,
        ) -> tuple[EpwSuperConWorkChainState, str]:
        if not workchain:
            return (
                finished_ok_state,
                '{namespace} is not found, should be skipped'
            )
        if workchain.process_state == ProcessState.WAITING:
            state = EpwSuperConWorkChainState.WAITING
            message = f'is waiting at {namespace}'
        elif workchain.process_state == ProcessState.RUNNING:
            state = EpwSuperConWorkChainState.RUNNING
            message = f'is running at {namespace}'
        elif workchain.process_state == ProcessState.EXCEPTED:
            state = excepted_state
            message = f'has excepted at {namespace}'
        elif workchain.process_state == ProcessState.FINISHED:
            if not workchain.is_finished_ok:
                state = failed_state
                message = f'has failed at {namespace}'
            else:
                state = finished_ok_state
                message = f'{namespace} is finished successfully'
        else:
            state = EpwSuperConWorkChainState.UNKNOWN
            message = f'unknown state at {namespace}'

        return state, message

    @property
    def structure(self):
        if self.node.inputs.structure is None:
            raise ValueError('structure is not found')
        else:
            return self.node.inputs.structure

    @property
    def pw_relax(self):
        if self.descendants['pw_relax'] == []:
            raise ValueError('pw_relax is not found')
        else:
            return self.descendants['pw_relax']

    @property
    def pw_bands(self):
        if self.descendants['pw_bands'] == []:
            raise ValueError('pw_bands is not found')
        else:
            return self.descendants['pw_bands']

    @property
    def b2w(self):
        if self.descendants['b2w'] == []:
            raise ValueError('b2w is not found')
        else:
            return self.descendants['b2w']

    @property
    def b2w_analyser(self):
        if self.descendants['b2w'] == []:
            raise ValueError('`b2w` is not found')
        else:
            return EpwB2WWorkChainAnalyser(self.descendants['b2w'][-1])

    @property
    def b2w_w90_intp(self):
        if self.descendants['b2w'] == []:
            raise ValueError('`b2w` is not found')
        else:
            return self.b2w_analyser.w90_intp

    @property
    def b2w_ph_base(self):
        if self.descendants['b2w'] == []:
            raise ValueError('`b2w` is not found')
        else:
            return self.b2w_analyser.ph_base

    @property
    def b2w_q2r_base(self):
        if self.descendants['b2w'] == []:
            raise ValueError('`b2w` is not found')
        else:
            return self.b2w_analyser.q2r_base

    @property
    def b2w_matdyn_base(self):
        if self.descendants['b2w'] == []:
            raise ValueError('`b2w` is not found')
        else:
            return self.b2w_analyser.matdyn_base

    @property
    def bands(self):
        if self.descendants['bands'] == []:
            raise ValueError('bands is not found')
        else:
            return self.descendants['bands']

    @property
    def a2f_conv(self):
        if self.descendants['a2f_conv'] == []:
            raise ValueError('a2f_conv is not found')
        else:
            return self.descendants['a2f_conv']

    @property
    def a2f(self):
        if self.descendants['a2f'] == []:
            raise ValueError('a2f is not found')
        else:
            return self.descendants['a2f']

    @property
    def iso(self):
        if self.descendants['iso'] == []:
            raise ValueError('iso is not found')
        else:
            return self.descendants['iso']

    @property
    def aniso(self):
        if self.descendants['aniso'] == []:
            raise ValueError('aniso is not found')
        else:
            return self.descendants['aniso']

    def check_pw_relax(self):
        """Check the state of the pw_relax workchain."""


        return self.base_check(
            self.pw_relax[-1],
            EpwSuperConWorkChainState.PW_RELAX_EXCEPTED,
            EpwSuperConWorkChainState.PW_RELAX_FAILED,
            EpwSuperConWorkChainState.PW_RELAX_FINISHED_OK,
            'pw_relax'
        )

    def check_pw_bands(self):
        """Check the state of the pw_bands workchain."""
        return self.base_check(
            self.pw_bands[0],
            EpwSuperConWorkChainState.PW_BANDS_EXCEPTED,
            EpwSuperConWorkChainState.PW_BANDS_FAILED,
            EpwSuperConWorkChainState.PW_BANDS_FINISHED_OK,
            'pw_bands'
        )

    def check_b2w(self):
        """Check the state of the b2w workchain."""
        b2w_analyser = EpwB2WWorkChainAnalyser(self.descendants['b2w'][-1])
        return b2w_analyser.check_process_state()

    def check_bands(self):
        """Check the state of the bands workchain."""
        return self.base_check(
            self.bands[0],
            EpwSuperConWorkChainState.BANDS_EXCEPTED,
            EpwSuperConWorkChainState.BANDS_FAILED,
            EpwSuperConWorkChainState.BANDS_FINISHED_OK,
            'bands'
        )

    def check_a2f(self):
        return self.base_check(
            self.a2f[0],
            EpwSuperConWorkChainState.A2F_EXCEPTED,
            EpwSuperConWorkChainState.A2F_FAILED,
            EpwSuperConWorkChainState.A2F_FINISHED_OK,
            'a2f'
        )

    def check_a2f_conv(self):
        for a2f_conv_workchain in self.a2f_conv:
            if a2f_conv_workchain.is_excepted:
                state = EpwSuperConWorkChainState.A2F_CONV_EXCEPTED
                message += f'has excepted at a2f_conv<{a2f_conv_workchain.pk}>'
            else:
                state = EpwSuperConWorkChainState.A2F_CONV_FAILED
                message += f'has failed at a2f_conv<{a2f_conv_workchain.pk}>'

        return state, message

    def check_iso(self):
        """Check the state of the iso workchain."""
        return self.base_check(
            self.iso[0],
            EpwSuperConWorkChainState.ISO_EXCEPTED,
            EpwSuperConWorkChainState.ISO_FAILED,
            EpwSuperConWorkChainState.ISO_FINISHED_OK,
            'iso'
        )

    def check_aniso(self):
        return self.base_check(
            self.aniso[0],
            EpwSuperConWorkChainState.ANISO_EXCEPTED,
            EpwSuperConWorkChainState.ANISO_FAILED,
            EpwSuperConWorkChainState.ANISO_FINISHED_OK,
            'aniso'
        )

    def check_process_state(self):
        """Check the state of the workchain."""
        state = EpwSuperConWorkChainState.UNKNOWN
        message = 'status is not known'

        state, message = self.check_pw_relax()

        if not state == EpwSuperConWorkChainState.PW_RELAX_FINISHED_OK:
            return state, message

        state, message = self.check_pw_bands()

        if not state == EpwSuperConWorkChainState.PW_BANDS_FINISHED_OK:
            return state, message

        state, message = self.check_b2w()

        if not state == EpwSuperConWorkChainState.B2W_FINISHED_OK:
            return state, message

        state, message = self.check_bands()

        if not state == EpwSuperConWorkChainState.BANDS_FINISHED_OK:
            return state, message

        state, message = self.check_a2f()

        if not state == EpwSuperConWorkChainState.A2F_FINISHED_OK:
            return state, message

        state, message = self.check_a2f_conv()

        if not state == EpwSuperConWorkChainState.A2F_CONV_FINISHED_OK:
            return state, message

        state, message = self.check_iso()

        if not state == EpwSuperConWorkChainState.ISO_FINISHED_OK:
            return state, message

        state, message = self.check_aniso()

        if not state == EpwSuperConWorkChainState.ANISO_FINISHED_OK:
            return state, message

        state = EpwSuperConWorkChainState.FINISHED_OK
        message = 'has finished successfully'

        return state, message


    def get_state(self):
        pk = self.node.pk
        formula = self.node.inputs.structure.get_formula()
        source_db, source_id = self.node.inputs.structure.base.extras.get_many(('source_db', 'source_id'))
        state, message = self.check_process_state()
        material_info = f'{source_db}-{source_id}<{formula}>'
        message = f'[{pk}]: {material_info:30s} {message}'

        return state, message

    def get_source(self):
        """Get the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            return (self.node.base.extras.get('source_db'), self.node.base.extras.get('source_id'))
        elif all(key in self.node.inputs.structure.base.extras for key in ['source_db', 'source_id']):
            return (self.node.inputs.structure.base.extras.get('source_db'), self.node.inputs.structure.base.extras.get('source_id'))
        else:
            raise ValueError('Source is not set')

    def get_pw_relax_remote_path(self):
        """Get the remote directory of the pw_relax workchain."""
        return self.pw_relax[0].outputs.remote_folder.get_remote_path()

    def get_pw_bands_scf_remote_path(self):
        """Get the remote directory of the pw_bands workchain."""
        scf_workchain = self.pw_bands[0].base.links.get_outgoing(link_label_filter='scf').first().node
        return scf_workchain.outputs.remote_folder.get_remote_path()

    def get_pw_bands_nscf_remote_path(self):
        """Get the remote directory of the pw_bands workchain."""
        nscf_workchain = self.pw_bands[0].base.links.get_outgoing(link_label_filter='bands').first().node
        return nscf_workchain.outputs.remote_folder.get_remote_path()

    def get_b2w_w90_intp_scf_remote_path(self):
        """Get the remote directory of the b2w workchain."""
        return self.b2w[0].outputs.w90_intp.scf.remote_folder.get_remote_path()

    def get_b2w_w90_intp_nscf_remote_path(self):
        """Get the remote directory of the b2w workchain."""
        return self.b2w[0].outputs.w90_intp.nscf.remote_folder.get_remote_path()

    def get_b2w_ph_base_remote_path(self):
        """Get the remote directory of the b2w workchain."""
        return self.b2w[0].outputs.ph_base.remote_folder.get_remote_path()

    def get_b2w_epw_base_remote_path(self):
        """Get the remote directory of the b2w workchain."""
        return self.b2w[0].outputs.epw_base.remote_folder.get_remote_path()

    def get_bands_remote_path(self):
        """Get the remote directory of the bands workchain."""
        return self.bands[0].outputs.remote_folder.get_remote_path()

    def get_a2f_conv_remote_path(self):
        """Get the remote directory of the a2f_conv workchain."""
        return self.a2f_conv[0].outputs.remote_folder.get_remote_path()

    def get_a2f_remote_path(self):
        """Get the remote directory of the a2f workchain."""
        return self.a2f[0].outputs.remote_folder.get_remote_path()

    def get_iso_remote_path(self):
        """Get the remote directory of the iso workchain."""
        return self.iso[0].outputs.remote_folder.get_remote_path()

    def get_aniso_remote_path(self):
        """Get the remote directory of the aniso workchain."""
        return self.aniso[0].outputs.remote_folder.get_remote_path()

    @property
    def source(self):
        """Get the source of the workchain."""
        try:
            source_db, source_id = self.get_source()
            return f'{source_db}-{source_id}'
        except (ValueError, KeyError):
            return None

    def set_source(self):
        """Set the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            raise Warning('Source is already set')
        else:
            source_db, source_id = self.get_source()
            self.node.base.extras.set_many({
                'source_db': source_db,
                'source_id': source_id
            })

    def get_descendants_by_label(
        self,
        link_label_filter: str
        ) -> orm.WorkChainNode:
        """Get the descendant workchains of the parent workchain by the link label."""
        try:
            return self.node.base.links.get_outgoing(
                link_label_filter=link_label_filter
                ).all()
        except AttributeError:
            return None

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        state, _ = self.check_process_state()
        message = ''
        if state in [
                EpwSuperConWorkChainState.FINISHED_OK,
                EpwSuperConWorkChainState.WAITING,
                EpwSuperConWorkChainState.RUNNING,
                EpwSuperConWorkChainState.B2W_PH_BASE_UNSTABLE,
                EpwSuperConWorkChainState.B2W_MATDYN_BASE_UNSTABLE,
                EpwSuperConWorkChainState.BANDS_UNSTABLE,
                EpwSuperConWorkChainState.CONVERGENCE_NOT_REACHED,
                EpwSuperConWorkChainState.ALLEN_DYNES_TC_TOO_LOW,
                EpwSuperConWorkChainState.ISO_TC_TOO_LOW,
                ]:
                message += 'Please check if you really want to clean this workchain.'
                return message
        
        cleaned_calcs = clean_workdir(self.node, dry_run=dry_run)
        message += f'Cleaned the workchain {self.node.pk}:\n'
        message += '  ' + ' '.join(map(str, cleaned_calcs)) + '\n'
        message += f'Deleted the workchain {self.node.pk}:\n'
        deleted_nodes, _ = delete_nodes([self.node.pk], dry_run=dry_run)
        message += '  ' + ' '.join(map(str, deleted_nodes))

        return message

    def check_convergence_allen_dynes_tc(
        self,
        convergence_threshold: float
        ) -> tuple[bool, str]:
        """Check if the convergence is reached."""

        a2f_conv_workchains = self.get_descendants_by_label('a2f_conv')

        try:
            prev_allen_dynes = a2f_conv_workchains[-2].outputs.output_parameters['Allen_Dynes_Tc']
            new_allen_dynes = a2f_conv_workchains[-1].outputs.output_parameters['Allen_Dynes_Tc']
            is_converged = (
                abs(prev_allen_dynes - new_allen_dynes) / new_allen_dynes
                < convergence_threshold
            )
            return (
                is_converged,
                f'Checking convergence: old {prev_allen_dynes}; new {new_allen_dynes} -> Converged = {is_converged}')
        except (AttributeError, IndexError, KeyError):
            return (False, 'Not enough data to check convergence.')


    def check_stability_epw_bands(
        self,
        min_freq: float # meV ~ 8.1 cm-1
        ) -> tuple[bool, str]:
        """Check if the epw.x interpolated phonon band structure is stable."""

        epw_bands = self.get_descendants_by_label('epw_bands')
        ph_bands = epw_bands.outputs.bands.ph_band_structure.get_bands()
        min_freq = numpy.min(ph_bands)
        max_freq = numpy.max(ph_bands)

        if min_freq < min_freq:
            return (False, max_freq)
        else:
            return (True, max_freq)

    def dump_inputs(self, destpath: Path):
        """Dump the inputs of the workchain."""

        if self.source is not None:
            destpath = destpath / self.source

        print('Writing pw_relax files to ', destpath / 'pw_relax')
        for node, _, link_label1 in self.pw_relax[0].base.links.get_outgoing(link_type=LinkType.CALL_WORK).all():
            pw_calculations = node.base.links.get_outgoing(link_type=LinkType.CALL_CALC).all()
            dirpath = destpath / f'pw_relax_{link_label1}'
            dirpath.mkdir(parents=True, exist_ok=True)

            for pw_calculation, _, link_label2 in pw_calculations:
                if not link_label2.startswith('iteration_'):
                    continue
                with open(dirpath / f'{link_label2}.in', 'w') as f:
                    f.write(pw_calculation.base.repository.get_object_content('aiida.in'))
                with open(dirpath / f'{link_label2}.out', 'w') as f:
                    f.write(pw_calculation.outputs.retrieved.get_object_content('aiida.out'))

        w90_intp_workchains = self.b2w[0].base.links.get_outgoing(link_label_filter='w90_intp').all()
        scf_workchain = w90_intp_workchains[-1].node.base.links.get_outgoing(link_label_filter='scf').all()[-1]
        for node, _, link_label in scf_workchain.node.base.links.get_outgoing(link_type=LinkType.CALL_CALC).all():
            if not link_label.startswith('iteration_'):
                continue
            dirpath = destpath / 'w90_intp_scf'
            dirpath.mkdir(parents=True, exist_ok=True)
            with open(dirpath / f'{link_label}.in', 'w') as f:
                f.write(node.base.repository.get_object_content('aiida.in'))
            with open(dirpath / f'{link_label}.out', 'w') as f:
                f.write(node.outputs.retrieved.get_object_content('aiida.out'))

        nscf_workchain = w90_intp_workchains[-1].node.base.links.get_outgoing(link_label_filter='nscf').all()[-1]
        for node, _, link_label in nscf_workchain.node.base.links.get_outgoing(link_type=LinkType.CALL_CALC).all():
            if not link_label.startswith('iteration_'):
                continue
            dirpath = destpath / 'w90_intp_nscf'
            dirpath.mkdir(parents=True, exist_ok=True)
            with open(dirpath / f'{link_label}.in', 'w') as f:
                f.write(node.base.repository.get_object_content('aiida.in'))
            with open(dirpath / f'{link_label}.out', 'w') as f:
                f.write(node.outputs.retrieved.get_object_content('aiida.out'))

        ph_workchain = self.b2w[0].base.links.get_outgoing(link_label_filter='ph_base').all()[-1]
        for node, _, link_label in ph_workchain.node.base.links.get_outgoing(link_type=LinkType.CALL_CALC).all():
            if not link_label.startswith('iteration_'):
                continue
            dirpath = destpath / 'ph_base'
            dirpath.mkdir(parents=True, exist_ok=True)
            with open(dirpath / f'{link_label}.in', 'w') as f:
                f.write(node.base.repository.get_object_content('aiida.in'))
            with open(dirpath / f'{link_label}.out', 'w') as f:
                f.write(node.outputs.retrieved.get_object_content('aiida.out'))

            for file in node.outputs.retrieved.list_object_names('DYN_MAT'):
                dirpath = destpath / 'ph_base' / 'DYN_MAT'
                dirpath.mkdir(parents=True, exist_ok=True)
                with open(dirpath / file, 'w') as f:
                    f.write(node.outputs.retrieved.get_object_content(f"DYN_MAT/{file}"))

        q2r_workchain = self.b2w[0].base.links.get_outgoing(link_label_filter='q2r_base').all()[-1]
        for node, _, link_label in q2r_workchain.node.base.links.get_outgoing(link_type=LinkType.CALL_CALC).all():
            if not link_label.startswith('iteration_'):
                continue
            dirpath = destpath / 'q2r_base'
            dirpath.mkdir(parents=True, exist_ok=True)
            with open(dirpath / f'{link_label}.in', 'w') as f:
                f.write(node.base.repository.get_object_content('aiida.in'))
            with open(dirpath / f'{link_label}.out', 'w') as f:
                f.write(node.outputs.retrieved.get_object_content('aiida.out'))
            with open(dirpath / f'{link_label}.fc', 'w') as f:
                f.write(node.outputs.retrieved.get_object_content('real_space_force_constants.dat'))

        matdyn_workchain = self.b2w[0].base.links.get_outgoing(link_label_filter='matdyn_base').all()[-1]
        for node, _, link_label in matdyn_workchain.node.base.links.get_outgoing(link_type=LinkType.CALL_CALC).all():
            if not link_label.startswith('iteration_'):
                continue
            dirpath = destpath / 'matdyn_base'
            dirpath.mkdir(parents=True, exist_ok=True)
            with open(dirpath / f'{link_label}.in', 'w') as f:
                f.write(node.base.repository.get_object_content('aiida.in'))
            with open(dirpath / f'{link_label}.out', 'w') as f:
                f.write(node.outputs.retrieved.get_object_content('aiida.out'))
            with open(dirpath / f'{link_label}.modes', 'w') as f:
                f.write(node.outputs.retrieved.get_object_content('phonon_displacements.dat'))
            with open(dirpath / f'{link_label}.freq', 'w') as f:
                f.write(node.outputs.retrieved.get_object_content('phonon_displacements.dat'))

    def show_bands(self):
        """Show the bands."""
        bands = self.bands[0].outputs.bands.band_structure.get_bands()
        print(bands)


    def show_eldos(
        self,
        axis = None,
        **kwargs,
        ):

        plot_eldos(
            a2f_workchain = self.a2f[0],
            axis = axis,
            **kwargs,
        )

    def show_phdos(
        self,
        axis = None,
        **kwargs,
        ):

        plot_phdos(
            phdos = self.a2f[0].outputs.a2f.phdos,
            axis = axis,
            **kwargs,
        )


    def show_a2f(self, axis=None, **kwargs):

        plot_a2f(
            a2f_workchain = self.a2f[0],
            axis = axis,
            **kwargs,
        )

    def show_epw_interpolated_bands(self, axes=None, **kwargs):

        plot_epw_interpolated_bands(
            epw_workchain = self.bands[0],
            axes = axes,
            **kwargs,
        )

    def show_gap_function(self, axis=None, **kwargs):

        plot_gap_function(
            aniso_workchain = self.aniso[0],
            axis = axis,
            **kwargs,
        )

    def plot_all(self):
        from matplotlib import gridspec
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 12))
        gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[0, 1])
        ax5 = fig.add_subplot(gs[2, 0])

        plot_epw_interpolated_bands(
            epw_workchain = self.bands[0],
            axes=numpy.array([ax1, ax2]),
        )

        plot_a2f(
            a2f_workchain = self.a2f_conv[-1],
            axis = ax3,
            show_data = True,
            )

        plot_eldos(
            a2f_workchain = self.a2f_conv[-1],
            axis = ax4,
            )

