from re import S
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict
from ..ph import check_stability_ph_base, check_stability_matdyn_base

class EpwB2WWorkChainState(Enum):
    """
    Analyser for the B2WWorkChain.
    """
    FINISHED_OK = 0
    WAITING = 1
    RUNNING = 2
    EXCEPTED = 3
    KILLED = 4
    W90_INTP_SCF_FAILED = 4004
    W90_INTP_NSCF_FAILED = 4005
    W90_INTP_PW2WAN_FAILED = 4006
    W90_INTP_WANNIER_FAILED = 4007
    W90_INTP_FINISHED_OK = 4008
    PH_BASE_FAILED = 4009
    PH_BASE_S_MATRIX_NOT_POSITIVE_DEFINITE = 4010
    PH_BASE_NODE_FAILURE = 4011
    PH_BASE_UNSTABLE = 4012
    PH_BASE_FINISHED_OK = 4013
    Q2R_BASE_FAILED = 4014
    Q2R_BASE_FINISHED_OK = 4015
    MATDYN_BASE_FAILED = 4016
    MATDYN_BASE_UNSTABLE = 4017
    MATDYN_BASE_FINISHED_OK = 4018
    EPW_BASE_FAILED = 4019
    EPW_BASE_FINISHED_OK = 4020
    W90_INTP_EXCEPTED = 904
    PH_BASE_EXCEPTED = 905
    Q2R_BASE_EXCEPTED = 906
    MATDYN_BASE_EXCEPTED = 907
    EPW_BASE_EXCEPTED = 908
    UNKNOWN = 999

class EpwB2WWorkChainAnalyser:
    """
    Analyser for the B2WWorkChain.
    """
    _all_descendants = OrderedDict([
        ('w90_intp', None),
        ('ph_base',  None),
        ('q2r_base', None),
        ('matdyn_base', None),
        ('epw_base', None),
    ])

    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain
        self.state = EpwB2WWorkChainState.UNKNOWN
        self.descendants = {}
        for link_label, _ in self._all_descendants.items():
            descendants = workchain.base.links.get_outgoing(link_label_filter=link_label).all_nodes()
            if descendants != []:
                self.descendants[link_label] = descendants


    @staticmethod
    def base_check(
        workchain: orm.WorkChainNode,
        excepted_state: EpwB2WWorkChainState,
        failed_state: EpwB2WWorkChainState,
        finished_ok_state: EpwB2WWorkChainState,
        namespace: str,
        ) -> tuple[EpwB2WWorkChainState, str]:
        if workchain.process_state == ProcessState.WAITING:
            state = EpwB2WWorkChainState.WAITING
            message = f'is waiting at {namespace}'
        elif workchain.process_state == ProcessState.RUNNING:
            state = EpwB2WWorkChainState.RUNNING
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
                message = f'{namespace} is finished'
        else:
            state = EpwB2WWorkChainState.UNKNOWN
            message = f'unknown state at {namespace}'

        return state, message

    @property
    def w90_intp(self):
        if self.descendants['w90_intp'] == []:
            raise ValueError('w90_intp is not found')
        else:
            return self.descendants['w90_intp']

    @property
    def ph_base(self):
        if self.descendants['ph_base'] == []:
            raise ValueError('ph_base is not found')
        else:
            return self.descendants['ph_base']

    @property
    def q2r_base(self):
        if self.descendants['q2r_base'] == []:
            raise ValueError('q2r_base is not found')
        else:
            return self.descendants['q2r_base']

    @property
    def matdyn_base(self):
        if self.descendants['matdyn_base'] == []:
            raise ValueError('matdyn_base is not found')
        else:
            return self.descendants['matdyn_base']

    @property
    def epw_base(self):
        if self.descendants['epw_base'] == []:
            raise ValueError('epw_base is not found')
        else:
            return self.descendants['epw_base']

    def check_w90_intp(self):
        """Check the state of the w90_intp workchain."""
        w90_intp = self.w90_intp[0]
        w90_intp_descendants = self.get_descendants_by_link_type(w90_intp, LinkType.CALL_WORK)
        if w90_intp.is_excepted:
            state = EpwB2WWorkChainState.W90_INTP_EXCEPTED
            message = 'has excepted at w90_intp'
        if w90_intp.exit_status == 430:
            nscf = w90_intp_descendants['nscf'][0]
            nscf_descendants = self.get_descendants_by_link_type(nscf, LinkType.CALL_CALC)
            max_iter = [int(key.split('_')[-1]) for key in nscf_descendants.keys() if 'iteration' in key]
            remote_folder = self.get_descendants_by_link_type(
                nscf_descendants[f'iteration_{max_iter[0]:02d}'][-1],
                LinkType.CREATE
            )['remote_folder'][0]
            state = EpwB2WWorkChainState.W90_INTP_NSCF_FAILED
            message = f'has failed at nscf, go check the remote folder {remote_folder.get_remote_path()}'
        elif w90_intp.exit_status == 0:
            state = EpwB2WWorkChainState.W90_INTP_FINISHED_OK
            message = 'has finished successfully at w90_intp'
        else:
            state = EpwB2WWorkChainState.UNKNOWN
            message = 'has unknown state at w90_intp'

        return state, message

    def get_iterations(self, link_label: str):
        """Get the iterations of the workchain."""

        iterations = []
        for (node, link_type, link_label) in self.descendants[link_label][-1].base.links.get_outgoing().all():
            if link_label.startswith('iteration'):
                iterations.append(node)
        return iterations

    def check_ph_base(self):
        """Check the state of the ph_base workchain."""

        state, message = self.base_check(
            self.ph_base[0],
            EpwB2WWorkChainState.PH_BASE_EXCEPTED,
            EpwB2WWorkChainState.PH_BASE_FAILED,
            EpwB2WWorkChainState.PH_BASE_FINISHED_OK,
            'ph_base'
        )

        from ..workchains import find_iterations

        iterations = find_iterations(self.ph_base[0])

        max_iteration = max(iterations, key=lambda x: int(x.split('_')[1]))

        last_iteration = self.ph_base[0].base.links.get_outgoing(link_label_filter=max_iteration).first().node

        stderr = last_iteration.get_scheduler_stderr()
        aiida_out = last_iteration.outputs.retrieved.get_object_content('aiida.out')

        if 'S matrix not positive definite' in aiida_out:
            state = EpwB2WWorkChainState.PH_BASE_S_MATRIX_NOT_POSITIVE_DEFINITE
            message = 'has failed at ph_base due to S matrix not positive definite'
        elif 'NODE FAILURE,' in stderr:
            state = EpwB2WWorkChainState.PH_BASE_NODE_FAILURE
            message = 'has failed at ph_base due to node failure'

        return state, message

    def check_q2r_base(self):
        """Check the state of the q2r_base workchain."""
        return self.base_check(
            self.q2r_base[0],
            EpwB2WWorkChainState.Q2R_BASE_EXCEPTED,
            EpwB2WWorkChainState.Q2R_BASE_FAILED,
            EpwB2WWorkChainState.Q2R_BASE_FINISHED_OK,
            'q2r_base'
        )

    def check_matdyn_base(self):
        """Check the state of the matdyn_base workchain."""
        return self.base_check(
            self.matdyn_base[0],
            EpwB2WWorkChainState.MATDYN_BASE_EXCEPTED,
            EpwB2WWorkChainState.MATDYN_BASE_FAILED,
            EpwB2WWorkChainState.MATDYN_BASE_FINISHED_OK,
            'matdyn_base'
        )

    def check_epw_base(self):
        """Check the state of the epw_base workchain."""
        epw_base = self.epw_base[0]
        if epw_base.is_excepted:
            state = EpwB2WWorkChainState.EPW_BASE_EXCEPTED
            message = 'has excepted at epw_base'
        elif epw_base.process_state == ProcessState.FINISHED:
            if not epw_base.is_finished_ok:
                state = EpwB2WWorkChainState.EPW_BASE_FAILED
                message = 'has failed at epw_base'
            else:
                state = EpwB2WWorkChainState.EPW_BASE_FINISHED_OK
                message = 'has finished successfully at epw_base'

        epw_descendants = self.get_descendants_by_link_type(epw_base, LinkType.CALL_CALC)
        max_iter = [int(key.split('_')[-1]) for key in epw_descendants.keys() if 'iteration' in key]
        remote_folder = self.get_descendants_by_link_type(
            epw_descendants[f'iteration_{max_iter[0]:02d}'][-1],
            LinkType.CREATE
        )['remote_folder'][0]
        state = EpwB2WWorkChainState.EPW_BASE_FAILED
        message = f'has failed at epw_base, go check the remote folder {remote_folder.get_remote_path()}'

        return state, message

    def check_process_state(self):
        """Check the state of the b2w workchain."""
        if self.node.is_excepted:
            state = EpwB2WWorkChainState.EXCEPTED
            message = 'has excepted at b2w'
        elif self.node.process_state == ProcessState.WAITING:
            state = EpwB2WWorkChainState.WAITING
            message = 'is waiting at b2w'
        elif self.node.process_state == ProcessState.RUNNING:
            state = EpwB2WWorkChainState.RUNNING
            message = 'is running at b2w'
        elif self.node.exit_status == 0:
            state = EpwB2WWorkChainState.FINISHED_OK
            message = 'has finished successfully at b2w'
        elif self.node.exit_status == 402:
            state, message = self.check_w90_intp()
        elif self.node.exit_status == 403:
            state, message = self.check_ph_base()
        elif self.node.exit_status == 404:
            state, message = self.check_q2r_base()
        elif self.node.exit_status == 405:
            state, message = self.check_matdyn_base()
        elif self.node.exit_status == 406:
            state, message = self.check_epw_base()
        elif self.node.exit_status == 407:
            state = EpwB2WWorkChainState.PH_BASE_UNSTABLE
            _, message = self.check_stability_ph_base()
        elif self.node.exit_status == 408:
            state = EpwB2WWorkChainState.MATDYN_BASE_UNSTABLE
            _, message = self.check_stability_matdyn_base()

        return state, message

    @staticmethod
    def get_descendants_by_link_type(
        node: orm.WorkChainNode,
        link_type: LinkType = LinkType.CALL_WORK
        ) -> dict:
        """Get the descendant nodes of the parent workchain."""

        descendants = {}
        try:
            for node, link_type, link_label in node.base.links.get_outgoing(link_type=link_type).all():
                if link_label not in descendants:
                    descendants[link_label] = []
                descendants[link_label].append(node)
        except AttributeError:
            pass

        return descendants

    def check_stability_ph_base(self):
        """Get the qpoints and frequencies of the ph_base workchain."""
        state, _ = self.check_ph_base()
        if state == EpwB2WWorkChainState.PH_BASE_FINISHED_OK:
            return check_stability_ph_base(self.ph_base[0])
        else:
            raise ValueError('ph_base is not finished')

    def check_stability_matdyn_base(self):
        """Get the qpoints and frequencies of the matdyn_base workchain."""
        state, _ = self.check_matdyn_base()
        if state == EpwB2WWorkChainState.MATDYN_BASE_FINISHED_OK:
            return check_stability_matdyn_base(self.matdyn_base[0])
        else:
            raise ValueError('matdyn_base is not finished')