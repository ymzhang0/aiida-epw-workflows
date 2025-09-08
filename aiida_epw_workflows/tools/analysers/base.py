from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict
from abc import ABC, abstractmethod

class BaseWorkChainState(Enum):
    FINISHED_OK = 0
    WAITING = 1
    RUNNING = 2
    EXCEPTED = 3
    KILLED = 4
    UNKNOWN = 999

class BaseWorkChainAnalyser(ABC):
    """
    BaseAnalyser for the WorkChain.
    """
    _all_descendants = OrderedDict([
        ('epw', None),
    ])

    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain
        self.state = BaseWorkChainState.UNKNOWN
        self.descendants = {}


    @staticmethod
    def base_check(
        workchain: orm.WorkChainNode,
        excepted_state: BaseWorkChainState,
        failed_state: BaseWorkChainState,
        namespace: str,
        ) -> tuple[BaseWorkChainState, str]:
        if workchain.process_state == ProcessState.WAITING:
            state = BaseWorkChainState.WAITING
            message = f'is waiting at {namespace}'
        elif workchain.process_state == ProcessState.RUNNING:
            state = BaseWorkChainState.RUNNING
            message = f'is running at {namespace}'
        elif workchain.process_state == ProcessState.EXCEPTED:
            state = excepted_state
            message = f'has excepted at {namespace}'
        elif workchain.process_state == ProcessState.FINISHED:
            if not workchain.is_finished_ok:
                state = failed_state
                message = f'has failed at {namespace}'
            else:
                state = BaseWorkChainState.UNKNOWN
                message = f'{namespace} is finished'
        else:
            state = BaseWorkChainState.UNKNOWN
            message = f'unknown state at {namespace}'

        return state, message
