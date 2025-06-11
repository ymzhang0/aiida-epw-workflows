from .base import EpwBaseWorkChain
from aiida import orm

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.engine import WorkChain, ToContext, if_, while_

class EpwTransportWorkChain(ProtocolMixin, WorkChain):
    """Workchain to calculate transport properties using EPW."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('transport_kpoints', valid_type=orm.KpointsData)
        spec.output('transport_kpoints', valid_type=orm.KpointsData)
        spec.outline(
            cls.run_epw,
            cls.inspect_epw,
            cls.run_transport,
            cls.inspect_transport,
        )
        
    def run_epw(self):
        pass
    
    def inspect_epw(self):
        pass
    
    def run_transport(self):
        pass
    
    def inspect_transport(self):
        pass