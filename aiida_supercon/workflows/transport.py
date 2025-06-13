from .base import EpwBaseWorkChain
from aiida import orm
from aiida.engine import WorkChain, ToContext, if_
from .intp import EpwBaseIntpWorkChain

class EpwTransportWorkChain(EpwBaseIntpWorkChain):
    """Workchain to calculate transport properties using EPW."""
    _B2W_NAMESPACE = 'b2w'
    _INTP_NAMESPACE = 'transport'
    
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.outline(
            cls.setup,
            cls.validate_parent_folders,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            cls.run_transport,
            cls.inspect_transport,
            cls.results
        )
        
    def run_transport(self):
        pass
    
    def inspect_transport(self):
        pass
    
    def run_transport(self):
        pass
    
    def inspect_transport(self):
        pass

    def results(self):
        pass
    
    def on_terminated(self):
        pass