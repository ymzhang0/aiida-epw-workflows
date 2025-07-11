
from aiida import orm
from aiida.engine import WorkChain, ToContext, if_
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from .b2w import EpwB2WWorkChain
from .bands import EpwBandsWorkChain

from .bte import EpwBteWorkChain
from .a2f import EpwA2fWorkChain

class EpwTransportWorkChain(ProtocolMixin, WorkChain):
    """Workchain to calculate transport properties using EPW."""

    _NAMESPACE = 'transport'
    _B2W_NAMESPACE = EpwB2WWorkChain._NAMESPACE
    _BANDS_NAMESPACE = EpwBandsWorkChain._INTP_NAMESPACE
    _A2F_NAMESPACE = EpwA2fWorkChain._INTP_NAMESPACE
    _BTE_NAMESPACE = EpwBteWorkChain._INTP_NAMESPACE


    @classmethod
    def define(cls, spec):

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))

        spec.expose_inputs(
            EpwB2WWorkChain,
            namespace=cls._B2W_NAMESPACE,
            exclude=(
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwB2WWorkChain`.'
            }
        )

        spec.expose_inputs(
            EpwBandsWorkChain,
            namespace=cls._BANDS_NAMESPACE,
            exclude=(
                'clean_workdir',
                'structure',
                f'{cls._BANDS_NAMESPACE}.parent_folder_nscf',
                f'{cls._BANDS_NAMESPACE}.parent_folder_chk',
                f'{cls._BANDS_NAMESPACE}.parent_folder_ph',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwBandsWorkChain`.'
            }
        )
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            if_(cls.should_run_bands)(
                cls.run_bands,
                cls.inspect_bands,
            ),
            if_(cls.should_run_a2f)(
                cls.run_a2f,
                cls.inspect_a2f,
            ),
            if_(cls.should_run_bte)(
                cls.run_bte,
                cls.inspect_bte,
            ),
            cls.results
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_B2W',
            message='The `b2w` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_BANDS',
            message='The `bands` sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_A2F',
            message='The `a2f` sub process failed')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_BTE',
            message='The `bte` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._NAMESPACE}.yaml'

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` for default protocol.
        :return: The overrides.
        """
        from importlib_resources import files
        import yaml

        from . import protocols

        path = files(protocols) / f"{cls._NAMESPACE}.yaml"
        with path.open() as file:
            return yaml.safe_load(file)

    @classmethod
    def get_builder_from_protocol(
            cls,
            codes,
            structure,
            protocol='moderate',
            overrides=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        args = (codes, structure, protocol)

        builder = cls.get_builder()

        b2w_builder = EpwB2WWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get(cls._B2W_NAMESPACE, None),
            **kwargs
        )

        builder[cls._B2W_NAMESPACE]._data = b2w_builder._data

        for (epw_namespace, epw_workchain_class) in (
            (cls._BANDS_NAMESPACE, EpwBandsWorkChain),
            (cls._A2F_NAMESPACE, EpwA2FWorkChain),
            (cls._BTE_NAMESPACE, EpwBteWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder_from_protocol(
                *args,
                overrides=inputs.get(epw_namespace, None),
                **kwargs
            )

            epw_builder.pop(epw_workchain_class._B2W_NAMESPACE)

            builder[epw_namespace]._data = epw_builder._data


        builder.structure = structure
        # builder.interpolation_distances = orm.List(inputs.get('interpolation_distances', None))
        # builder.convergence_threshold = orm.Float(inputs['convergence_threshold'])
        # builder.always_run_final = orm.Bool(inputs.get('always_run_final', True))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def run_b2w(self):
        pass

    def inspect_b2w(self):
        pass

    def run_bands(self):
        pass

    def inspect_bands(self):
        pass

    def run_a2f(self):
        pass

    def inspect_a2f(self):
        pass

    def run_bte(self):
        pass

    def inspect_bte(self):
        pass

    def results(self):
        pass

    def on_terminated(self):
        pass