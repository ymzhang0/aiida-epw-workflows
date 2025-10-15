
from aiida import orm
from aiida.engine import WorkChain, ToContext, if_
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.common import AttributeDict, LinkType, NotExistentAttributeError

from .b2w import EpwB2WWorkChain
from .bands import EpwBandsWorkChain

from .ibte import EpwIBTEWorkChain
from .a2f import EpwA2fWorkChain

class EpwTransportWorkChain(ProtocolMixin, WorkChain):
    """Workchain to calculate transport properties using EPW."""

    _NAMESPACE = 'transport'
    _B2W_NAMESPACE = EpwB2WWorkChain._NAMESPACE
    _BANDS_NAMESPACE = EpwBandsWorkChain._INTP_NAMESPACE
    _A2F_NAMESPACE = EpwA2fWorkChain._INTP_NAMESPACE
    _IBTE_NAMESPACE = EpwIBTEWorkChain._INTP_NAMESPACE


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
            if_(cls.should_run_ibte)(
                cls.run_ibte,
                cls.inspect_ibte,
            ),
            cls.results
        )

        spec.expose_outputs(
            EpwB2WWorkChain,
            namespace=cls._B2W_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the anisotropic `EpwCalculation`.'
            }
        )

        spec.expose_outputs(
            EpwBandsWorkChain,
            namespace=cls._BANDS_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwBandsWorkChain`.'
            }
        )

        spec.expose_outputs(
            EpwA2fWorkChain,
            namespace=cls._A2F_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwA2fWorkChain`.'
            }
        )
        spec.expose_outputs(
            EpwIBTEWorkChain,
            namespace=cls._IBTE_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwIBTEWorkChain`.'
            }
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_B2W',
            message='The `b2w` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_BANDS',
            message='The `bands` sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_A2F',
            message='The `a2f` sub process failed')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_IBTE',
            message='The `ibte` sub process failed')

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
            (cls._A2F_NAMESPACE, EpwA2fWorkChain),
            (cls._IBTE_NAMESPACE, EpwIBTEWorkChain),
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

    def setup(self):
        if self._BANDS_NAMESPACE in self.inputs:
            self.ctx.inputs_bands = AttributeDict(
                self.exposed_inputs(
                    EpwBandsWorkChain,
                    namespace=self._BANDS_NAMESPACE
                    )
                )

        if self._A2F_NAMESPACE in self.inputs:
            self.ctx.inputs_a2f = AttributeDict(
                self.exposed_inputs(
                    EpwA2fWorkChain,
                    namespace=self._A2F_NAMESPACE
                    )
                )

        if self._IBTE_NAMESPACE in self.inputs:
            self.ctx.inputs_ibte = AttributeDict(
                self.exposed_inputs(
                    EpwIBTEWorkChain,
                    namespace=self._IBTE_NAMESPACE
                    )
                )

    def validate_inputs(self):
        pass

    def should_run_b2w(self):
        return self._B2W_NAMESPACE in self.inputs

    def run_b2w(self):
        """Run the b2w workflow."""
        inputs = AttributeDict(
            self.exposed_inputs(
                EpwB2WWorkChain,
                namespace=self._B2W_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = self._B2W_NAMESPACE
        workchain_node = self.submit(EpwB2WWorkChain, **inputs)

        self.report(f'launching EpwB2WWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_b2w=workchain_node)


    def inspect_b2w(self):
        """Inspect the b2w workflow."""
        b2w_workchain = self.ctx.workchain_b2w

        if not b2w_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {b2w_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_b2w,
                EpwB2WWorkChain,
                namespace=self._B2W_NAMESPACE
            )
        )

    def should_run_bands(self):
        """Check if the bands workflow should continue or not."""
        if self._BANDS_NAMESPACE in self.inputs:
            if self.should_run_b2w():
                b2w_workchain = self.ctx.workchain_b2w

                parent_folder_epw = b2w_workchain.outputs.epw.remote_folder

                self.ctx.inputs_bands[self._BANDS_NAMESPACE].parent_folder_epw = parent_folder_epw

            return True
        else:
            return False

    def run_bands(self):
        """Run the bands workflow."""
        inputs = self.ctx.inputs_bands
        inputs.structure = self.inputs.structure
        inputs.metadata.call_link_label = self._BANDS_NAMESPACE

        workchain_node = self.submit(EpwBandsWorkChain, **inputs)

        self.report(f'launching EpwBandsWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_bands=workchain_node)

    def inspect_bands(self):
        """Inspect the bands workflow."""
        bands_workchain = self.ctx.workchain_bands

        if not bands_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {bands_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_BANDS

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_bands,
                EpwBandsWorkChain,
                namespace=self._BANDS_NAMESPACE
            )
        )

    def should_run_a2f(self):
        """Check if the a2f workflow should continue or not."""
        if self._A2F_NAMESPACE in self.inputs:
            if self.should_run_b2w():
                b2w_workchain = self.ctx.workchain_b2w
                parent_folder_epw = b2w_workchain.outputs.epw.remote_folder
                self.ctx.inputs_a2f[self._A2F_NAMESPACE].parent_folder_epw = parent_folder_epw
            return True
        else:
            return False

    def run_a2f(self):
        """Run the a2f workflow."""
        inputs = self.ctx.inputs_a2f
        inputs.structure = self.inputs.structure

        inputs.metadata.call_link_label = self._A2F_NAMESPACE
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)

        self.report(f'launching EpwA2fWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_a2f=workchain_node)

    def inspect_a2f(self):
        """Inspect the a2f workflow."""
        a2f_workchain = self.ctx.workchain_a2f
        if not a2f_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {a2f_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_A2F

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_a2f,
                EpwA2fWorkChain,
                namespace=self._A2F_NAMESPACE
            )
        )

    def should_run_ibte(self):
        """Check if the bte workflow should continue or not."""
        if self._IBTE_NAMESPACE in self.inputs:
            if self.should_run_a2f():
                a2f_workchain = self.ctx.workchain_a2f
                parent_folder_epw = a2f_workchain.outputs.remote_folder
                self.ctx.inputs_ibte[self._IBTE_NAMESPACE].parent_folder_epw = parent_folder_epw
            return True
        else:
            return False

    def run_ibte(self):
        """Run the ibte workflow."""
        inputs = self.ctx.inputs_ibte
        inputs.structure = self.inputs.structure
        inputs.metadata.call_link_label = self._IBTE_NAMESPACE
        workchain_node = self.submit(EpwIBTEWorkChain, **inputs)

        self.report(f'launching EpwIBTEWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_ibte=workchain_node)

    def inspect_ibte(self):
        """Inspect the ibte workflow."""
        ibte_workchain = self.ctx.workchain_ibte
        if not ibte_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {ibte_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_IBTE

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_ibte,
                EpwIBTEWorkChain,
                namespace=self._IBTE_NAMESPACE
            )
        )

    def results(self):
        pass

    def on_terminated(self):
        pass