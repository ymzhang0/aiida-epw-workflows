# -*- coding: utf-8 -*-

from pathlib import Path

from aiida import orm
from aiida.common import AttributeDict

import warnings

from aiida.engine import ProcessBuilder, WorkChain, ToContext, if_
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from aiida_wannier90_workflows.workflows import Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints
from aiida_wannier90_workflows.common.types import WannierProjectionType

from ..tools.kpoints import is_compatible

from .base import EpwBaseWorkChain


class EpwB2WWorkChain(ProtocolMixin, WorkChain):
    """Main work chain that orchestrates the `Wannier90OptimizeWorkChain`, `PhBaseWorkChain` and `EpwBaseWorkChain`.

    This work chain is designed to transform the electron-phonon matrix from the Bloch basis (coarse grid) to the Wannier basis.
    """

    _QFPOINTS = [1, 1, 1]
    _KFPOINTS_FACTOR = 1

    _forced_parameters = {
        'INPUTEPW': {
          'a2f': False,
          'elph': True,
          'epbread': False,
          'epbwrite': True,
          'epwread': False,
          'epwwrite': True,
          'wannierize': False,
          'fsthick': 100,
          'phonselfen': False,
        }
    }

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
            # 'selecq.fmt',
            'dmedata.fmt',
            'aiida.kgmap',
            'aiida.kmap',
            'aiida.ukk',
            'out/aiida.epmatwp',
            'save'
            ]
        }

    _NAMESPACE = 'b2w'
    _W90_NAMESPACE = 'w90_intp'
    _PH_NAMESPACE = 'ph_base'
    _EPW_NAMESPACE = 'epw'

    _NAMESPACE_LIST = [ _W90_NAMESPACE, _PH_NAMESPACE, _EPW_NAMESPACE]

    @classmethod
    def validate_inputs(cls, inputs, ctx=None):  # pylint: disable=unused-argument
        """Validate the inputs of the entire input namespace of `Wannier90OptimizeWorkChain`."""

        if hasattr(inputs, cls._W90_NAMESPACE):
            validate_inputs_w90(inputs)
        else:
            pass

        return None

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('kpoints_factor_nscf', valid_type=orm.Int, required=False)
        spec.input('qpoints_distance', valid_type=orm.Float, required=False)
        spec.input('qpoints', valid_type=orm.KpointsData, required=False)

        spec.expose_inputs(
            Wannier90OptimizeWorkChain,
            namespace=cls._W90_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
                'nscf.kpoints',
                'nscf.kpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `Wannier90OptimizeWorkChain/Wannier90BandsWorkChain`.'
            }
        )
        spec.expose_inputs(
            PhBaseWorkChain,
            namespace=cls._PH_NAMESPACE,
            exclude=(
                'clean_workdir',
                'qpoints',
                'qpoints_distance',
                'ph.parent_folder'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` that does the `ph.x` calculation.'
            }
        )

        spec.expose_inputs(
            EpwBaseWorkChain,
            namespace=cls._EPW_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
                'qfpoints_distance',
                'qfpoints',
                'kfpoints',
                'kfpoints_factor',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwBaseWorkChain` that does the `epw.x` calculation.'
            }
        )
        # spec.inputs[cls._EPW_NAMESPACE].validator = cls.validate_inputs
        # spec.inputs.validator = cls.validate_inputs

        spec.expose_outputs(
            Wannier90OptimizeWorkChain,
            namespace=cls._W90_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the `Wannier90OptimizeWorkChain`.'
            }
        )
        spec.expose_outputs(
            PhBaseWorkChain,
            namespace=cls._PH_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )
        spec.expose_outputs(
            EpwBaseWorkChain,
            namespace=cls._EPW_NAMESPACE,
        )

        spec.outline(
            cls.generate_reciprocal_points,
            cls.setup,
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
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_BANDS',
            message='The `PwBandsWorkChain` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_WANNIER90',
            message='The `Wannier90BandsWorkChain` sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_PHONON',
            message='The electron-phonon `PhBaseWorkChain` sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_EPW',
            message='The `EpwWorkChain` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._NAMESPACE}.yaml'

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` of the default protocol."""
        from importlib_resources import files
        import yaml

        from . import protocols

        path = files(protocols) / f"{cls._NAMESPACE}.yaml"
        with path.open() as file:
            return yaml.safe_load(file)


    @property
    def namespace_list(self):
        """Return the list of namespaces within this work chain."""
        return self._NAMESPACE_LIST

    @staticmethod
    def set_target_base(
        inputs,
        target_base_prefix,
        ):
        """We stash the outputs of PhBaseWorkChain and EpwBaseWorkChain for possible restart purposes.
        """
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
                'source_list': EpwB2WWorkChain.SOURCE_LIST[target_base_prefix]
                }


    @classmethod
    def get_builder_restart(
        cls,
        from_b2w_workchain=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs extracted from the previous EpwB2WWorkChain.
        """

        from .utils.overrides import get_parent_folder_chk_from_w90_workchain

        # Firstly check if the previous EpwB2WWorkChain is finished_ok

        if from_b2w_workchain.process_class != EpwB2WWorkChain:
            raise TypeError(f"Node <{from_b2w_workchain.pk}> is not of type EpwB2WWorkChain.")

        if from_b2w_workchain.is_finished_ok:
            raise ValueError('There is no reason to restart a finished `EpwB2WWorkChain`')


        # If a previous EpwB2WWorkChain is provided, use it to populate the builder
        builder = from_b2w_workchain.get_builder_restart()

        # Firstly get descendants of the previous EpwB2WWorkChain
        # Then check if the previous EpwB2WWorkChain is finished_ok

        # Get the wannier90 workchain
        from ..tools.links import get_descendants
        from aiida.common.links import LinkType

        descendants = get_descendants(from_b2w_workchain, LinkType.CALL_WORK)

        # The logic here is: if there is not w90 namespace in the previous workchain, the workchain must have been restarted that the wannier90 workchain is popped,
        # or there is one but it is finished_ok, we don't need to restart either.

        if cls._W90_NAMESPACE not in from_b2w_workchain.inputs:
            builder.pop(cls._W90_NAMESPACE)

            builder[cls._PH_NAMESPACE].ph.parent_folder = descendants[cls._W90_NAMESPACE][0].inputs.ph.parent_folder
            builder[cls._EPW_NAMESPACE].parent_folder_nscf = descendants[cls._W90_NAMESPACE][0].inputs.epw.parent_folder_nscf
            builder[cls._EPW_NAMESPACE].parent_folder_chk = descendants[cls._W90_NAMESPACE][0].inputs.epw.parent_folder_chk

        elif descendants[cls._W90_NAMESPACE][0].is_finished_ok:
            builder.pop(cls._W90_NAMESPACE)

            builder[cls._PH_NAMESPACE].ph.parent_folder = descendants[cls._W90_NAMESPACE][0].outputs.scf.remote_folder
            builder[cls._EPW_NAMESPACE].parent_folder_nscf = descendants[cls._W90_NAMESPACE][0].outputs.nscf.remote_folder
            builder[cls._EPW_NAMESPACE].parent_folder_chk = get_parent_folder_chk_from_w90_workchain(descendants[cls._W90_NAMESPACE][0])
        else:
            builder[cls._W90_NAMESPACE]._data = descendants[cls._W90_NAMESPACE][0]._data
            # The logic here is: if there is not ph namespace in the previous workchain, it must have been restarted that the phonon workchain is popped,
            # or there is one but it is finished_ok, we don't need to restart either.

        if cls._PH_NAMESPACE not in from_b2w_workchain.inputs:
            builder.pop(cls._PH_NAMESPACE)

            builder[cls._EPW_NAMESPACE].parent_folder_ph = descendants[cls._PH_NAMESPACE][0].outputs.remote_folder

        elif descendants[cls._PH_NAMESPACE][0].is_finished_ok:
            builder.pop(cls._PH_NAMESPACE)

            builder[cls._EPW_NAMESPACE].parent_folder_ph = descendants[cls._PH_NAMESPACE][0].outputs.remote_folder

        else:
            builder[cls._PH_NAMESPACE]._data = descendants[cls._PH_NAMESPACE][0]._data

        # Get the epw workchain
        if descendants[cls._EPW_NAMESPACE][0].is_finished_ok:
            raise Warning(
                f"The `EpwB2WWorkChain` <{from_b2w_workchain.pk}> is already finished.",
                )

        return builder

    @classmethod
    def get_builder_restart_from_phonon(
        cls,
        codes, protocol=None, overrides=None,
        from_ph_workchain=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs according to the previous PhBaseWorkChain and protocol for other namespaces.
        """

        if from_ph_workchain.process_label != 'PhBaseWorkChain':
            raise ValueError('Currently we only accept a finished `PhBaseWorkChain`')

        inputs = cls.get_protocol_inputs(protocol, overrides)

        from aiida_quantumespresso.calculations.pw import PwCalculation

        if not from_ph_workchain or not from_ph_workchain.is_finished_ok:
            raise ValueError('Currently we only accept a finished `PhBaseWorkChain`')

        # builder = cls.get_builder()
        # builder.pop(cls._PH_NAMESPACE)

        # Currently we assume that the parent folder is created by a `PwCalculation`
        # So a StructureData is always associated with it.
        # If the creator is not a `PwCalculation`, we raise an error.

        parent_calcjob = from_ph_workchain.inputs.ph.parent_folder.creator

        if parent_calcjob.process_label != 'PwCalculation':
            raise ValueError('Cannot find the structure associated with this `PhBaseWorkChain`')

        structure = parent_calcjob.inputs.structure

        builder = cls.get_builder_from_protocol(
            codes=codes,
            structure=structure,
            protocol=protocol,
            overrides=inputs,
            **kwargs
        )

        builder.pop(cls._PH_NAMESPACE)
        # builder.structure = structure
        # w90_intp = Wannier90OptimizeWorkChain.get_builder_from_protocol(
        #     codes=codes,
        #     structure=structure,
        #     protocol=protocol,
        #     overrides=inputs.get(cls._W90_NAMESPACE, {}),
        #     pseudo_family=inputs.get(cls._W90_NAMESPACE, {}).get('pseudo_family', None),
        #     projection_type=kwargs.get('wannier_projection_type', WannierProjectionType.ATOMIC_PROJECTORS_QE),
        #     reference_bands=kwargs.get('reference_bands', None),
        #     bands_kpoints=kwargs.get('bands_kpoints', None),
        # )

        builder[cls._W90_NAMESPACE].pop('projwfc', None)
        builder[cls._W90_NAMESPACE].pop('open_grid', None)
        # builder[cls._W90_NAMESPACE]._data = w90_intp._data

        builder[cls._W90_NAMESPACE].optimize_disproj = orm.Bool(False)

        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._EPW_NAMESPACE, {}),
            **kwargs
        )

        epw_builder.parent_folder_ph = from_ph_workchain.outputs.remote_folder

        builder[cls._EPW_NAMESPACE]._data = epw_builder._data

        if 'qpoints_distance' in from_ph_workchain.inputs:
            builder.qpoints_distance = orm.Float(from_ph_workchain.inputs.qpoints_distance)


        return builder

    @classmethod
    def get_builder_restart_from_wannier90(
        cls,
        codes, protocol=None, overrides=None,
        from_w90_workchain=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs according to the previous Wannier90OptimizeWorkChain and protocol for other namespaces.
        """

        from .utils.overrides import get_parent_folder_chk_from_w90_workchain
        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()
        builder.pop(cls._W90_NAMESPACE)

        if not from_w90_workchain or not from_w90_workchain.is_finished_ok:
            raise ValueError('Currently we only accept a finished `Wannier90OptimizeWorkChain`')

        if from_w90_workchain.outputs.nscf.remote_folder.is_cleaned:
            raise ValueError(
                'The `nscf` remote folder is clean. '
                'Why not restart from scratch?')

        builder.structure = from_w90_workchain.inputs.structure
        builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])
        builder.kpoints_factor_nscf = orm.Int(inputs['kpoints_factor_nscf'])

        ph_base = PhBaseWorkChain.get_builder_from_protocol(
            codes['ph'],
            protocol=protocol,
            overrides=inputs.get(cls._PH_NAMESPACE, {}),
            **kwargs
            )

        ph_base.ph.parent_folder = from_w90_workchain.outputs.scf.remote_folder

        # NOTE: It's not good to use internal ._data property but it's quite convenient here
        # Because for first-time running, we can't provide the parent scf folder so normally we should exclude it from ph_base namespace
        # But if we restart from a finished wannier90 workchain we can provide it so it should not be excluded.
        # TODO: Find a better way to do this

        builder[cls._PH_NAMESPACE]._data = ph_base._data

        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=builder.structure,
            protocol=protocol,
            overrides=inputs.get(cls._EPW_NAMESPACE, {}),
            **kwargs
        )

        epw_builder.parent_folder_nscf = from_w90_workchain.outputs.nscf.remote_folder
        epw_builder.parent_folder_chk = get_parent_folder_chk_from_w90_workchain(from_w90_workchain)
        builder[cls._EPW_NAMESPACE]._data = epw_builder._data

        return builder

    @classmethod
    def get_builder_from_wannier90_and_phonon(
        cls,
        code,
        from_w90_workchain=None,
        from_ph_workchain=None,
        protocol=None,
        overrides=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs according to the previous Wannier90OptimizeWorkChain and PhBaseWorkChain and protocol for other namespaces.
        """

        from .utils.overrides import get_parent_folder_chk_from_w90_workchain
        builder = cls.get_builder()

        builder.structure = from_w90_workchain.inputs.structure

        if not from_w90_workchain or not from_w90_workchain.is_finished_ok:
            raise ValueError('Currently we only accept a finished `Wannier90OptimizeWorkChain`')

        if not from_ph_workchain or not from_ph_workchain.is_finished_ok:
            raise ValueError('Currently we only accept a finished `PhBaseWorkChain`')

        builder.pop(cls._W90_NAMESPACE)
        builder.pop(cls._PH_NAMESPACE)

        if 'qpoints' in from_ph_workchain.inputs:
            builder.qpoints = from_ph_workchain.inputs.qpoints
        elif 'qpoints_distance' in from_ph_workchain.inputs:
            builder.qpoints_distance = orm.Float(from_ph_workchain.inputs.qpoints_distance)

        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=code,
            structure=builder.structure,
            protocol=protocol,
            overrides=overrides.get(cls._EPW_NAMESPACE, {}),
            **kwargs
        )

        epw_builder.parent_folder_nscf = from_w90_workchain.outputs.nscf.remote_folder
        epw_builder.parent_folder_chk = get_parent_folder_chk_from_w90_workchain(from_w90_workchain)
        epw_builder.parent_folder_ph = from_ph_workchain.outputs.remote_folder

        builder[cls._EPW_NAMESPACE]._data = epw_builder._data

        return builder

    @classmethod
    def get_builder_from_protocol(
        cls, codes, structure, protocol=None, overrides=None,
        wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        w90_chk_to_ukk_script=None,
        band_kpoints=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs according to the previous Wannier90OptimizeWorkChain and PhBaseWorkChain and protocol for other namespaces.
        :param codes: A dictionary of codes for the different calculations. Should be in the following format:
            {
                'pw': code pw.x,
                'ph': code ph.x,
                'epw': code epw.x,
                'pw2wannier90': code pw2wannier90.x,
                'wannier': code wannier90.x,
            }
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
        builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])
        builder.kpoints_factor_nscf = orm.Int(inputs['kpoints_factor_nscf'])
        if band_kpoints:
            builder.band_kpoints = band_kpoints
        # Set up the wannier90 sub-workchain
        w90_intp_inputs = inputs.get(cls._W90_NAMESPACE, {})
        pseudo_family = inputs.pop('pseudo_family', None)

        w90_intp = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            codes=codes,
            structure=structure,
            overrides=w90_intp_inputs,
            pseudo_family=pseudo_family,
            projection_type=wannier_projection_type,
            print_summary=False,
        )
        w90_intp.pop('projwfc', None)
        w90_intp.pop('open_grid', None)

        # TODO: Only for testing, will remove later
        w90_intp.optimize_disproj = orm.Bool(False)
        builder[cls._W90_NAMESPACE]._data = w90_intp._data

        # Set up the phonon sub-workchain
        ph_base = PhBaseWorkChain.get_builder_from_protocol(
            codes['ph'],
            protocol=protocol,
            overrides=inputs.get(cls._PH_NAMESPACE, {}),
            **kwargs
            )
        builder[cls._PH_NAMESPACE] = ph_base

        epw = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._EPW_NAMESPACE, {}),
            w90_chk_to_ukk_script=w90_chk_to_ukk_script,
            **kwargs
        )

        builder[cls._EPW_NAMESPACE]._data = epw._data

        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def setup(self):
        """Setup the work chain."""

        inputs = AttributeDict(
            self.exposed_inputs(
                EpwBaseWorkChain,
                namespace=self._EPW_NAMESPACE)
            )

        parameters = inputs.epw.parameters.get_dict()
        for namespace, values in self._forced_parameters.items():
            for key, value in values.items():
                parameters[namespace][key] = value

        inputs.epw.parameters = orm.Dict(parameters)
        self.ctx.inputs = inputs

    def generate_reciprocal_points(self):
        """Generate the reciprocal points."""

        if 'qpoints' in self.inputs:
            qpoints = self.inputs.qpoints
        elif 'qpoints_distance' in self.inputs:
            self.report('Generating q-points and k-points')

            inputs = {
                'structure': self.inputs.structure,
                'distance': self.inputs.qpoints_distance,
                'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_qpoints_from_distance'
                }
            }

            qpoints = create_kpoints_from_distance(**inputs)

        self.ctx.qpoints = qpoints

        if 'kpoints_factor_nscf' in self.inputs:
            qpoints_mesh = qpoints.get_kpoints_mesh()[0]
            kpoints_nscf = orm.KpointsData()
            kpoints_nscf.set_kpoints_mesh([v * self.inputs.kpoints_factor_nscf.value for v in qpoints_mesh])

            self.ctx.kpoints_nscf = kpoints_nscf


    def should_run_wannier90(self):
        """Check if the wannier90 workflow should be run.
        If 'w90_intp' is not in the inputs or the 'kpoints_nscf' is not in the context, it will return False.
        """

        return (
            self._W90_NAMESPACE in self.inputs
            and
            'kpoints_nscf' in self.ctx
            )

    def run_wannier90(self):
        """Run the wannier90 workflow."""

        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace=self._W90_NAMESPACE)
        )

        # TODO: Remove this once we have a better way to handle the kpoints
        try:
            settings = inputs.wannier90.wannier90.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['additional_retrieve_list'] = ['aiida.chk']
        inputs.structure = self.inputs.structure

        inputs.wannier90.wannier90.settings = orm.Dict(settings)

        inputs.metadata.call_link_label = self._W90_NAMESPACE

        set_kpoints(inputs, self.ctx.kpoints_nscf, Wannier90OptimizeWorkChain)

        # TODO: Remove this once we have a better way to handle the kpoints
        inputs.scf.pop('kpoints', None)

        workchain_node = self.submit(Wannier90OptimizeWorkChain, **inputs)
        self.report(f'launching wannier90 work chain {workchain_node.pk}')

        return ToContext(workchain_w90_intp=workchain_node)

    def inspect_wannier90(self):
        """Verify that the wannier90 workflow finished successfully.
        If the wannier90 workflow passed, it will generate the parent folders for the phonon and epw workflows and the outputs of the wannier90 workflow.
        """
        from .utils.overrides import get_parent_folder_chk_from_w90_workchain

        workchain = self.ctx.workchain_w90_intp

        self.ctx.parent_folder_scf = workchain.outputs.scf.remote_folder
        self.ctx.inputs.parent_folder_nscf = workchain.outputs.nscf.remote_folder
        self.ctx.inputs.parent_folder_chk = get_parent_folder_chk_from_w90_workchain(workchain)

        if not workchain.is_finished_ok:
            self.report(f'`Wannier90BandsWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_w90_intp,
                Wannier90OptimizeWorkChain,
                namespace=self._W90_NAMESPACE,
            ),
        )

    def should_run_ph(self):
        """Check if the phonon workflow should be run.
        If 'ph_base' is not in the inputs or the 'qpoints' is not in the context, it will return False.
        """
        return (
            self._PH_NAMESPACE in self.inputs
            and
            'qpoints' in self.ctx
            )

    def run_ph(self):
        """Run the `PhBaseWorkChain`."""
        inputs = AttributeDict(self.exposed_inputs(PhBaseWorkChain, namespace=self._PH_NAMESPACE))

        if 'workchain_w90_intp' in self.ctx:
            inputs.ph.parent_folder = self.ctx.workchain_w90_intp.outputs.scf.remote_folder


        inputs.qpoints = self.ctx.qpoints

        inputs.metadata.call_link_label = self._PH_NAMESPACE

        self.set_target_base(inputs.ph, self._PH_NAMESPACE)
        workchain_node = self.submit(PhBaseWorkChain, **inputs)

        self.report(f'launching `ph` {workchain_node.pk}')

        return ToContext(workchain_ph=workchain_node)

    def inspect_ph(self):
        """Verify that the `PhBaseWorkChain` finished successfully.
        If the phonon workflow passed, it will generate the parent folder for the epw workflow and the outputs of the phonon workflow.
        """
        workchain = self.ctx.workchain_ph

        self.ctx.inputs.parent_folder_ph = workchain.outputs.remote_folder

        if not workchain.is_finished_ok:
            self.report(f'Electron-phonon `PhBaseWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PHONON

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_ph,
                PhBaseWorkChain,
                namespace=self._PH_NAMESPACE,
            ),
        )
    def run_epw(self):
        """Run the `epw.x` calculation."""

        # inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='epw'))
        inputs = self.ctx.inputs

        fpoints = orm.KpointsData()
        fpoints.set_kpoints_mesh(self._QFPOINTS)
        inputs.qfpoints = fpoints
        inputs.kfpoints_factor = orm.Int(self._KFPOINTS_FACTOR)

        inputs.structure = self.inputs.structure

        inputs.metadata.call_link_label = self._EPW_NAMESPACE
        self.set_target_base(inputs.epw, self._EPW_NAMESPACE)
        workchain_node = self.submit(EpwBaseWorkChain, **inputs)
        self.report(f'launching `epw` {workchain_node.pk}')

        return ToContext(workchain_epw=workchain_node)

    def inspect_epw(self):
        """Verify that the `epw.x` calculation finished successfully.
        If the epw workflow passed, it will generate the outputs of the epw workflow.
        """
        workchain = self.ctx.workchain_epw

        if not workchain.is_finished_ok:
            self.report(f'`EpwBaseWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EPW

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_epw,
                EpwBaseWorkChain,
                namespace=self._EPW_NAMESPACE,
            ),
        )
        ## TODO: If the workchain finished OK, we will clean the remote folder:
        ## rm out/aiida.epb*, out/aiida.wfc*, out/aiida.save/*


    def results(self):
        """Add the most important results to the outputs of the work chain.
        """

        pass

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

