# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from aiida.engine import calcfunction
from aiida.plugins import DataFactory

RemoteData = DataFactory("remote")
RemoteStashFolderData = DataFactory("remote.stash.folder")


@calcfunction
def stash_to_remote(stash_data: RemoteStashFolderData) -> RemoteData:
    """Convert a ``RemoteStashFolderData`` into a ``RemoteData``."""

    if stash_data.get_attribute("stash_mode") != "copy":
        raise NotImplementedError("Only the `copy` stash mode is supported.")

    remote_data = RemoteData()
    remote_data.set_attribute(
        "remote_path", stash_data.get_attribute("target_basepath")
    )
    remote_data.computer = stash_data.computer

    return remote_data


class SuperConWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute the electron-phonon coupling."""

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('epw_folder', valid_type=orm.RemoteStashFolderData)
        spec.input('interpolation_distance', valid_type=(orm.Float, orm.List))

        spec.outline(
            cls.generate_reciprocal_points,
            cls.restart_epw,
            cls.results
        )
        spec.output('epw_folder', valid_type=orm.RemoteData)

        spec.expose_inputs(
            EpwCalculation, namespace='epw', exclude=(
                'parent_folder_ph', 'parent_folder_nscf', 'kfpoints', 'qfpoints'
            ),
            namespace_options={
                'help': 'Inputs for the `EpwCalculation`.'
            }
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / 'supercon.yaml'

    @classmethod
    def get_builder_from_protocol(cls, epw_code, parent_epw, protocol=None, overrides=None, **kwargs):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :TODO:
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        epw_builder = EpwCalculation.get_builder()

        epw_builder.code = epw_code
        epw_inputs = inputs.get('epw', None)

        epw_calc = parent_epw.base.links.get_outgoing(link_label_filter='epw').first().node

        epw_builder.kpoints = epw_calc.inputs.kpoints
        epw_builder.qpoints = epw_calc.inputs.qpoints

        parameters = epw_inputs['parameters']
        parameters['INPUTEPW']['nbndsub'] = epw_calc.inputs.parameters['INPUTEPW']['nbndsub']
        if 'bands_skipped' in epw_calc.inputs.parameters['INPUTEPW']:
            parameters['INPUTEPW']['bands_skipped'] = epw_calc.inputs.parameters['INPUTEPW'].get('bands_skipped')

        epw_builder.parameters = orm.Dict(parameters)
        epw_builder.metadata = epw_inputs['metadata']
        if 'settings' in epw_inputs:
            epw_builder.settings = orm.Dict(epw_inputs['settings'])

        builder = cls.get_builder()
        builder.interpolation_distance = orm.Float(inputs['interpolation_distance'])
        builder.structure = parent_epw.inputs.structure
        builder.epw_folder = parent_epw.outputs.epw_folder
        builder.epw = epw_builder
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def generate_reciprocal_points(self):
        """Generate the qpoints and kpoints meshes for the interpolation."""

        inputs = {
            'structure': self.inputs.structure,
            'distance': self.inputs.interpolation_distance,
            'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
            'metadata': {
                'call_link_label': 'create_qpoints_from_distance'
            }
        }
        inter_points = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        self.ctx.inter_points = inter_points

    def restart_epw(self):
        """Restart the EPW calculation from the input folder."""
        inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='epw'))

        inputs.parent_folder_epw = self.inputs.epw_folder
        inputs.kfpoints = self.ctx.inter_points
        inputs.qfpoints = self.ctx.inter_points

        inputs.metadata.call_link_label = 'epw'

        calcjob_node = self.submit(EpwCalculation, **inputs)
        self.report(f'launching `epw` {calcjob_node.pk}')

        return ToContext(calcjob_epw=calcjob_node)

    def results(self):
        """TODO"""
        self.out('epw_folder', self.ctx.calcjob_epw.outputs.remote_folder)

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

