"""pytest fixtures for simplified testing."""
import pytest
from aiida import orm
from aiida.engine import ExitCode
from aiida.plugins import WorkflowFactory
from aiida.common import LinkType

pytest_plugins = [
    "aiida.manage.tests.pytest_fixtures",
    # "aiida.tools.pytest_fixtures",
    ]

@pytest.fixture(scope='function')
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder
    with SandboxFolder() as folder:
        yield folder

@pytest.fixture(scope='session')
def filepath_tests():
    """Return the absolute filepath of the `tests` folder.

    .. warning:: if this file moves with respect to the `tests` folder, the implementation should change.

    :return: absolute filepath of `tests` folder which is the basepath for all test resources.
    """
    import os
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the directory containing the file `fixtures`."""
    import os
    return os.path.join(filepath_tests, 'fixtures')

@pytest.fixture
def generate_remote_data(aiida_localhost):
    """Return a `RemoteData` node."""

    def _generate_remote_data(entry_point_name=None):
        """Return a `RemoteData` node."""
        from aiida.common.links import LinkType
        from aiida.orm import CalcJobNode, RemoteData
        from aiida.plugins.entry_point import format_entry_point_string

        entry_point = format_entry_point_string('aiida.calculations', entry_point_name)

        remote = RemoteData(computer=aiida_localhost, remote_path='/tmp')

        if entry_point_name is not None:
            creator = CalcJobNode(computer=aiida_localhost, process_type=entry_point)
            creator.set_option('resources', {'num_machines': 1, 'num_mpiprocs_per_machine': 1})
            remote.base.links.add_incoming(creator, link_type=LinkType.CREATE, link_label='remote_folder')
            creator.store()

        return remote

    return _generate_remote_data

@pytest.fixture(scope='session')
def generate_upf_data():
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

    def _generate_upf_data(element):
        """Return `UpfData` node."""
        import io
        from aiida_pseudo.data.pseudo import UpfData
        content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="4.0"\n/></UPF>\n'
        stream = io.BytesIO(content.encode('utf-8'))
        return UpfData(stream, filename=f'{element}.upf')

    return _generate_upf_data

@pytest.fixture
def generate_calc_job(temp_dir):
    """Fixture to construct a new `CalcJob` instance and call `prepare_for_submission` for testing `CalcJob` classes.

    The fixture will return the `CalcInfo` returned by `prepare_for_submission` and the temporary folder that was passed
    to it, into which the raw input files will have been written.
    """

    def _generate_calc_job(entry_point_name, folder=None, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        if not folder:
            folder = temp_dir

        calc_info = process.prepare_for_submission(folder)

        return calc_info

    return _generate_calc_job

@pytest.fixture
def generate_calc_job_node(aiida_localhost, generate_remote_data, filepath_fixtures):
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""
    from collections.abc import Mapping
    def flatten_inputs(inputs, prefix=""):
        """Flatten inputs recursively like :meth:`aiida.engine.processes.process::Process._flatten_inputs`."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + "__"))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(
        entry_point_name="base",
        computer=None,
        test_name=None,
        inputs=None,
        attributes=None,
        retrieve_temporary=None,
        store=True,
    ):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param test_name: relative path of directory with test output files in the `fixtures/{entry_point_name}` folder.
        :param inputs: any optional nodes to add as input links to the corrent CalcJobNode
        :param attributes: any optional attributes to set on the node
        :param retrieve_temporary: optional tuple of an absolute filepath of a temporary directory and a list of
            filenames that should be written to this directory, which will serve as the `retrieved_temporary_folder`.
            For now this only works with top-level files and does not support files nested in directories.
        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node.
        """
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string
        import pathlib
        import shutil

        if computer is None:
            computer = aiida_localhost

        filepath_folder = None

        if test_name is not None:
            for name in ("quantumespresso.", "wannier90."):
                if name in entry_point_name:
                    plugin_name = entry_point_name[len(name) :]
                    break
            filepath_folder = filepath_fixtures / "calcjob" / plugin_name / test_name
            filepath_input = filepath_folder / "aiida.in"

        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.base.attributes.set("input_filename", "aiida.in")
        node.base.attributes.set("output_filename", "aiida.out")
        node.base.attributes.set("error_filename", "aiida.err")
        node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
        node.set_option("max_wallclock_seconds", 1800)

        if attributes:
            node.base.attributes.set_many(attributes)

        if filepath_folder:
            from qe_tools.exceptions import ParsingError

            from aiida_quantumespresso.tools.pwinputparser import PwInputFile

            try:
                with open(filepath_input, encoding="utf-8") as input_file:
                    parsed_input = PwInputFile(input_file.read())
            except (ParsingError, FileNotFoundError):
                pass
            else:
                inputs["structure"] = parsed_input.get_structuredata()
                inputs["parameters"] = orm.Dict(parsed_input.namelists)

        if inputs:
            metadata = inputs.pop("metadata", {})
            options = metadata.get("options", {})

            for name, option in options.items():
                node.set_option(name, option)

            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.base.links.add_incoming(
                    input_node, link_type=LinkType.INPUT_CALC, link_label=link_label
                )

        if store:
            node.store()

        if retrieve_temporary:
            dirpath, filenames = retrieve_temporary
            for filename in filenames:
                shutil.copy(
                    filepath_folder / filename, pathlib.Path(dirpath) / filename
                )

        if filepath_folder:
            retrieved = orm.FolderData()
            retrieved.put_object_from_tree(filepath_folder)

            # Remove files that are supposed to be only present in the retrieved temporary folder
            if retrieve_temporary:
                for filename in filenames:
                    retrieved.delete_object(filename)

            retrieved.base.links.add_incoming(
                node, link_type=LinkType.CREATE, link_label="retrieved"
            )
            retrieved.store()

            remote_folder = generate_remote_data()
            remote_folder.base.links.add_incoming(
                node, link_type=LinkType.CREATE, link_label="remote_folder"
            )
            remote_folder.store()

        return node

    return _generate_calc_job_node

@pytest.fixture
def generate_structure():
    """Return a ``StructureData`` representing either bulk silicon or a water molecule."""

    def _generate_structure(structure_id="Si"):
        """Return a ``StructureData`` representing bulk silicon or a snapshot of a single water molecule dynamics.

        :param structure_id: identifies the ``StructureData`` you want to generate. Either 'Si' or 'H2O' or 'GaAs'.
        """
        from aiida.orm import StructureData

        if structure_id == "Si":
            param = 5.43
            cell = [
                [param / 2.0, param / 2.0, 0],
                [param / 2.0, 0, param / 2.0],
                [0, param / 2.0, param / 2.0],
            ]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
            structure.append_atom(
                position=(param / 4.0, param / 4.0, param / 4.0),
                symbols="Si",
                name="Si",
            )
        elif structure_id == "H2O":
            structure = StructureData(
                cell=[
                    [5.29177209, 0.0, 0.0],
                    [0.0, 5.29177209, 0.0],
                    [0.0, 0.0, 5.29177209],
                ]
            )
            structure.append_atom(
                position=[12.73464656, 16.7741411, 24.35076238], symbols="H", name="H"
            )
            structure.append_atom(
                position=[-29.3865565, 9.51707929, -4.02515904], symbols="H", name="H"
            )
            structure.append_atom(
                position=[1.04074437, -1.64320127, -1.27035021], symbols="O", name="O"
            )
        elif structure_id == "GaAs":
            structure = StructureData(
                cell=[
                    [0.0, 2.8400940897, 2.8400940897],
                    [2.8400940897, 0.0, 2.8400940897],
                    [2.8400940897, 2.8400940897, 0.0],
                ]
            )
            structure.append_atom(position=[0.0, 0.0, 0.0], symbols="Ga", name="Ga")
            structure.append_atom(
                position=[1.42004704485, 1.42004704485, 4.26014113455],
                symbols="As",
                name="As",
            )
        elif structure_id == "BaTiO3":
            structure = StructureData(
                cell=[
                    [3.93848606, 0.0, 0.0],
                    [0.0, 3.93848606, 0.0],
                    [0.0, 0.0, 3.93848606],
                ]
            )
            structure.append_atom(position=[0.0, 0.0, 0.0], symbols="Ba", name="Ba")
            structure.append_atom(
                position=[1.969243028987539, 1.969243028987539, 1.969243028987539],
                symbols="Ti",
                name="Ti",
            )
            structure.append_atom(
                position=[0.0, 1.969243028987539, 1.969243028987539],
                symbols="O",
                name="O",
            )
            structure.append_atom(
                position=[1.969243028987539, 1.969243028987539, 0.0],
                symbols="O",
                name="O",
            )
            structure.append_atom(
                position=[1.969243028987539, 0.0, 1.969243028987539],
                symbols="O",
                name="O",
            )
        else:
            raise KeyError(f"Unknown structure_id='{structure_id}'")
        return structure

    return _generate_structure

@pytest.fixture
def generate_kpoints_mesh():
    """Return a `KpointsData` node."""

    def _generate_kpoints_mesh(npoints):
        """Return a `KpointsData` with a mesh of npoints in each direction."""
        from aiida.orm import KpointsData

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([npoints] * 3)

        return kpoints

    return _generate_kpoints_mesh

@pytest.fixture
def generate_workchain():
    """Generate an instance of a `WorkChain`."""

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with the given entry point and inputs.

        :param entry_point: entry point name of the work chain subclass.
        :param inputs: inputs to be passed to process construction.
        :return: a `WorkChain` instance.
        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        return process

    return _generate_workchain

@pytest.fixture
def generate_inputs_pw(
    aiida_local_code_factory, generate_structure, generate_kpoints_mesh, generate_upf_data
):
    """Generate default inputs for a `PwCalculation."""

    def _generate_inputs_pw():
        """Generate default inputs for a `PwCalculation."""
        from aiida.orm import Dict

        from aiida_quantumespresso.utils.resources import get_default_options

        parameters = Dict(
            {
                "CONTROL": {"calculation": "scf"},
                "SYSTEM": {"ecutrho": 240.0, "ecutwfc": 30.0},
                "ELECTRONS": {
                    "electron_maxstep": 60,
                },
            }
        )
        structure = generate_structure()
        inputs = {
            "code": aiida_local_code_factory("quantumespresso.pw", "/bin/true"),
            "structure": structure,
            "kpoints": generate_kpoints_mesh(2),
            "parameters": parameters,
            'pseudos': {kind: generate_upf_data(kind) for kind in structure.get_kind_names()},
            "metadata": {"options": get_default_options()},
        }
        return inputs

    return _generate_inputs_pw

@pytest.fixture
def generate_inputs_ph(
    generate_calc_job_node,
    generate_structure,
    generate_remote_data,
    aiida_local_code_factory,
    generate_kpoints_mesh
):
    """Generate default inputs for a `PhCalculation."""

    def _generate_inputs_ph(with_output_structure=False):
        """Generate default inputs for a `PhCalculation.

        :param with_output_structure: whether the PwCalculation has a StructureData in its outputs.
            This is needed to test some PhBaseWorkChain logics.
        """
        from aiida.common import LinkType
        from aiida.orm import Dict, RemoteData

        from aiida_quantumespresso.utils.resources import get_default_options

        pw_node = generate_calc_job_node(
            entry_point_name='quantumespresso.pw', inputs={
                'parameters': Dict(),
                'structure': generate_structure()
            }
        )
        remote_folder = generate_remote_data()
        remote_folder.base.links.add_incoming(pw_node, link_type=LinkType.CREATE, link_label='remote_folder')
        remote_folder.store()
        parent_folder = pw_node.outputs.remote_folder

        if with_output_structure:
            structure = generate_structure()
            structure.base.links.add_incoming(pw_node, link_type=LinkType.CREATE, link_label='output_structure')
            structure.store()

        inputs = {
            'code': aiida_local_code_factory('quantumespresso.ph', '/bin/true'),
            'parent_folder': parent_folder,
            'qpoints': generate_kpoints_mesh(2),
            'parameters': Dict({'INPUTPH': {}}),
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_ph

@pytest.fixture
def generate_inputs_epw(
    aiida_local_code_factory,
    generate_kpoints_mesh,
    generate_workchain_pw,
    generate_workchain_ph,
    generate_remote_data
    ):
    """Generate default inputs for an `EpwCalculation`."""
    def _generate_inputs_epw():
        from aiida_quantumespresso.utils.resources import get_default_options
        parameters = orm.Dict({
            'INPUTEPW': {
                'prefix': 'test',
                }
            })

        parent_pw_workchain = generate_workchain_pw(pw_outputs={
            'remote_folder': generate_remote_data()
        })
        parent_ph_workchain = generate_workchain_ph(ph_outputs={
            'remote_folder': generate_remote_data()
        })

        inputs = {
            'code': aiida_local_code_factory('quantumespresso.epw', '/bin/true'),
            'qpoints': generate_kpoints_mesh(2),
            'kpoints': generate_kpoints_mesh(2),
            'qfpoints': generate_kpoints_mesh(6),
            'kfpoints': generate_kpoints_mesh(6),
            'parent_folder_nscf': parent_pw_workchain.outputs.pw.remote_folder,
            'parent_folder_ph': parent_ph_workchain.outputs.ph.remote_folder,
            'parameters': parameters,
            'metadata': {'options': get_default_options()}
        }
        return inputs
    return _generate_inputs_epw

@pytest.fixture
def generate_workchain_pw(
    generate_workchain,
    generate_inputs_pw,
    generate_calc_job_node
):
    """Generate an instance of a ``PwBaseWorkChain``."""

    def _generate_workchain_pw(
        exit_code=None, inputs=None, return_inputs=False, pw_outputs=None
    ):
        """Generate an instance of a ``PwBaseWorkChain``.

        :param exit_code: exit code for the ``PwCalculation``.
        :param inputs: inputs for the ``PwBaseWorkChain``.
        :param return_inputs: return the inputs of the ``PwBaseWorkChain``.
        :param pw_outputs: ``dict`` of outputs for the ``PwCalculation``. The keys must correspond to the link labels
            and the values to the output nodes.
        """
        from plumpy import ProcessState

        from aiida.common import LinkType
        from aiida.orm import Dict

        entry_point = "quantumespresso.pw.base"

        if inputs is None:
            pw_inputs = generate_inputs_pw()
            kpoints = pw_inputs.pop("kpoints")
            inputs = {"pw": pw_inputs, "kpoints": kpoints}

        if return_inputs:
            return inputs

        process = generate_workchain(entry_point, inputs)

        pw_node = generate_calc_job_node(inputs={"parameters": Dict()})
        process.ctx.iteration = 1
        process.ctx.children = [pw_node]

        if pw_outputs is not None:
            for link_label, output_node in pw_outputs.items():
                output_node.base.links.add_incoming(
                    pw_node, link_type=LinkType.CREATE, link_label=link_label
                )
                output_node.store()

        if exit_code is not None:
            pw_node.set_process_state(ProcessState.FINISHED)
            pw_node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain_pw

@pytest.fixture
def generate_workchain_ph(
    generate_workchain,
    generate_inputs_ph,
    generate_calc_job_node
):
    """Generate an instance of a `PhBaseWorkChain`."""

    def _generate_workchain_ph(
        exit_code=None, inputs=None, return_inputs=False, ph_outputs=None
    ):
        from plumpy import ProcessState

        from aiida.common import LinkType
        from aiida.orm import Dict

        entry_point = 'quantumespresso.ph.base'

        if inputs is None:
            ph_inputs = generate_inputs_ph()
            qpoints = ph_inputs.pop('qpoints')
            inputs = {'ph': ph_inputs, 'qpoints': qpoints}

        if return_inputs:
            return inputs

        process = generate_workchain(entry_point, inputs)

        ph_node = generate_calc_job_node(inputs={"parameters": Dict()})
        process.ctx.iteration = 1
        process.ctx.children = [ph_node]

        if ph_outputs is not None:
            for link_label, output_node in ph_outputs.items():
                output_node.base.links.add_incoming(
                    ph_node, link_type=LinkType.CREATE, link_label=link_label
                )
                output_node.store()

        if exit_code is not None:
            ph_node.set_process_state(ProcessState.FINISHED)
            ph_node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain_ph

@pytest.fixture
def generate_workchain_epw_base(
    generate_workchain,
    generate_inputs_epw,
    generate_calc_job_node
    ):
    """Generate an instance of a `EpwBaseWorkChain`."""

    def _generate_workchain_epw_base(exit_code=None, inputs=None, return_inputs=False):
        from plumpy import ProcessState

        entry_point = 'epw.base'

        if inputs is None:
            epw_inputs = generate_inputs_epw()
            qpoints = epw_inputs.pop('qpoints')
            kpoints = epw_inputs.pop('kpoints')
            qfpoints = epw_inputs.pop('qfpoints')
            kfpoints = epw_inputs.pop('kfpoints')

            inputs = {
                'epw': epw_inputs,
                'qpoints': qpoints,
                'kpoints': kpoints,
                'qfpoints': qfpoints,
                'kfpoints': kfpoints
                }

        if return_inputs:
            return inputs

        process = generate_workchain(entry_point, inputs)

        epw_node = generate_calc_job_node()
        process.ctx.iteration = 1
        process.ctx.children = [epw_node]

        if exit_code is not None:
            epw_node.set_process_state(ProcessState.FINISHED)
            epw_node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain_epw_base

# This fixture is for test_base.py
@pytest.fixture
def generate_builder_epw_base(aiida_local_code_factory, generate_structure):
    """Data factory fixture for the EpwBaseWorkChain."""
    def _generate_builder(parent_folders=None):
        EpwBaseWorkChain = WorkflowFactory('epw.base')
        epw_code = aiida_local_code_factory('quantumespresso.epw', '/bin/true')
        structure = generate_structure()

        builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=epw_code,
            structure=structure,
            protocol='fast',
        )
        if parent_folders:
            for name, node in parent_folders.items():
                builder[f'parent_folder_{name}'] = node
        return builder
    return _generate_builder

@pytest.fixture
def generate_failed_epw_calculation(generate_calc_job):
    """Fixture to generate a FAILED EpwCalculation node."""
    def _generate(exit_code: ExitCode):
        # CORRECTED: Call generate_calc_job with the entry point name
        calc_node = generate_calc_job('quantumespresso.epw')
        calc_node.set_process_state('finished')
        calc_node.set_exit_status(exit_code.status)
        calc_node.set_exit_message(exit_code.message)
        return calc_node
    return _generate