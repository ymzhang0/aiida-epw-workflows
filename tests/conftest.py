"""pytest fixtures for simplified testing."""
import pytest
from aiida import orm
from aiida.engine import ExitCode
from aiida.plugins import WorkflowFactory, CalculationFactory

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]

@pytest.fixture(scope='function')
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder
    with SandboxFolder() as folder:
        yield folder

@pytest.fixture
def generate_structure():
    """Fixture to generate a StructureData instance for testing."""
    def _generate_structure(symbol='Si'):
        from ase.build import bulk
        # CORRECTED: Use the StructureData constructor with the `ase` keyword
        crystal_ase = bulk(symbol, 'diamond', a=5.43)
        structure = orm.StructureData(ase=crystal_ase)
        return structure
    return _generate_structure

@pytest.fixture
def generate_kpoints():
    """Fixture to generate a KpointsData instance."""
    def _generate_kpoints():
        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh([2, 2, 2])
        return kpoints
    return _generate_kpoints


@pytest.fixture
def generate_remote_data(aiida_localhost, tmp_path):
    """Fixture to generate a RemoteData instance for testing."""
    def _generate_remote_data():
        remote_data = orm.RemoteData(computer=aiida_localhost, remote_path=str(tmp_path))
        return remote_data
    return _generate_remote_data

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
def generate_inputs_epw(aiida_local_code_factory, generate_kpoints, generate_remote_data):
    """Generate default inputs for an `EpwCalculation`."""
    def _generate_inputs_epw():
        from aiida_quantumespresso.utils.resources import get_default_options
        parameters = orm.Dict({'INPUT_EPW': {'epbwrite': True, 'epwwrite': True}})

        # CORRECTED: Calls to generate_remote_data() now have no arguments.
        inputs = {
            'code': aiida_local_code_factory('quantumespresso.epw', '/bin/true'),
            'qpoints': generate_kpoints(),
            'kpoints': generate_kpoints(),
            'qfpoints': generate_kpoints(),
            'kfpoints': generate_kpoints(),
            'parent_folder_nscf': generate_remote_data(),
            'parent_folder_ph': generate_remote_data(),
            'parameters': parameters,
            'metadata': {'options': get_default_options()}
        }
        return inputs
    return _generate_inputs_epw

@pytest.fixture
def generate_calc_job(aiida_local_code_factory):
    """Fixture to generate an instance of a CalcJob process."""
    def _generate(entry_point_name, inputs=None):
        if inputs is None:
            inputs = {}
        process_class = CalculationFactory(entry_point_name)
        return process_class(inputs=inputs)
    return _generate

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

