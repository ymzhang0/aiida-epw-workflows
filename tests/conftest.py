"""pytest fixtures for simplified testing."""
import pytest

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]

@pytest.fixture(scope='function')
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder
    with SandboxFolder() as folder:
        yield folder

@pytest.fixture
def generate_inputs_epw(fixture_code, generate_structure, generate_kpoints_mesh, generate_remote_data):
    """Generate default inputs for an `EpwCalculation`."""

    def _generate_inputs_epw():
        """Generate default inputs for an `EpwCalculation`."""
        from aiida.orm import Dict
        from aiida_quantumespresso.utils.resources import get_default_options

        # 创建 EPW 的输入参数
        parameters = Dict({
            'INPUT_EPW': {
                'epbwrite': True,
                'epwwrite': True,
                'nbndsub': 10,
                'elph': True,
                'phons': True,
                'nq1': 2,
                'nq2': 2,
                'nq3': 2,
            }
        })

        # 创建输入字典
        inputs = {
            'code': fixture_code('supercon.epw'),
            'qpoints': generate_kpoints_mesh(2),  # 声子计算用的 q 点
            'kpoints': generate_kpoints_mesh(2),  # 电子计算用的 k 点
            'qfpoints': generate_kpoints_mesh(2), # 精细 q 点网格
            'kfpoints': generate_kpoints_mesh(2), # 精细 k 点网格
            'parent_folder_nscf': generate_remote_data(fixture_localhost, fixture_sandbox.abspath, 'quantumespresso.pw'),
            'parent_folder_ph': generate_remote_data(fixture_localhost, fixture_sandbox.abspath, 'quantumespresso.ph'),
            'parameters': parameters,
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_epw

@pytest.fixture
def generate_calc_job():
    """Fixture to construct a new `CalcJob` instance and call `prepare_for_submission` for testing `CalcJob` classes.

    The fixture will return the `CalcInfo` returned by `prepare_for_submission` and the temporary folder that was passed
    to it, into which the raw input files will have been written.
    """

    def _generate_calc_job(folder, entry_point_name, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return calc_info

    return _generate_calc_job