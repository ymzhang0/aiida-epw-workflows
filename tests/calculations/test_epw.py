import pytest

from aiida import orm
from aiida.common import datastructures
from aiida.engine import run_get_node
from aiida.common.exceptions import InputValidationError
from aiida.common.warnings import AiidaDeprecationWarning

def test_epw_default(fixture_sandbox, generate_calc_job, generate_inputs_epw, file_regression):
    """Test a default `EpwCalculation`."""
    entry_point_name = 'quantumespresso.epw'

    inputs = generate_inputs_epw()
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)
    upf = inputs['pseudos']['Si']

    cmdline_params = ['-in', 'aiida.in']
    local_copy_list = [(upf.uuid, upf.filename, './pseudo/Si.upf')]
    retrieve_list = ['aiida.out', './out/aiida.save/data-file-schema.xml', './out/aiida.save/data-file.xml', 'CRASH']
    retrieve_temporary_list = [['./out/aiida.save/K*[0-9]/eigenval*.xml', '.', 2]]

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == cmdline_params
    assert sorted(calc_info.local_copy_list) == sorted(local_copy_list)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
    assert sorted(calc_info.retrieve_temporary_list) == sorted(retrieve_temporary_list)
    assert sorted(calc_info.remote_symlink_list) == sorted([])

    with fixture_sandbox.open('aiida.in') as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox.get_content_list()) == sorted(['aiida.in', 'pseudo', 'out'])
    file_regression.check(input_written, encoding='utf-8', extension='.in')

def test_epw_invalid_inputs(epw_calculation):
    """Test EPW calculation with invalid inputs."""
    with pytest.raises(ValueError):
        run_get_node(epw_calculation)

def test_epw_prepare_for_submission(epw_calculation, epw_inputs):
    """Test the prepare_for_submission method."""
    calc_info, _ = epw_calculation.prepare_for_submission(epw_inputs)
    assert 'cmdline_params' in calc_info
    assert 'input_filename' in calc_info
    assert 'output_filename' in calc_info