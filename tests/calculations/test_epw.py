# In tests/calculations/test_epw.py
from aiida.common.folders import SandboxFolder

def test_epw_default(generate_calc_job, generate_inputs_epw, file_regression):
    """Test a default `EpwCalculation`."""
    entry_point_name = 'quantumespresso.epw'
    inputs = generate_inputs_epw()
    process = generate_calc_job(entry_point_name, inputs)

    with SandboxFolder() as sandbox:
        calc_info = process.prepare_for_submission(sandbox)
        # Check that the generated input file is correct
        file_regression.check(sandbox.get_content(process.DEFAULT_INPUT_FILE))

def test_epw_invalid_inputs(generate_calc_job):
    """Test `EpwCalculation` with invalid inputs."""
    # CORRECTED: Use the existing fixture `generate_calc_job`
    process = generate_calc_job('quantumespresso.epw')
    # This is a placeholder for a test where you would provide invalid inputs
    # and assert that AiiDA raises a ValueError upon submission.
    assert process is not None

def test_epw_prepare_for_submission(generate_calc_job, generate_inputs_epw):
    """Test the `prepare_for_submission` method of `EpwCalculation`."""
    # CORRECTED: Use the existing fixture `generate_calc_job`
    inputs = generate_inputs_epw()
    process = generate_calc_job('quantumespresso.epw', inputs)
    # This is a placeholder for a more detailed test.
    assert process is not None