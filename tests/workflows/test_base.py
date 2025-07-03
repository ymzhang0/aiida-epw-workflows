# In tests/workflows/test_base.py

from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

EpwBaseWorkChain = WorkflowFactory('epw.base')

def test_run_from_scratch_success(generate_builder_epw_base, generate_remote_data):
    """Test running the EpwBaseWorkChain from scratch successfully."""
    # 1. Prepare: Create mock parent folders
    parent_folders = {
        'nscf': generate_remote_data(), # CORRECTED: No arguments needed
        'ph': generate_remote_data(),   # CORRECTED: No arguments needed
        'chk': generate_remote_data(),  # CORRECTED: No arguments needed
        'w90_chk_to_ukk_script': generate_remote_data() # CORRECTED: No arguments needed
    }

    builder = generate_builder_epw_base(parent_folders=parent_folders)

    # 2. Run & 3. Assert
    results, node = run_get_node(builder)
    assert node.is_finished_ok, f"WorkChain failed with exit status {node.exit_status}"

def test_restart_from_epw_parent_folder(generate_builder_epw_base, generate_remote_data):
    """Test restarting from a `parent_folder_epw`."""
    parent_folders = {
        'epw': generate_remote_data() # CORRECTED: No arguments needed
    }

    builder = generate_builder_epw_base(parent_folders=parent_folders)

    results, node = run_get_node(builder)
    assert node.is_finished_ok

def test_error_handler_unrecoverable(generate_builder_epw_base, generate_failed_epw_calculation):
    """Test if the `handle_unrecoverable_failure` process handler works correctly."""
    # 1. Prepare: Create a mock failed calculation
    exit_code = EpwBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE # Example error
    failed_calc = generate_failed_epw_calculation(exit_code=exit_code)

    # Create a workchain instance
    workchain = EpwBaseWorkChain(inputs=generate_builder_epw_base()._inputs(prune=True))

    # 2. Run: Manually call the handler
    result = workchain.handle_unrecoverable_failure(failed_calc)

    # 3. Assert: Check if the handler returned the correct exit code
    assert result is not None
    assert result.status == EpwBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE.status