# In tests/workflows/test_base.py

from aiida.engine import ProcessHandlerReport
from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_epw_workflows.workflows import EpwBaseWorkChain
from aiida.common import AttributeDict
from aiida.orm import StructureData
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

def test_generate_structure(generate_structure):
    """Test `PwBaseWorkChain.generate_structure`."""
    Si = generate_structure('Si')

    assert isinstance(Si, StructureData)
    assert Si.get_formula() == 'Si2'

def test_generate_workchain_pw(generate_workchain_pw, submit_and_await):
    """Test `PwBaseWorkChain.generate_workchain_pw`."""
    process = generate_workchain_pw()

    assert process.setup() is None

    process.validate_kpoints()
    process.prepare_process()

    pw_workchain = process.run_process()['children']
    print(pw_workchain.keys())

    print(pw_workchain.base.links.get_outgoing().all())
    print(pw_workchain.base.links.get_incoming().all())
    assert isinstance(process, PwBaseWorkChain)

def test_generate_inputs_epw(generate_inputs_epw):
    """Test `EpwBaseWorkChain.generate_inputs_epw`."""
    inputs = generate_inputs_epw()
    assert isinstance(inputs, dict)
    assert 'epw' in inputs
    assert 'qpoints' in inputs
    assert 'kpoints' in inputs
    assert 'qfpoints' in inputs
    assert 'kfpoints' in inputs
    assert 'parent_folder_nscf' in inputs
    assert 'parent_folder_ph' in inputs
    assert 'parameters' in inputs
    assert 'metadata' in inputs

def test_setup(generate_workchain_epw_base):
    """Test `PwBaseWorkChain.setup`."""
    process = generate_workchain_epw_base()
    process.setup()

    assert isinstance(process.ctx.inputs, AttributeDict)

def test_handle_unrecoverable_failure(generate_workchain_epw_base):
    """Test `PwBaseWorkChain.handle_unrecoverable_failure`."""
    process = generate_workchain_epw_base(exit_code=EpwCalculation.exit_codes.ERROR_NO_RETRIEVED_FOLDER)
    process.setup()

    result = process.handle_unrecoverable_failure(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert result.exit_code == EpwBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE

    result = process.inspect_process()
    assert result == EpwBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE

