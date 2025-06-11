from pathlib import Path

from aiida import orm
from aiida.common import AttributeDict

import logging


from aiida.engine import PortNamespace, ProcessBuilder, WorkChain, ToContext, if_, while_
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from aiida_wannier90_workflows.workflows import Wannier90BaseWorkChain, Wannier90BandsWorkChain, Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.workflows.optimize import validate_inputs as validate_inputs_w90_intp
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints
from aiida_wannier90_workflows.common.types import WannierProjectionType

def get_overrides_from_w90_workchain(
    w90_intp: orm.WorkChainNode,
    ) -> orm.Dict:
    
    assert w90_intp.is_finished_ok, "Wannier90Optimize workchain did not finish successfully"
    assert w90_intp.process_class in (
        Wannier90OptimizeWorkChain,
    ), "Wannier90OptimizeWorkChain expected"
    
    scf = w90_intp.outputs.scf
    nscf = w90_intp.outputs.nscf

    assert not nscf.remote_folder.is_cleaned, "Nscf remote folder is cleaned"
    
    if w90_intp.process_class is Wannier90OptimizeWorkChain:
        if hasattr(w90_intp.inputs, 'optimize_disproj') and w90_intp.inputs.optimize_disproj:
            parent_folder_chk = w90_intp.outputs.wannier90_optimal.remote_folder
        else:
            parent_folder_chk = w90_intp.outputs.wannier90.remote_folder
    else:
        parent_folder_chk = w90_intp.outputs.wannier90.remote_folder
    
    wannier_params = w90_intp.inputs.wannier90.wannier90.get('parameters')
    
    kpoints_nscf = orm.KpointsData()
    kpoints_nscf.set_kpoints_mesh(wannier_params.get('mp_grid'))
    
    w90_overrides = orm.Dict({
        'kpoints_nscf': kpoints_nscf,
        'parent_folder_scf': scf.remote_folder,
        'parent_folder_nscf': nscf.remote_folder,
        'parent_folder_chk': parent_folder_chk,
        'parameters': wannier_params
        })
    
    return w90_overrides


def update_epw_from_w90_overrides(
    inputs: AttributeDict,
    w90_chk_to_ukk_script: orm.RemoteData,
    w90_overrides: orm.Dict,
    ):
    
    parent_folder_nscf = w90_overrides['parent_folder_nscf']
    parent_folder_chk = w90_overrides['parent_folder_chk']
    
    parameters = inputs.parameters.get_dict()
    
    wannier_params = w90_overrides['parameters']
    
    exclude_bands = wannier_params.get('exclude_bands', None) #TODO check this!
    
    if exclude_bands:
        parameters['INPUTEPW']['bands_skipped'] = f'exclude_bands = {exclude_bands[0]}:{exclude_bands[-1]}'

    parameters['INPUTEPW']['nbndsub'] = wannier_params['num_wann']
    

    wannier_chk_path = Path(parent_folder_chk.get_remote_path(), 'aiida.chk')
    nscf_xml_path = Path(parent_folder_nscf.get_remote_path(), 'out/aiida.xml')

    prepend_text = inputs.metadata.options.get('prepend_text', '')
    prepend_text += f'\n{w90_chk_to_ukk_script.get_remote_path()} {wannier_chk_path} {nscf_xml_path} aiida.ukk'
    
    inputs.parameters = orm.Dict(parameters)
    inputs.parent_folder_nscf = parent_folder_nscf
    inputs.metadata.options.prepend_text = prepend_text


def update_epw_from_w90_intp(
    inputs: AttributeDict,
    w90_chk_to_ukk_script: orm.RemoteData,
    w90_intp: orm.WorkChainNode,
    ):
    
    parent_folder_nscf = w90_intp.outputs.nscf.remote_folder
    
    if w90_intp.process_class is Wannier90OptimizeWorkChain:
        if hasattr(w90_intp.inputs, 'optimize_disproj') and w90_intp.inputs.optimize_disproj:
            parent_folder_chk = w90_intp.outputs.wannier90_optimal.remote_folder
            wannier_params = w90_intp.inputs.wannier90_optimal.wannier90.get('parameters')
        else:
            parent_folder_chk = w90_intp.outputs.wannier90.remote_folder
            wannier_params = w90_intp.inputs.wannier90.wannier90.get('parameters')
    else:
        parent_folder_chk = w90_intp.outputs.wannier90.remote_folder
        wannier_params = w90_intp.inputs.wannier90.wannier90.get('parameters')
    
    parameters = inputs.parameters.get_dict()
        
    exclude_bands = wannier_params.get('exclude_bands', None) #TODO check this!
    
    if exclude_bands:
        parameters['INPUTEPW']['bands_skipped'] = f'exclude_bands = {exclude_bands[0]}:{exclude_bands[-1]}'

    parameters['INPUTEPW']['nbndsub'] = wannier_params['num_wann']
    
    wannier_chk_path = Path(parent_folder_chk.get_remote_path(), 'aiida.chk')
    nscf_xml_path = Path(parent_folder_nscf.get_remote_path(), 'out/aiida.xml')

    prepend_text = inputs.metadata.options.get('prepend_text', '')
    prepend_text += f'\n{w90_chk_to_ukk_script.get_remote_path()} {wannier_chk_path} {nscf_xml_path} aiida.ukk'
    
    inputs.parent_folder_nscf = parent_folder_nscf
    inputs.parameters = orm.Dict(parameters)
    inputs.metadata.options.prepend_text = prepend_text
    
    return None

def get_overrides_from_ph_base(
    ph_base: orm.WorkChainNode,
    ) -> orm.Dict:
    
    assert ph_base.is_finished_ok, "PhBase workchain did not finish successfully"
    
    assert ph_base.process_class is PhBaseWorkChain, "PhBaseWorkChain expected"
    
    assert 'remote_stash' in ph_base.outputs, "ph remote stash is cleaned"

    assert not ph_base.outputs.remote_folder.is_cleaned, "ph remote folder is cleaned"
    
    if 'qpoints' in ph_base.inputs:
        qpoints = ph_base.inputs.qpoints
    else:
        create_qpoints_from_distance = ph_base.base.links.get_outgoing(
            link_label_filter='create_qpoints_from_distance'
            ).first().node
        
        qpoints = create_qpoints_from_distance.outputs.qpoints
        
    ph_base_overrides = orm.Dict({
        'parent_folder_ph': ph_base.outputs.remote_folder,
        'qpoints': qpoints
        })
    
    return ph_base_overrides

def update_epw_from_ph_overrides(
    inputs: AttributeDict,
    ph_base_overrides: orm.Dict,
    ):
    
    parent_folder_ph = ph_base_overrides['parent_folder_ph']
    
    inputs.parent_folder_ph = parent_folder_ph
    
    return None

def update_epw_from_ph_base(
    inputs: AttributeDict,
    ph_base: orm.WorkChainNode,
    ):
    
    inputs.parent_folder_ph = ph_base.outputs.remote_folder
    
    return None
