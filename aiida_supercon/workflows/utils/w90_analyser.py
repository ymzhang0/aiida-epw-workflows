from aiida import orm
from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain

def Wannier90WorkChainAnalyser(
    epw_builder,
    w90_intp: orm.WorkChainNode,
    ) -> dict:
    
    assert w90_intp.is_finished_ok, "Wannier90Optimize workchain did not finish successfully"
    assert w90_intp.process_class in (
        Wannier90OptimizeWorkChain,
    ), "Wannier90OptimizeWorkChain expected"
    
    nscf = w90_intp.outputs.nscf

    assert not nscf.outputs.nscf.remote_folder.is_cleaned, "Nscf remote folder is cleaned"
    
    epw_builder.parent_folder_nscf = nscf.remote_folder

    if w90_intp.process_class is Wannier90OptimizeWorkChain:
        if hasattr(w90_intp.inputs, 'optimize_disproj') and w90_intp.inputs.optimize_disproj:
            epw_builder.parent_folder_chk = w90_intp.outputs.wannier90_optimal.remote_folder
        else:
            epw_builder.parent_folder_chk = w90_intp.outputs.wannier90.remote_folder
    else:
        epw_builder.parent_folder_chk = w90_intp.outputs.wannier90.remote_folder

    parameters = w90_intp.parameters.get_dict()


    wannier_params = w90_intp.inputs.wannier90.wannier90.parameters.get_dict()
    exclude_bands = wannier_params.get('exclude_bands', None) #TODO check this!
    if exclude_bands:
        parameters['INPUTEPW']['bands_skipped'] = f'exclude_bands = {exclude_bands[0]}:{exclude_bands[-1]}'

    parameters['INPUTEPW']['nbndsub'] = wannier_params['num_wann']
    epw_builder.parameters = orm.Dict(parameters)

    return
    
def PhBaseWorkChainAnalyser(
    epw_builder,
    ph_base: orm.WorkChainNode,
    ) -> None:
    
    assert ph_base.is_finished_ok, "PhBase workchain did not finish successfully"
    
    assert ph_base.process_class is PhBaseWorkChain, "PhBaseWorkChain expected"
    
    assert not ph_base.outputs.remote_folder.is_cleaned, "Scf remote folder is cleaned"

    epw_builder.parent_folder_ph = ph_base.outputs.remote_folder

    epw_builder.qpoints = ph_base.inputs.qpoints
    
    return