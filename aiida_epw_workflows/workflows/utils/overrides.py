from aiida import orm


from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain, Wannier90OptimizeWorkChain

import collections.abc


def recursive_copy(left: dict, right: dict) -> dict:
    """Recursively merge two dictionaries into a single dictionary.

    If any key is present in both ``left`` and ``right`` dictionaries, the value from the ``right`` dictionary is
    assigned to the key.

    :param left: first dictionary
    :param right: second dictionary
    :return: the recursively merged dictionary
    """
    import collections

    # Note that a deepcopy is not necessary, since this function is called recusively.
    right = right.copy()

    for key, value in left.items():
        if key in right:
            if isinstance(value, collections.abc.Mapping) and isinstance(right[key], collections.abc.Mapping):
                right[key] = recursive_merge(value, right[key])

    merged = left.copy()
    merged.update(right)

    return merged

def recursive_merge(namespace, dict_to_merge):
    """

    :param builder_or_ns: The builder-like object to merge into (will be modified in place).
    :param data_dict: A Python dictionary containing the data to merge.
    """
    # We still need to check the source data_dict explicitly
    if not isinstance(dict_to_merge, collections.abc.Mapping):
        raise TypeError('The data to merge must be a dictionary-like object (a Mapping).')

    for key, value in dict_to_merge.items():
        # The key check to see if we should recurse.
        is_nested_merge = (
            key in namespace and
            isinstance(namespace[key], collections.abc.Mapping) and
            isinstance(value, collections.abc.Mapping)
        )

        if is_nested_merge:
            recursive_merge(namespace[key], value)
        else:
            # Otherwise, the new value simply overwrites the old one.
            namespace[key] = value

    return namespace

def get_parent_folder_chk_from_w90_workchain(
    workchain: orm.WorkChainNode,
    ) -> orm.RemoteData:

    if workchain.process_class is Wannier90OptimizeWorkChain:
        if hasattr(workchain.inputs, 'optimize_disproj') and workchain.inputs.optimize_disproj:
            parent_folder_chk = workchain.outputs.wannier90_optimal.remote_folder
        else:
            parent_folder_chk = workchain.outputs.wannier90.remote_folder
    elif workchain.process_class is Wannier90BandsWorkChain:
        parent_folder_chk = workchain.outputs.wannier90.remote_folder
    else:
        raise ValueError(f"Workchain {workchain.process_label} not supported")

    return parent_folder_chk
