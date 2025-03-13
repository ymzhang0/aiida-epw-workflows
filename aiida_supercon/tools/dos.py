import json
import numpy as np
import matplotlib.pyplot as plt

from monty.json import MontyEncoder

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rcParams['axes.facecolor'] = 'white'

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

ANGULAR_NAME_MAP = {
    0: 'S',
    1: 'P',
    2: 'D',
    3: 'F',
}


def get_fermi_dos(pdos_data):

    fermi_level = pdos_data['fermi_energy']

    energy = np.array(pdos_data['tdos']['energy | eV']['data'])
    dos = np.array(pdos_data['tdos']['values']['dos | states/eV']['data'])

    return np.interp(fermi_level, energy, dos)


def get_dos_data(pdos_wc):

    # PDOS work chain processing
    dos_fermi_level = pdos_wc.outputs.nscf.output_parameters.get_dict()['fermi_energy']
    xlabel, energy_dos, energy_units = pdos_wc.outputs.dos.output_dos.get_x()
    tdos_values = {f'{n} | {u}': v for n, v, u in pdos_wc.outputs.dos.output_dos.get_y()}

    # Conversion into suitable format for widget
    pdos_orbitals = {}

    for orbital, pdos, energy in pdos_wc.outputs.projwfc.projections.get_pdos():
        orbital_data = orbital.get_orbital_dict()
        kind_name = orbital_data['kind_name']
        try:
            orbital_name = orbital.get_name_from_quantum_numbers(
                orbital_data['angular_momentum'], orbital_data['magnetic_number']
            )
        except AttributeError:
            orbital_name = ANGULAR_NAME_MAP[orbital_data['angular_momentum']]
            orbital_name += f'j{orbital_data["total_angular_momentum"]}'
            orbital_name += f'm_j{orbital_data["magnetic_number"]}'
            orbital_name += f'r{orbital_data["radial_nodes"]}'

        pdos_orbitals.setdefault(kind_name, {})
        pdos_orbitals[kind_name][orbital_name] = {
            'energy | eV': energy,
            'pdos | states/eV': pdos
        }

    return {
        'fermi_energy': dos_fermi_level,
        'tdos': {
            f'energy | {energy_units}': energy_dos,
            'values': tdos_values
        },
        'pdos': pdos_orbitals
    }

def get_widget_data(bands_wc, pdos_wc, verbose=False):

    # Bands work chain processing
    bands_fermi_level = bands_wc.outputs.band_parameters.get_dict()['fermi_energy']
    band_structure = bands_wc.outputs.band_structure
    bands_dict = json.loads(band_structure._exportcontent('json')[0])
    bands_dict['fermi_level'] = bands_fermi_level

    # PDOS work chain processing
    dos_fermi_level = pdos_wc.outputs.nscf.output_parameters.get_dict()['fermi_energy']
    xlabel, energy_dos, energy_units = pdos_wc.outputs.dos.output_dos.get_x()
    tdos_values = {f'{n} | {u}': v for n, v, u in pdos_wc.outputs.dos.output_dos.get_y()}

    # Conversion into suitable format for widget
    pdos_orbitals = {}

    for orbital, pdos, energy in pdos_wc.outputs.projwfc.projections.get_pdos():
        orbital_data = orbital.get_orbital_dict()
        kind_name = orbital_data['kind_name']
        try:
            orbital_name = orbital.get_name_from_quantum_numbers(
                orbital_data['angular_momentum'], orbital_data['magnetic_number']
            )
        except AttributeError:
            orbital_name = ANGULAR_NAME_MAP[orbital_data['angular_momentum']]
            orbital_name += f'j{orbital_data["total_angular_momentum"]}'
            orbital_name += f'm_j{orbital_data["magnetic_number"]}'
            orbital_name += f'r{orbital_data["radial_nodes"]}'

        pdos_orbitals.setdefault(kind_name, {})
        pdos_orbitals[kind_name][orbital_name] = {
            'energy | eV': energy,
            'pdos | states/eV': pdos
        }

    full_dos_dict = {
        'fermi_energy': dos_fermi_level,
        'tdos': {
            f'energy | {energy_units}': energy_dos,
            'values': tdos_values
        },
        'pdos': pdos_orbitals
    }

    if verbose:
        # Print all orbitals for each element
        for element, orbital_data in pdos_orbitals.items():
            print(f'Element {element}: {list(orbital_data.keys())}')

    # Final conversion into format with `MontyEncoder`
    bands_data = json.loads(json.dumps(bands_dict, cls=MontyEncoder))
    pdos_data = json.loads(json.dumps(full_dos_dict, cls=MontyEncoder))

    return bands_data, pdos_data, pdos_orbitals

def get_number_of_electrons_from_pseudos(structure, pseudo_family): 
    """Return the number of electrons that will be present in the structure for each element using pseudos from the given pseudo family.
    :param structure: a ``StructureData`` node
    :param pseudos_family: AiiDA group of ``UpfData`` of a given pseudo family
    :returns: the number of electrons and kind charge
    """
    kind_charge = {}
    number_electrons = 0
    for kind in structure.kinds:
        z_valence = pseudo_family.get_pseudo(kind.symbol).z_valence
        kind_charge[kind.name] = z_valence
    for site in structure.sites:
        number_electrons += kind_charge[site.kind_name]
    return number_electrons, kind_charge

def get_pdos(pdos_orbitals, element, orbital):

    pdos = {}

    for orbi, data in pdos_orbitals[element].items():
        if orbital in orbi:
            if len(pdos) == 0:
                pdos = data
            else:
                pdos['pdos | states/eV'] += data['pdos | states/eV']
    return pdos

def get_states(pdos_data, energy_range):
    
    energy = np.array(pdos_data['tdos']['energy | eV']['data'])
    int_dos = np.array(pdos_data['tdos']['values']['integrated_dos | states']['data'])

    energy_range.append(pdos_data['fermi_energy'])
    
    states = np.interp(energy_range, energy, int_dos)

    return states[1] - states[0], states[2]

def get_super_metric(pdos_data, super_fermi_range=0.1, considered_states_range=1, states_above_thresh=1e-2):

    fermi_level = pdos_data['fermi_energy']

    states_above, _ = get_states(
        pdos_data, [fermi_level, fermi_level + super_fermi_range]
    )
    states_below, _ = get_states(
        pdos_data, [fermi_level - super_fermi_range, fermi_level]
    )

    if states_above < states_above_thresh or states_below < states_above_thresh:
        return 0

    fermi_states, _ = get_states(
        pdos_data, [fermi_level - super_fermi_range, fermi_level + super_fermi_range]
    )
    considered_states, _ = get_states(
        pdos_data, [fermi_level - considered_states_range, fermi_level + considered_states_range]
    )

    return fermi_states / considered_states
