
from aiida import orm
import re
import tempfile
from aiida.common.exceptions import NotExistentAttributeError
from .constants import THZ_TO_MEV

def get_qpoints_and_frequencies(
    wc: orm.WorkChainNode,
    success_code = [0],
    ):
    pattern_q = re.compile(
        r'''q\s*=\s*
        \(\s*
        ([+-]?\d+\.\d+)
        \s+([+-]?\d+\.\d+)
        \s+([+-]?\d+\.\d+)
        \s*\)
        ''',
        re.VERBOSE
    )

    pattern_freq = re.compile(
        r'''freq\s*\( \s*\d+ \s*\)
        \s*=\s*
        ([+-]?\d+\.\d+)
        \s*\[THz\]\s*=\s*
        ([+-]?\d+\.\d+)
        \s*\[cm-1\]
        ''',
        re.VERBOSE
        )

    if wc.process_label == 'EpwWorkChain':
        wc_ph = wc.base.links.get_outgoing(link_label_filter='ph_base').first().node
    elif wc.process_label == 'PhBaseWorkChain':
        wc_ph = wc
    else:
        wc_ph = wc.base.links.get_outgoing(link_label_filter='ph_base').first().node
        if wc_ph is not None:
            raise Warning(
                'This workchain is unknown, but it called a PhBaseWorkChain'
                )
        else:
            raise ValueError('Invalid input workchain')

    if not (wc_ph.exit_status in success_code):
        raise ValueError('The PhBaseWorkChain failed')

    remote_folder = wc_ph.outputs.remote_folder
    try:
        retrieved = wc_ph.outputs.retrieved
        if 'DYN_MAT' not in retrieved.list_object_names():
            raise ValueError('DYN_MAT/ not found in the retrieved list.')

        dyn0 = retrieved.get_object_content('DYN_MAT/dynamical-matrix-0')

    except NotExistentAttributeError:
        print(f'⚠️ Node<{wc_ph.pk}> outputs not retrieved. Directly query remote folder.')
        if 'DYN_MAT' not in remote_folder.listdir():
            raise ValueError('No DYN_MAT found.')

        with remote_folder.computer.get_transport() as transport:
            transport.chdir(remote_folder.get_remote_path())
            with tempfile.NamedTemporaryFile(mode='r') as dyn0:
                transport.get('DYN_MAT/dynamical-matrix-0', dyn0.name)
                dyn0 = dyn0.read()
    q_list = []
    freq_list = []

    lines = dyn0.strip().splitlines()
    nirrqpts = int(lines[1].strip())

    for iq in range(1, 1+nirrqpts):
        try:
            retrieved = wc_ph.outputs.retrieved
            dyn_file = retrieved.get_object_content(f'DYN_MAT/dynamical-matrix-{iq}')
        except AttributeError:
            with remote_folder.computer.get_transport() as transport:
                transport.chdir(remote_folder.get_remote_path())
                with tempfile.NamedTemporaryFile(mode='r') as dyn_file:
                    transport.get(f'DYN_MAT/dynamical-matrix-{iq}', dyn_file.name)
                    dyn_file = dyn_file.read()
        # lines = dyn_file.strip().splitlines()
        lines = dyn_file
        # for line in lines:
        q_match = pattern_q.search(lines)
        freq_match = pattern_freq.findall(lines)
        if q_match:
            qx, qy, qz = q_match.groups()
            q_list.append([float(qx), float(qy), float(qz)])
        if freq_match:
            # thz, _ = freq_match.groups()
            freq_list.append([float(thz) for thz, _ in freq_match])

    return q_list, freq_list

def get_negative_frequencies(
    wc: orm.WorkChainNode,
    tolerance = -0.01,
    success_code = [0],
    ):
    is_stable = True
    negative_freqs = []

    qs, freqs = get_qpoints_and_frequencies(wc, success_code)
    q0 = qs[0]
    neg_freq0 = [f * THZ_TO_MEV for f in freqs[0] if f < tolerance]
    if len(neg_freq0) > 3:
        is_stable = False
        negative_freqs.append((q0, neg_freq0))
    for q, freq in zip(qs[1:], freqs[1:]):
        neg_freq = [f * THZ_TO_MEV for f in freq if f < tolerance]
        if len(neg_freq) > 0:
            is_stable = False
            negative_freqs.append((q, neg_freq))

    return is_stable, negative_freqs


def get_phonon_wc_from_epw_wc(
    epw_wc: orm.WorkChainNode
    ) -> orm.WorkChainNode:

    if epw_wc.process_label == 'EpwWorkChain':
        try:
            ph_base_wc = epw_wc.base.links.get_outgoing(link_label_filter='ph_base').first().node
            return ph_base_wc
        except Exception as e:
            try:
                epw_calcjob = epw_wc.base.links.get_outgoing(link_label_filter='epw').first().node
                ph_base_wc = epw_calcjob.inputs.parent_folder_ph.creator.caller
                return ph_base_wc
            except Exception as e:
                raise ValueError(f"Failed to get phonon workchain from EpwWorkChain {epw_wc.pk}: {e}")
    else:
        raise ValueError('Invalid input workchain')