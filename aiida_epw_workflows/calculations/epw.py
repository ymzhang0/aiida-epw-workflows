# -*- coding: utf-8 -*-
"""Plugin to create a Quantum Espresso epw.x input file."""
from pathlib import Path

from aiida import orm
from aiida.common import datastructures, exceptions

from aiida_quantumespresso.calculations import _lowercase_dict, _uppercase_dict
from aiida_quantumespresso.calculations.ph import PhCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry

from aiida_quantumespresso.calculations.base import CalcJob


class EpwCalculation(CalcJob):
    """`CalcJob` implementation for the epw.x code of Quantum ESPRESSO."""

    # Keywords that cannot be set by the user but will be set by the plugin
    _blocked_keywords = [('INPUTEPW', 'outdir'), ('INPUTEPW', 'verbosity'), ('INPUTEPW', 'prefix'),
                         ('INPUTEPW', 'dvscf_dir'), ('INPUTEPW', 'amass'), ('INPUTEPW', 'nq1'), ('INPUTEPW', 'nq2'),
                         ('INPUTEPW', 'nq3'), ('INPUTEPW', 'nk1'), ('INPUTEPW', 'nk2'), ('INPUTEPW', 'nk3')]

    _use_kpoints = True

    _compulsory_namelists = ['INPUTEPW']

    # Default input and output files
    _PREFIX = 'aiida'
    _DEFAULT_INPUT_FILE = 'aiida.in'
    _kfpoints_input_file = 'kfpoints.kpt'
    _qfpoints_input_file = 'qfpoints.kpt'
    _DEFAULT_OUTPUT_FILE = 'aiida.out'
    _OUTPUT_XML_TENSOR_FILE_NAME = 'tensors.xml'
    _OUTPUT_A2F_FILE = 'aiida.a2f'
    _OUTPUT_SUBFOLDER = './out/'
    _output_elbands_file = 'band.eig'
    _output_phbands_file = 'phband.freq'
    _FOLDER_SAVE = 'save'
    _FOLDER_DYNAMICAL_MATRIX = 'DYN_MAT'

    # Not using symlink in pw to allow multiple nscf to run on top of the same scf
    _default_symlink_usage = False

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('metadata.options.input_filename', valid_type=str, default=cls._DEFAULT_INPUT_FILE)
        spec.input('metadata.options.output_filename', valid_type=str, default=cls._DEFAULT_OUTPUT_FILE)
        spec.input('metadata.options.withmpi', valid_type=bool, default=True)
        spec.input('kpoints', valid_type=orm.KpointsData, help='coarse kpoint mesh')
        spec.input('qpoints', valid_type=orm.KpointsData, help='coarse qpoint mesh')
        spec.input('kfpoints', valid_type=orm.KpointsData, help='fine kpoint mesh')
        spec.input('qfpoints', valid_type=orm.KpointsData, help='fine qpoint mesh')
        spec.input('parameters', valid_type=orm.Dict, help='')
        spec.input('settings', valid_type=orm.Dict, required=False, help='')
        spec.input('parent_folder_nscf', required=False, valid_type=orm.RemoteData,
                   help='the folder of a completed nscf `PwCalculation`')
        spec.input('parent_folder_ph', required=False, valid_type=orm.RemoteData,
                   help='the folder of a completed `PhCalculation`')
        spec.input('parent_folder_epw', required=False, valid_type=(orm.RemoteData, orm.RemoteStashFolderData),
                   help='folder that contains all files required to restart an `EpwCalculation`')
        spec.inputs['metadata']['options']['parser_name'].default = 'quantumespresso.epw'

        spec.output('output_parameters', valid_type=orm.Dict,
            help='The `output_parameters` output node of the successful calculation.')
        spec.output('max_eigenvalue', valid_type=orm.XyData, required=False,
            help='The temperature dependence of the max eigenvalue.')

        spec.output('dos', valid_type=orm.XyData, required=False,
            help='The electron density of states.')
        spec.output('phdos', valid_type=orm.XyData, required=False,
            help='The phonon density of states.')
        spec.output('phdos_proj', valid_type=orm.XyData, required=False,
            help='The phonon density of states projected on the atomic orbitals.')
        spec.output('a2f', valid_type=orm.XyData, required=False,
            help='The contents of the `.a2f` file.')
        spec.output('a2f_proj', valid_type=orm.XyData, required=False,
            help='The contents of the `.a2f_proj` file.')
        spec.output('lambda_FS', valid_type=orm.ArrayData, required=False,
            help='The electron-phonon coupling on the Fermi surface.')
        spec.output('lambda_k_pairs', valid_type=orm.XyData, required=False,
            help='The density of the electron-phonon coupling on the k-points.')
        spec.output('el_band_structure', valid_type=orm.BandsData, required=False,
            help='The interpolated electronic band structure.')
        spec.output('ph_band_structure', valid_type=orm.BandsData, required=False,
            help='The interpolated phonon band structure.')
        spec.output('aniso_gap_functions', valid_type=orm.ArrayData, required=False,
            help='The interpolated anisotropic gap function.')

        spec.exit_code(300, 'ERROR_NO_RETRIEVED_FOLDER',
            message='The retrieved folder data node could not be accessed.')
        spec.exit_code(310, 'ERROR_OUTPUT_STDOUT_READ',
            message='The stdout output file could not be read.')
        spec.exit_code(312, 'ERROR_OUTPUT_STDOUT_INCOMPLETE',
            message='The stdout output file was incomplete probably because the calculation got interrupted.')
        spec.exit_code(313, 'ERROR_MEMORY_EXCEEDS_MAX_MEMLT',
            message='The memory required for the calculation exceeds the maximum memory limit.')
        # yapf: enable

    def prepare_for_submission(self, folder):
        """Prepare the calculation job for submission by transforming input nodes into input files.

        In addition to the input files being written to the sandbox folder, a `CalcInfo` instance will be returned that
        contains lists of files that need to be copied to the remote machine before job submission, as well as file
        lists that are to be retrieved after job completion.

        :param folder: a sandbox folder to temporarily write files on disk.
        :return: :class:`~aiida.common.datastructures.CalcInfo` instance.
        """

        # pylint: disable=too-many-statements,too-many-branches, protected-access

        def test_offset(offset):
            """Check if the grid has an offset."""
            if any(i != 0. for i in offset):
                raise NotImplementedError(
                    'Computation of electron-phonon on a mesh with non zero offset is not implemented, '
                    'at the level of epw.x'
                )

        local_copy_list = []
        remote_copy_list = []
        remote_symlink_list = []
        retrieve_list = [self.metadata.options.output_filename]

        parameters = _uppercase_dict(self.inputs.parameters.get_dict(), dict_name='parameters')
        parameters = {k: _lowercase_dict(v, dict_name=k) for k, v in parameters.items()}

        if 'INPUTEPW' not in parameters:
            raise exceptions.InputValidationError('required namelist INPUTEPW not specified')

        if 'settings' in self.inputs:
            settings = _uppercase_dict(self.inputs.settings.get_dict(), dict_name='settings')
        else:
            settings = {}

        # We will have a mixture of remote_copy_list and remote_symlink_list so I modify the following codes
        # remote_list = remote_symlink_list if settings.pop(
        #     'PARENT_FOLDER_SYMLINK', self._default_symlink_usage
        # ) else remote_copy_list
        remote_list = remote_copy_list

        if 'parent_folder_nscf' in self.inputs:
            parent_folder_nscf = self.inputs.parent_folder_nscf

            remote_list.append((
                parent_folder_nscf.computer.uuid,
                Path(parent_folder_nscf.get_remote_path(), PwCalculation._OUTPUT_SUBFOLDER).as_posix(),
                self._OUTPUT_SUBFOLDER,
            ))

        if 'parent_folder_ph' in self.inputs:
            parent_folder_ph = self.inputs.parent_folder_ph

            # Create the save folder with dvscf and dyn files
            folder.get_subfolder(self._FOLDER_SAVE, create=True)

            if 'NUMBER_OF_QPOINTS' in settings:
                nqpt = settings.pop('NUMBER_OF_QPOINTS')
            else:
                # List of IBZ q-point to be added below EPW. To be removed when removed from EPW.
                qibz_ar = []
                for key, value in sorted(parent_folder_ph.creator.outputs.output_parameters.get_dict().items()):
                    if key.startswith('dynamical_matrix_'):
                        qibz_ar.append(value['q_point'])

                nqpt = len(qibz_ar)

            # Append the required contents of the `save` folder to the remove copy list, copied from the `ph.x`
            # calculation

            prefix = self._PREFIX
            outdir = PhCalculation._OUTPUT_SUBFOLDER
            fildvscf = PhCalculation._DVSCF_PREFIX
            fildyn = PhCalculation._OUTPUT_DYNAMICAL_MATRIX_PREFIX

            ph_path = Path(parent_folder_ph.get_remote_path())

            remote_list.append(
                (parent_folder_ph.computer.uuid, Path(ph_path, outdir, '_ph0', f'{prefix}.phsave').as_posix(), 'save')
            )

            for iqpt in range(1, nqpt + 1):
                remote_list.append((
                    parent_folder_ph.computer.uuid,
                    Path(ph_path, outdir, '_ph0', '' if iqpt == 1 else f'{prefix}.q_{iqpt}',
                         f'{prefix}.{fildvscf}1').as_posix(), Path('save', f'{prefix}.dvscf_q{iqpt}').as_posix()
                ))
                # remote_copy_list.append((
                #     parent_folder_ph.computer.uuid,
                #     Path(
                #     ph_path, outdir, '_ph0', '' if iqpt == 1 else f'{prefix}.q_{iqpt}', f'{prefix}.{fildvscf}_paw1'
                #     ).as_posix(),
                #     Path('save', f"{prefix}.dvscf_paw_q{iqpt}").as_posix()
                # ))
                remote_list.append((
                    parent_folder_ph.computer.uuid, Path(ph_path, f'{fildyn}{iqpt}').as_posix(),
                    Path('save', f'{prefix}.dyn_q{iqpt}').as_posix()
                ))

        if 'parent_folder_epw' in self.inputs:

            parent_folder_epw = self.inputs.parent_folder_epw
            if isinstance(parent_folder_epw, orm.RemoteStashFolderData):
                epw_path = Path(parent_folder_epw.target_basepath)
            else:
                epw_path = Path(parent_folder_epw.get_remote_path())

            vme_fmt_dict = {
                'dipole': 'dmedata.fmt',
                'wannier': 'vmedata.fmt',
            }
            # file_list = [
            #     'selecq.fmt', 'crystal.fmt', 'epwdata.fmt', vme_fmt_dict[parameters['INPUTEPW']['vme']],
            #     f'{self._PREFIX}.kgmap', f'{self._PREFIX}.kmap',
            #     f'{self._PREFIX}.ukk', self._OUTPUT_SUBFOLDER,
            #     self._FOLDER_SAVE
            # ]
            # If ephwrite = .false.and restart = .true., it must be that ephmat folder is saved.
            # We can restart by linking the parent_folder_epw/prefix.ephmat folder to the current folder since this folder won't be modified.
            if (
                (not parameters['INPUTEPW'].get('ephwrite', True))
                and
                parameters['INPUTEPW'].get('restart', False)
                ):
                file_list = ['crystal.fmt', 'selecq.fmt', 'restart.fmt', f'{self._PREFIX}.a2f']
                remote_symlink_list.append(
                    (
                        parent_folder_epw.computer.uuid,
                        Path(epw_path, f'{self._OUTPUT_SUBFOLDER}/{self._PREFIX}.ephmat').as_posix(),
                        Path(f'{self._OUTPUT_SUBFOLDER}/{self._PREFIX}.ephmat').as_posix()
                    )
                )
            # If epwread = .true., it must be that prefix.epmatwp file is saved.
            # From EPW 5.9, vmedata.fmt and dmedata.fmt are always saved and used no matter vme = dipole or wannier.
            # and prefix.mmn, prefix.bvec are also used.
            elif parameters['INPUTEPW'].get('epwread', False):
                file_list = [
                    'crystal.fmt', 'epwdata.fmt', 'vmedata.fmt', 'dmedata.fmt',
                    f'{self._PREFIX}.kgmap', f'{self._PREFIX}.kmap',
                    f'{self._PREFIX}.ukk', f'{self._PREFIX}.mmn', f'{self._PREFIX}.bvec'
                ]
                remote_symlink_list.append(
                    (
                        parent_folder_epw.computer.uuid,
                        Path(epw_path, f'{self._OUTPUT_SUBFOLDER}/{self._PREFIX}.epmatwp').as_posix(),
                        Path(f'{self._OUTPUT_SUBFOLDER}/{self._PREFIX}.epmatwp').as_posix()
                    )
                )

            for filename in file_list:
                remote_list.append(
                    (parent_folder_epw.computer.uuid, Path(epw_path, filename).as_posix(), Path(filename).as_posix())
                )

        parameters['INPUTEPW']['outdir'] = self._OUTPUT_SUBFOLDER
        parameters['INPUTEPW']['dvscf_dir'] = self._FOLDER_SAVE
        parameters['INPUTEPW']['prefix'] = self._PREFIX

        try:
            mesh, offset = self.inputs.qpoints.get_kpoints_mesh()
            test_offset(offset)
            parameters['INPUTEPW']['nq1'] = mesh[0]
            parameters['INPUTEPW']['nq2'] = mesh[1]
            parameters['INPUTEPW']['nq3'] = mesh[2]
        except NotImplementedError as exception:
            raise exceptions.InputValidationError('Cannot get the coarse q-point grid') from exception

        try:
            mesh, offset = self.inputs.kpoints.get_kpoints_mesh()
            test_offset(offset)
            parameters['INPUTEPW']['nk1'] = mesh[0]
            parameters['INPUTEPW']['nk2'] = mesh[1]
            parameters['INPUTEPW']['nk3'] = mesh[2]
        except NotImplementedError as exception:
            raise exceptions.InputValidationError('Cannot get the coarse k-point grid') from exception

        try:
            mesh, offset = self.inputs.qfpoints.get_kpoints_mesh()
            test_offset(offset)
            parameters['INPUTEPW']['nqf1'] = mesh[0]
            parameters['INPUTEPW']['nqf2'] = mesh[1]
            parameters['INPUTEPW']['nqf3'] = mesh[2]
        except AttributeError:
            qfpoints = self.inputs.qfpoints.get_kpoints()
            with folder.open(self._qfpoints_input_file, 'w') as handle:
                handle.write(f'{len(qfpoints)} crystal\n')
                for kpt in qfpoints:
                    handle.write(' '.join([f'{coord:.12}' for coord in kpt]) + '   1.0\n')
            parameters['INPUTEPW']['filqf'] = self._qfpoints_input_file
        except NotImplementedError as exception:
            raise exceptions.InputValidationError('Cannot get the fine q-point grid') from exception

        try:
            mesh, offset = self.inputs.kfpoints.get_kpoints_mesh()
            test_offset(offset)
            parameters['INPUTEPW']['nkf1'] = mesh[0]
            parameters['INPUTEPW']['nkf2'] = mesh[1]
            parameters['INPUTEPW']['nkf3'] = mesh[2]
        except AttributeError:
            kfpoints = self.inputs.kfpoints.get_kpoints()
            with folder.open(self._kfpoints_input_file, 'w') as handle:
                handle.write(f'{len(kfpoints)} crystal\n')
                for kpt in kfpoints:
                    handle.write(' '.join([f'{coord:.12}' for coord in kpt]) + '   1.0\n')
            parameters['INPUTEPW']['filkf'] = self._kfpoints_input_file
        except NotImplementedError as exception:
            raise exceptions.InputValidationError('Cannot get the fine k-point grid') from exception

        if parameters['INPUTEPW'].get('band_plot'):
            retrieve_list += [self._output_elbands_file, self._output_phbands_file]

        if parameters['INPUTEPW'].get('eliashberg', False):
            retrieve_list += [f'{self._PREFIX}.a2f*', f'{self._PREFIX}.phdos*', f"{self._OUTPUT_SUBFOLDER}/{self._PREFIX}.dos"]

        if parameters['INPUTEPW'].get('laniso', False):
            retrieve_list.append(f'{self._PREFIX}.imag_aniso_gap*')
            if parameters['INPUTEPW'].get('lpade', False):
                retrieve_list.append(f'{self._PREFIX}.pade_aniso_gap0_*')

        # customized namelists, otherwise not present in the distributed epw code
        try:
            namelists_toprint = settings.pop('NAMELISTS')
            if not isinstance(namelists_toprint, list):
                raise exceptions.InputValidationError(
                    "The 'NAMELISTS' value, if specified in the settings input "
                    'node, must be a list of strings'
                )
        except KeyError:  # list of namelists not specified in the settings; do automatic detection
            namelists_toprint = self._compulsory_namelists

        with folder.open(self.metadata.options.input_filename, 'w') as infile:
            for namelist_name in namelists_toprint:
                infile.write(f'&{namelist_name}\n')
                # namelist content; set to {} if not present, so that we leave an empty namelist
                namelist = parameters.pop(namelist_name, {})
                for key, value in sorted(namelist.items()):
                    inputs = convert_input_to_namelist_entry(key, value)
                    if key == 'temps':
                        inputs = inputs.replace("'", '')
                    infile.write(inputs)
                infile.write('/\n')

        if parameters:
            raise exceptions.InputValidationError(
                'The following namelists are specified in parameters, but are not valid namelists for the current type '
                f'of calculation: {",".join(list(parameters.keys()))}'
            )

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = (list(settings.pop('CMDLINE', [])) + ['-in', self.metadata.options.input_filename])
        codeinfo.stdout_name = self.metadata.options.output_filename
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list

        calcinfo.retrieve_list = retrieve_list
        calcinfo.retrieve_list += settings.pop('ADDITIONAL_RETRIEVE_LIST', [])

        if settings:
            unknown_keys = ', '.join(list(settings.keys()))
            raise exceptions.InputValidationError(f'`settings` contained unexpected keys: {unknown_keys}')

        return calcinfo
