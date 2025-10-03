from dataclasses import dataclass
from typing import Optional
import struct

import numpy as np

from aiida.orm import SinglefileData


Int32 = "int32"
Float64 = "float64"

class FString:
    def __init__(self, length):
        self.length = length

class FortranFile:
    def __init__(self, filename):
        self.f = open(filename, "rb")

    def read_record(self):
        n1 = struct.unpack("i", self.f.read(4))[0]
        data = self.f.read(n1)
        n2 = struct.unpack("i", self.f.read(4))[0]
        assert n1 == n2, "Record length mismatch"
        return data

    def read(self, dtype, count=1):
        raw = self.read_record()
        if isinstance(dtype, FString):
            return raw[:dtype.length].decode("utf-8").rstrip()
        elif dtype == "int32":
            return struct.unpack(f"{count}i", raw) if count > 1 else struct.unpack("i", raw)[0]
        elif dtype == "float64":
            return struct.unpack(f"{count}d", raw) if count > 1 else struct.unpack("d", raw)[0]
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def close(self):
        self.f.close()

class ChkData(SinglefileData):
    """Class to handle Wannier90 chk file."""

    def set_file(self, file, filename=None, **kwargs):
        """Add a file to the node, parse it and set the attributes found.

        :param file: absolute path to the file or a filelike object
        :param filename: specify filename to use (defaults to name of provided file).
        """
        # pylint: disable=redefined-builtin
        super().set_file(file, filename, **kwargs)

        # Parse the force constants file
        parsed_data = parse_chk_file(self.get_content().splitlines())

        # Add all other attributes found in the parsed dictionary
        for key, value in parsed_data.items():
            self.base.attributes.set(key, value)

    @property
    def header(self):
        """Return the number of atoms.

        :return: a scalar
        """
        return self.base.attributes.get('header')

    @property
    def n_bands(self):
        """Return the number of bands.

        :return: a scalar
        """
        return self.base.attributes.get('n_bands')

    @property
    def lattice(self):
        """Return the crystal unit cell where rows are the crystal vectors.

        :return: a 3x3 numpy.array
        """
        return self.base.attributes.get('lattice')

    @property
    def recip_lattice(self):
        """Return the reciprocal lattice unit cell where rows are the reciprocal vectors.

        :return: a 3x3 numpy.array
        """
        return self.base.attributes.get('recip_lattice')

    @property
    def n_kpts(self):
        """Return the number of k-points.

        :return: a scalar
        """
        return self.base.attributes.get('n_kpts')

    @property
    def kgrid(self):
        """Return the k-point grid.

        :return: a tuple of 3 integers
        """
        return self.base.attributes.get('kgrid')

    @property
    def kpoints(self):
        """Return the k-points.

        :return: a 3xN numpy.array
        """
        return self.base.attributes.get('kpoints')

    @property
    def n_bvecs(self):
        """Return the number of reciprocal vectors.

        :return: a scalar
        """
        return self.base.attributes.get('n_bvecs')

    @property
    def n_wann(self):
        """Return the number of Wannier functions.

        :return: a scalar
        """
        return self.base.attributes.get('n_wann')

    @property
    def checkpoint(self):
        """Return the checkpoint name.

        :return: a string
        """
        return self.base.attributes.get('checkpoint')

    @property
    def have_disentangled(self):
        """Return whether the calculation has disentangled bands.

        :return: a boolean
        """
        return self.base.attributes.get('have_disentangled')

    @property
    def omega_invariant(self):
        """Return the omega invariant.

        :return: a scalar
        """
        return self.base.attributes.get('omega_invariant')

    @property
    def n_dis(self):
        """Return the number of disentangled bands.

        :return: a list of integers
        """
        return self.base.attributes.get('n_dis')

    def get_Udis(self):

        n_kpts = self.n_kpts
        n_wann = self.n_wann

        # 类型 T = np.complex128（由 Uml[0] 推断）
        dtype = chk.Uml[0].dtype

        if not self.have_disentangled:
            # 每个 k 点下单位矩阵（形状 n_wann x n_wann）
            return [np.eye(n_wann, dtype=dtype) for _ in range(n_kpts)]

        Udis_sorted = []
        for ik in range(n_kpts):
            dis_flags = self.dis_bands[ik]
            p = np.argsort(dis_flags)[::-1]
            Udis_sorted.append(self.Udis[ik][p, :])  # reorder Bloch bands

        return Udis_sorted

    def get_U(chk):
        if not chk.have_disentangled:
            # Return deepcopy for safety, so that chk.Uml is not modified
            return self.Uml.copy()

        Udis = self.get_Udis()  # 这是一个 list of np.ndarray，长度为 n_kpts
        Uml = self.Uml # list of np.ndarray

        U = [d @ m for d, m in zip(Udis, Uml)]
        return U

    def Chk2Ukk(self, alat):
        n_bands = self.n_bands
        exclude_bands = self.exclude_bands
        n_kpts = self.n_kpts
        n_wann = self.n_wann

        # Create frozen bands: all True for each k-point
        frozen_bands = [np.ones(n_bands, dtype=bool) for _ in range(n_kpts)]
        n_excl_bands = len(exclude_bands)
        n_bands_tot = n_bands + n_excl_bands

        included = np.ones(n_bands_tot, dtype=bool)
        included[np.array(exclude_bands)-1] = False
        excluded_bands_bool = ~included
        if n_excl_bands > 0:
            ibndstart = np.argmax(included) + 1  # first True (1-based)
            ibndend = n_bands_tot - np.argmax(included[::-1])  # last True (1-based)
        else:
            ibndstart = 1
            ibndend = n_bands_tot
        # Convert centers from Angstrom to dimensionless (alat unit)
        r_scaled = [v/alat for v in self.r]

        U = self.get_U()  # Returns List[np.ndarray] of shape (n_bands, n_wann) per k-point

        return {
            "ibndstart": ibndstart,
            "ibndend": ibndend,
            "n_kpts": n_kpts,
            "n_bands": n_bands,
            "n_wann": n_wann,
            "U": U,
            "frozen_bands": frozen_bands,
            "excluded_bands": excluded_bands_bool,
            "centers": r_scaled
        }


def parse_chk_file(filename):
    io = FortranFile(filename)

    header = io.read(FString(33))
    n_bands = io.read(Int32)
    n_exclude_bands = io.read(Int32)
    exclude_bands = list(io.read(Int32, n_exclude_bands)) if n_exclude_bands > 0 else (io.read_record() or [])

    lattice = np.array(io.read(Float64, 9)).reshape((3, 3), order="F").T
    recip_lattice = np.array(io.read(Float64, 9)).reshape((3, 3), order="F").T

    n_kpts = io.read(Int32)
    kgrid = tuple(io.read(Int32, 3))
    kpoints = np.array(io.read(Float64, 3 * n_kpts)).reshape((3, n_kpts), order="F").T

    n_bvec = io.read(Int32)
    n_wann = io.read(Int32)

    checkpoint = io.read(FString(20))
    have_disentangled = bool(io.read(Int32))

    if have_disentangled:
        omega_invariant = io.read(Float64)
        record = io.read_record()  # 一次 Fortran 记录
        # returns a full record in bytes
        tmp = np.frombuffer(record, dtype=np.int32).reshape((n_bands, n_kpts), order="F")
        print(tmp)
        dis_bands = [tmp[:, ik] != 0 for ik in range(n_kpts)]
        for ik in range(n_kpts):
            print(f"k-point {ik}: {np.count_nonzero(dis_bands[ik])} disentangled bands")
        n_dis = list(io.read(Int32, n_kpts))
        for ik in range(n_kpts):
            assert n_dis[ik] == np.count_nonzero(dis_bands[ik]), f"Mismatch at k-point {ik}"
        Udis = np.frombuffer(io.read_record(), dtype=np.complex128).reshape((n_bands, n_wann, n_kpts), order="F")
    else:
        omega_invariant = -1.0
        dis_bands = []
        Udis = []

    Uml = np.frombuffer(io.read_record(), dtype=np.complex128).reshape((n_wann, n_wann, n_kpts), order="F")
    M = np.frombuffer(io.read_record(), dtype=np.complex128).reshape((n_wann, n_wann, n_bvec, n_kpts), order="F")

    r = np.array(io.read(Float64, 3 * n_wann)).reshape((3, n_wann), order="F")

    spreads = np.array(io.read(Float64, n_wann))

    io.close()

    return {
        "header": header,
        "n_bands": n_bands,
        "n_exclude_bands": n_exclude_bands,
        "exclude_bands": exclude_bands,
        "lattice": lattice,
        "recip_lattice": recip_lattice,
        "n_kpts": n_kpts,
        "kgrid": kgrid,
        "kpoints": kpoints,
        "n_bvecs": n_bvec,
        "n_wann": n_wann,
        "checkpoint": checkpoint,
        "have_disentangled": have_disentangled,
        "omega_invariant": omega_invariant,
        "dis_bands": dis_bands,
        "n_dis": n_dis,
        "Udis": [Udis[:, :, ik] for ik in range(n_kpts)],
        "Uml": [Uml[:, :, ik] for ik in range(n_kpts)],
        "M": [[M[:, :, ib, ik] for ib in range(n_bvec)] for ik in range(n_kpts)],
        "r": [r[:, iw] for iw in range(n_wann)],
        "spreads": spreads
    }

