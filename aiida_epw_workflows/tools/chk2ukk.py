# Re-defining required classes and imports after reset

import struct
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


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


Int32 = "int32"
Float64 = "float64"


@dataclass
class Chk:
    header: str
    n_bands: int
    n_exclude_bands: int
    exclude_bands: list
    lattice: np.ndarray
    recip_lattice: np.ndarray
    n_kpts: int
    kgrid: tuple
    kpoints: np.ndarray
    n_bvecs: int
    n_wann: int
    checkpoint: str
    have_disentangled: bool
    omega_invariant: Optional[float]
    dis_bands: Optional[list]
    n_dis: Optional[list]
    Udis: Optional[list]
    Uml: list
    M: list
    r: list
    spreads: np.ndarray

def get_U(chk):
    if not chk.have_disentangled:
        # Return deepcopy for safety, so that chk.Uml is not modified
        return chk.Uml.copy()

    Udis = get_Udis(chk)  # 这是一个 list of np.ndarray，长度为 n_kpts
    Uml = chk.Uml # list of np.ndarray

    U = [d @ m for d, m in zip(Udis, Uml)]
    return U

def get_Udis(chk):

    n_kpts = chk.n_kpts
    n_wann = chk.n_wann

    # 类型 T = np.complex128（由 Uml[0] 推断）
    dtype = chk.Uml[0].dtype

    if not chk.have_disentangled:
        # 每个 k 点下单位矩阵（形状 n_wann x n_wann）
        return [np.eye(n_wann, dtype=dtype) for _ in range(n_kpts)]

    Udis_sorted = []
    for ik in range(n_kpts):
        dis_flags = chk.dis_bands[ik]
        p = np.argsort(dis_flags)[::-1]
        Udis_sorted.append(chk.Udis[ik][p, :])  # reorder Bloch bands

    return Udis_sorted

def read_chk_bin(filename):
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

    return Chk(
        header=header,
        n_bands=n_bands,
        n_exclude_bands=n_exclude_bands,
        exclude_bands=exclude_bands,
        lattice=lattice,
        recip_lattice=recip_lattice,
        n_kpts=n_kpts,
        kgrid=kgrid,
        kpoints=kpoints,
        n_bvecs=n_bvec,
        n_wann=n_wann,
        checkpoint=checkpoint,
        have_disentangled=have_disentangled,
        omega_invariant=omega_invariant,
        dis_bands=dis_bands,
        n_dis=n_dis,
        Udis=[Udis[:, :, ik] for ik in range(n_kpts)],
        Uml=[Uml[:, :, ik] for ik in range(n_kpts)],
        M=[[M[:, :, ib, ik] for ib in range(n_bvec)] for ik in range(n_kpts)],
        r=[r[:, iw] for iw in range(n_wann)],
        spreads=spreads
    )

# chk = read_chk_bin("aiida.chk")

@dataclass
class Ukk:

    ibndstart: int  # index of the first band
    ibndend: int    # index of the last band
    n_kpts: int     # number of k-points
    n_bands: int    # number of bands
    n_wann: int     # number of Wannier functions

    # List of shape-(n_bands, n_wann) complex matrices
    U: List[np.ndarray]

    # List of shape-(n_bands,) boolean arrays
    frozen_bands: List[np.ndarray]

    # Single boolean array of shape-(n_bands + n_excl_bands,)
    excluded_bands: np.ndarray

    # List of Vec3 objects (length = n_wann)
    centers: List[np.ndarray]

def Chk2Ukk(chk, alat):
    n_bands = chk.n_bands
    exclude_bands = chk.exclude_bands
    n_kpts = chk.n_kpts
    n_wann = chk.n_wann

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
    r_scaled = [v/alat for v in chk.r]

    U = get_U(chk)  # Returns List[np.ndarray] of shape (n_bands, n_wann) per k-point

    return Ukk(
        ibndstart=ibndstart,
        ibndend=ibndend,
        n_kpts=n_kpts,
        n_bands=n_bands,
        n_wann=n_wann,
        U=U,
        frozen_bands=frozen_bands,
        excluded_bands=excluded_bands_bool,
        centers=r_scaled
    )

# ukk = Chk2Ukk(chk, 1.0)

def write_epw_ukk(ukk: Ukk, filukk: str = "aiida.ukk"):
    with open(filukk, "w") as f:
        # Write ibndstart and ibndend
        f.write(f"{ukk.ibndstart} {ukk.ibndend}\n")

        # Write unitary matrices U
        for ik in range(ukk.n_kpts):
            for ib in range(ukk.n_bands):
                for iw in range(ukk.n_wann):
                    u = ukk.U[ik][ib, iw]
                    f.write("({:25.18E},{:25.18E})\n".format(u.real, u.imag))

        # Write lwindow flags (frozen bands)
        for ik in range(ukk.n_kpts):
            for ib in range(ukk.n_bands):
                flag = "T" if ukk.frozen_bands[ik][ib] else "F"
                f.write(f"{flag}\n")

        # Write excluded bands
        for ex in ukk.excluded_bands:
            flag = "T" if ex else "F"
            f.write(f"{flag}\n")

        # Write Wannier centers (in alat units, dimensionless)
        for vec in ukk.centers:
            f.write("{:22.12E}  {:22.12E}  {:22.12E}\n".format(vec[0], vec[1], vec[2]))


# write_epw_ukk(ukk, "aiida.ukk")