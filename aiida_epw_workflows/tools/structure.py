"""Tools for structures."""
import pathlib
import typing as ty

from aiida import orm

def read_structure_from_file(
    filename: ty.Union[str, pathlib.Path],
    store: bool = False
) -> orm.StructureData:
    """Read a xsf/xyz/cif/.. file and return aiida ``StructureData``."""
    from ase.io import read as aseread

    struct = orm.StructureData(ase=aseread(filename))

    if store:
        struct.store()
        print(f"Read and stored structure {struct.get_formula()}<{struct.pk}>")

    return struct

def dilate_structure(
    structure: orm.StructureData,
    factor: float = 1.1
) -> orm.StructureData:
    """Dilate the structure by a factor."""

    ase_structure = structure.get_ase()
    cell = ase_structure.cell * factor
    ase_structure.set_cell(cell, scale_atoms=True)

    return orm.StructureData(ase=ase_structure)