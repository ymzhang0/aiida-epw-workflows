from aiida import orm

def is_compatible(
    kpoints,
    qpoints,
) -> bool:
    """Check if the kpoints and qpoints are compatible."""
    
    kpoints_mesh = kpoints.get_kpoints_mesh()[0]
    kpoints_shift = kpoints.get_kpoints_mesh()[1]
    qpoints_mesh = qpoints.get_kpoints_mesh()[0]
    qpoints_shift = qpoints.get_kpoints_mesh()[1]
    
    compatibility = None
    multiplicities = []
    remainder = []
    
    for k, q in zip(kpoints_mesh, qpoints_mesh):
        multiplicities.append(k // q)
        remainder.append(k % q)


    if kpoints_shift != [0.0, 0.0, 0.0] or qpoints_shift != [0.0, 0.0, 0.0]:
        compatibility = False
    else:
        if remainder == [0, 0, 0]:
            compatibility = True
        else:
            compatibility = False
            
    assert compatibility, "Kpoints and qpoints are not compatible"
    
    return compatibility
