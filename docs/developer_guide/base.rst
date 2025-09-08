.. _base:

=======================
EpwBaseWorkChain
=======================

This is the base workchain for a EpwCalculation.

Similar to PwBaseWorkChain, PhBaseWorkChain, it is used to manage the inputs of the EpwCalculation in a unified way.

It will,

- Provide a builder generated from a protocol for the submission of the EpwCalculation process.
- Check the parent folder of finished wannier90 calculation and a phonon calculation; check the compatibility of the coarse k/q grids.
- Automatically generate the fine k/q grids for epw interpolation.
- Automatically restart the calculation upon failures.
- TODO: Handle the EPW errors automatically.

This work chain accepts the parent folders of the Wannier90 and Phonon work chains.

Firstly it will validate the parent folders.

- If the phonon parent folders are valid, it will use the qpoints from the phonon parent folders.

Then it will check if the wannier90 parent folders are valid.

- If so, it will check if the kpoints of the wannier90 work chain are compatible with the qpoints of the phonon work chain.

- If they are not compatible, it will re-generate the kpoints of the wannier90 work chain based on the qpoints of the phonon work chain. Then it will re-run the wannier90 work chain.

- If the wannier90 parent folders are not valid, it will run the wannier90 work chain from scratch.

If none of the parent folders are provided or are not valid, it will run the Wannier90 and Phonon work chains from scratch.

