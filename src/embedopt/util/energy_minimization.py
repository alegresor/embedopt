"""
Energy minimization utility functions for PDB structures.

This module provides functions for preparing PDB files and performing
energy minimization using OpenMM.
"""

import os
import tempfile
import openmm.app as app
import openmm.unit as unit
from openmm import LangevinIntegrator


def prep_pdb(filename, outfile, pH=7.0, keepHOH=False, verbose=True):
    """
    Prepare a PDB file ready for OpenMM simulation.
    
    Parameters
    ----------
    filename: str
        Path to the input PDB file
    outfile: str
        Path of the output PDB file name
    pH: float, default 7.0
        pH value of the system, used to add missing hydrogen
    keepHOH: bool, default False
        When removing the Heterogens in PDB file, keep HOH or not.
    verbose: bool, default True
        Print progress messages
    
    Returns
    -------
    None
    """
    from pdbfixer import PDBFixer
    
    if verbose:
        print(f"Preparing PDB file: {filename}")
    fixer = PDBFixer(filename=filename)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH)
    fixer.removeHeterogens(keepHOH)
    
    app.PDBFile.writeFile(fixer.topology, fixer.positions, open(outfile, "w"))
    if verbose:
        print(f"PDB preparation complete. Saved to: {outfile}")


def preserve_cryst1_line(original_pdb, output_pdb, verbose=True):
    """
    Copy the CRYST1 line from the original PDB to the output PDB.
    This preserves the correct spacegroup information.
    
    Parameters
    ----------
    original_pdb: str
        Path to original PDB file
    output_pdb: str
        Path to output PDB file to update
    verbose: bool, default True
        Print progress messages
    """
    # Read the contents of original file
    with open(original_pdb, "r") as file_a:
        lines_a = file_a.readlines()
    
    # Read the contents of output file
    with open(output_pdb, "r") as file_b:
        lines_b = file_b.readlines()
    
    # Find the index of the CRYST1 line in origin file
    cryst1_index_a = None
    for i, line in enumerate(lines_a):
        if line.startswith("CRYST1"):
            cryst1_index_a = i
            break
    
    # Find the index of the CRYST1 line in output file
    cryst1_index_b = None
    for i, line in enumerate(lines_b):
        if line.startswith("CRYST1"):
            cryst1_index_b = i
            break
    
    # Replace the CRYST1 line in output file with the CRYST1 line from original
    if (cryst1_index_a is not None) and (cryst1_index_b is not None):
        lines_b[cryst1_index_b] = lines_a[cryst1_index_a]
        
        # Write the updated contents
        with open(output_pdb, "w") as file_b:
            file_b.writelines(lines_b)
        if verbose:
            print("CRYST1 line preserved from original PDB")


def verify_topology_match(original_pdb, output_pdb, verbose=True):
    """
    Verify that the output PDB has the same heavy atom topology as the input PDB.
    
    Parameters
    ----------
    original_pdb: str
        Path to original input PDB file
    output_pdb: str
        Path to output PDB file
    verbose: bool, default True
        Print progress messages
        
    Returns
    -------
    bool: True if topologies match, False otherwise
    """
    original = app.PDBFile(original_pdb)
    output = app.PDBFile(output_pdb)
    
    # Get heavy atoms (non-hydrogen) from both structures
    original_heavy = [atom for atom in original.topology.atoms() 
                      if atom.element.symbol != 'H']
    output_heavy = [atom for atom in output.topology.atoms() 
                    if atom.element.symbol != 'H']
    
    # Check if the number of heavy atoms matches
    if len(original_heavy) != len(output_heavy):
        if verbose:
            print(f"WARNING: Heavy atom count mismatch!")
            print(f"  Original: {len(original_heavy)} heavy atoms")
            print(f"  Output: {len(output_heavy)} heavy atoms")
        return False
    
    # Check if atom names and residue information match
    mismatches = []
    for i, (orig_atom, out_atom) in enumerate(zip(original_heavy, output_heavy)):
        if (orig_atom.name != out_atom.name or 
            orig_atom.residue.name != out_atom.residue.name or
            orig_atom.residue.id != out_atom.residue.id):
            mismatches.append(i)
    
    if mismatches:
        if verbose:
            print(f"WARNING: Found {len(mismatches)} atom mismatches in topology!")
            if len(mismatches) <= 5:
                for idx in mismatches:
                    orig = original_heavy[idx]
                    out = output_heavy[idx]
                    print(f"  Atom {idx}: {orig.residue.name}{orig.residue.id}:{orig.name} -> "
                          f"{out.residue.name}{out.residue.id}:{out.name}")
        return False
    
    if verbose:
        print(f"✓ Topology verified: {len(original_heavy)} heavy atoms match between input and output")
    return True


def geo_minimize(pdbmodel_dir, temp, implicit_solvent=True, verbose=True):
    """
    Geometrically minimize model energy.
    
    Args:
        pdbmodel_dir: Path to input PDB file
        temp: Temperature in Kelvin
        implicit_solvent: Whether to use implicit solvent (True) or explicit solvent (False)
        verbose: Print progress messages
    
    Returns:
        energy: Minimized energy in RT units
        topology: OpenMM topology object
        positions: Minimized positions
    """
    INTEGRATOR_ARGS = (
        temp * unit.kelvin,
        1.0 / unit.picoseconds,
        2.0 * unit.femtoseconds,
    )
    
    pdb = app.PDBFile(pdbmodel_dir)
    if implicit_solvent:
        forcefield = app.ForceField(
            "amber99sb.xml", "amber99_obc.xml"
        )  # implicit Solvent
    else:
        forcefield = app.ForceField(
            "amber14/protein.ff14SB.xml", "amber14/tip3p.xml"
        )  # explicit Solvent
    
    system = forcefield.createSystem(
        pdb.topology,
        removeCMMotion=False,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=None,
        rigidWater=True,
    )
    integrator = LangevinIntegrator(*INTEGRATOR_ARGS)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    
    if verbose:
        print("Running energy minimization...")
    simulation.minimizeEnergy()
    
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    
    unit_reciprocal = 1 / (
        temp * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R
    ).value_in_unit(unit.kilojoule_per_mole)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * unit_reciprocal
    
    return energy, pdb.topology, minimized_positions


def minimize_energy_pdb(
    input_pdb: str,
    output_pdb: str,
    temperature: float = 270.0,
    pH: float = 7.0,
    keepHOH: bool = False,
    keep_hydrogens: bool = False,
    explicit_solvent: bool = False,
    verbose: bool = False,
) -> tuple[float, bool]:
    """
    Complete energy minimization pipeline.
    
    This function performs the complete energy minimization workflow:
    1. Prepares PDB (adds hydrogens, fixes missing atoms)
    2. Runs energy minimization using OpenMM
    3. Saves output (removes hydrogens if needed)
    4. Preserves CRYST1 line from original
    5. Verifies topology matches input
    
    Parameters
    ----------
    input_pdb: str
        Path to input PDB file
    output_pdb: str
        Path to output PDB file for relaxed structure
    temperature: float, default 270.0
        Temperature in Kelvin for minimization
    pH: float, default 7.0
        pH value for adding hydrogens
    keepHOH: bool, default False
        Keep water molecules (HOH) for explicit solvent model
    keep_hydrogens: bool, default False
        Keep hydrogen atoms in the output PDB file
    explicit_solvent: bool, default False
        Use explicit solvent instead of implicit solvent
    verbose: bool, default False
        Print detailed progress messages
    
    Returns
    -------
    tuple[float, bool]:
        - energy_in_rt: Minimized energy in RT units
        - topology_matches: True if topology verification passed
    """
    # Step 1: Prepare PDB
    use_temp_file = True
    temp_fd = None
    prepared_pdb_path = None
    
    try:
        # Create a temporary file for the prepared PDB
        temp_fd, prepared_pdb_path = tempfile.mkstemp(suffix=".pdb", prefix="prepared_")
        os.close(temp_fd)  # Close the file descriptor
        
        prep_pdb(
            filename=input_pdb,
            outfile=prepared_pdb_path,
            pH=pH,
            keepHOH=keepHOH,
            verbose=verbose
        )
        
        # Step 2: Energy minimization
        energy, topology, positions = geo_minimize(
            prepared_pdb_path,
            temperature,
            implicit_solvent=not explicit_solvent,
            verbose=verbose
        )
        
        if verbose:
            print(f"Minimized energy: {energy:.4f} RT")
        
        # Step 3: Save output
        if not keep_hydrogens:
            # Remove hydrogen atoms (default behavior)
            modeller = app.Modeller(topology, positions)
            hydrogen_atoms = [atom for atom in modeller.topology.atoms() if atom.element.symbol == 'H']
            modeller.delete(hydrogen_atoms)
            
            if verbose:
                print(f"Removed {len(hydrogen_atoms)} hydrogen atoms")
            
            with open(output_pdb, 'w') as f:
                app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
        else:
            if verbose:
                print("Keeping hydrogen atoms in output")
            with open(output_pdb, 'w') as f:
                app.PDBFile.writeFile(topology, positions, f)
        
        if verbose:
            print(f"Minimized structure saved to: {output_pdb}")
        
        # Preserve CRYST1 line from original PDB
        preserve_cryst1_line(input_pdb, output_pdb, verbose=verbose)
        
        # Step 4: Verify topology
        topology_matches = verify_topology_match(input_pdb, output_pdb, verbose=verbose)
        
        return energy, topology_matches
        
    finally:
        # Clean up temporary file if needed
        if use_temp_file and prepared_pdb_path and os.path.exists(prepared_pdb_path):
            os.remove(prepared_pdb_path)
            if verbose:
                print("Temporary prepared PDB file removed")

