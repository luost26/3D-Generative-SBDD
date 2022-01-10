import os
import py3Dmol
from rdkit import Chem


def visualize_protein_ligand(pdb_block, sdf_block, show_ligand=True, show_surface=True):
    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model':-1},{'cartoon': {'color':'spectrum'}, 'line': {}})

    # Add ligand to the canvas
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model':-1},{'stick':{}})
        if show_surface:
            view.addSurface(py3Dmol.VDW,{'opacity':0.8}, {'model':-1})

    view.zoomTo()
    return view


def visualize_data(data, root, show_ligand=True, show_surface=True):
    protein_path = os.path.join(root, data.protein_filename)
    ligand_path = os.path.join(root, data.ligand_filename)
    with open(protein_path, 'r') as f:
        pdb_block = f.read()
    with open(ligand_path, 'r') as f:
        sdf_block = f.read()
    return visualize_protein_ligand(pdb_block, sdf_block, show_ligand=show_ligand, show_surface=show_surface)


def visualize_generated(data, root, show_ligand=False):
    ptable = Chem.GetPeriodicTable()

    num_atoms = data.ligand_context_element.size(0)
    xyz = "%d\n\n" % (num_atoms, )
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(data.ligand_context_element[i].item())
        x, y, z = data.ligand_context_pos[i].clone().cpu().tolist()
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)

    print(xyz)

    protein_path = os.path.join(root, data.protein_filename)
    ligand_path = os.path.join(root, data.ligand_filename)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()
    with open(ligand_path, 'r') as f:
        sdf_block = f.read()
    
    view = py3Dmol.view()
    # Generated molecule
    view.addModel(xyz, 'xyz')
    view.setStyle({'model':-1},{'sphere':{'radius': 0.3}, 'stick':{}})
    # Focus on the generated
    view.zoomTo()

    # Pocket
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model':-1},{'cartoon': {'color':'spectrum'}, 'line': {}})
    # Ligand
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model':-1},{'stick':{}})

    return view


def visualize_generated_sdf(data, root, show_ligand=False):
    ptable = Chem.GetPeriodicTable()

    num_atoms = data.ligand_context_element.size(0)
    num_bonds = data.ligand_context_bond_type.size(0) if 'ligand_context_bond_type' in data else 0

    mol = 'Generated\nvisualize_generated\n\n'
    mol += '% 3d% 3d  0  0  0  0  0  0  0  0999 V2000\n' % (num_atoms, num_bonds)
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(data.ligand_context_element[i].item())
        x, y, z = data.ligand_context_pos[i].clone().cpu().tolist()
        mol += "% 10.4f% 10.4f% 10.4f %s   0  0  0  0  0  0  0  0  0  0  0  0\n" % (x, y, z, symb)
    
    for j in range(num_bonds):
        mol += '% 3d% 3d% 3d  0  0  0  0\n' % (
            data.ligand_context_bond_index[0, j].item() + 1,
            data.ligand_context_bond_index[1, j].item() + 1,
            data.ligand_context_bond_type[j].item() + 1,
        )
    
    mol += 'M  END\n'
    print(mol)
  

    protein_path = os.path.join(root, data.protein_filename)
    ligand_path = os.path.join(root, data.ligand_filename)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()
    with open(ligand_path, 'r') as f:
        sdf_block = f.read()
    
    view = py3Dmol.view()
    # Generated molecule
    view.addModel(mol, 'sdf')
    view.setStyle({'model':-1},{'sphere':{'radius': 0.3}, 'stick':{}})
    # Focus on the generated
    view.zoomTo()

    # Pocket
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model':-1},{'cartoon': {'color':'spectrum'}, 'line': {}})
    # Ligand
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model':-1},{'stick':{}})

    return view


def visualize_mol(mol, size=(500, 500), surface=False, opacity=0.5):
    """Draw molecule in 3D
    
    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    # assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({'stick':{}, 'sphere':{'radius':0.35}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer


def visualize_generated_mol(data, mol, root, show_ligand=True, show_surface=False, opacity=0.5):
    protein_path = os.path.join(root, data.protein_filename)
    with open(protein_path, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model':-1},{'cartoon': {'color':'spectrum'}, 'line': {}})

    mblock = Chem.MolToMolBlock(mol)
    view.addModel(mblock, 'mol')
    view.setStyle({'model':-1},{'stick':{}, 'sphere':{'radius':0.35}})
    if show_surface:
        view.addSurface(py3Dmol.SAS, {'opacity': opacity}, {'model':-1})

    view.zoomTo()
    return view