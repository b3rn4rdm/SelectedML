#####################################################
#
# Set of functions used to separate the QM9 and QM7b datasets
# into subsets according to the LUMO energies of
# the molecules.
#
#####################################################



import numpy as np
import tarfile
import pickle
import os
import sys
from datetime import datetime
from uuid import uuid4
from rdkit import Chem
from rdkit.Chem.Lipinski import NumAromaticHeterocycles

# insert here the path to your copy of the xyz2mol repo
sys.path.insert(1, '../../xyz2mol/')

from xyz2mol import xyz2mol

from QM7bfile import *
from QM9file import *


hartree2ev = 27.211386245988


# =========================================================
# Bunch of functions to match certain features in molecules
# =========================================================


def match_fully_saturated(mol):
    """
    matches only a fully saturated molecule
    
    args: 
        mol : rdkit mol object

    returns: 
        bool: True or False
    """
    check = True
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        hyb = atom.GetHybridization()
        if (sym == 'H') & (hyb != Chem.rdchem.HybridizationType.S):
            check = False
        if (sym in ['C', 'N', 'O', 'S']) & (hyb != Chem.rdchem.HybridizationType.SP3):
            check = False
    return check


def match_aromatic(mol):
    """
    matches only a molecule with at least one aromatic bond
    
    args: 
        mol : rdkit mol object

    returns: 
        bool: True or False
    """
    if len(mol.GetAromaticAtoms()) > 0:
        return True
    else:
        return False


def match_aromatic_heterocycle(mol):
    """
    matches molecules with aromatic heterocycles
    """
    return NumAromaticHeterocycles(mol) >= 1


def match_sulfur(mol):
    """
    matches only molecules that have sulfur

    args:
        mol : rdkit mol object

    returns:
        bool : True or False
    """
    check = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'S':
            check = True
    return check


def match_halogen(mol):
    """
    matches only molecules that have a halogen atom

    args: 
        mol : rdkit mol object

    returns:
        bool : True or False
    """
    check = False
    halogens = ['F', 'Cl', 'Br', 'I']
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in halogens:
            check = True
    return check


def match_one_cc3_bond(mol):
    """
    matches molecules that have exactly one CC-triple bond

    args: 
        mol : rdkit mol object

    returns:
        bool : True or False
    """
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            at1 = bond.GetBeginAtom().GetSymbol()
            at2 = bond.GetEndAtom().GetSymbol()
            if (at1 == 'C') & (at2 == 'C'):
                count += 1
    return count == 1


def match_1_unsat_cc3(mol):
    """
    matches molecules that have one CC-triple bond and no other unsturated bond

    args: 
        mol : rdkit mol object
    
    return:
        bool : True or False
    """
    return match_one_cc3_bond(mol) & (count_unsaturated_bonds(mol) == 1)


def match_bond_type(mol, bond_type='double'):
    """
    matches only a molecule with at least one given bond type
    
    args: 
        mol : rdkit mol object
        bond_type(='double') : type of bond ('double', 'triple' or 'aromatic')

    returns: 
        bool: True or False
    """
    check = False
    for bond in mol.GetBonds():
        if bond_type == 'double':
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                check = True
        elif bond_type == 'triple':
            if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                check = True
        elif bond_type == 'aromatic':
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                check = True
    return check


def match_amino_acid(mol):
    """
    match an amino acid

    args: 
        mol : rdkit mol object

    returns:
        bool : True or False
    """
    has_carboxy = mol.HasSubstructMatch(Chem.MolFromSmarts('[#8-][#6](=[#8])'))
    has_amino = mol.HasSubstructMatch(Chem.MolFromSmarts('[#7+]'))
    return has_carboxy & has_amino


def match_n_bonds(mol, n_double, n_triple, n_arom):
    """
    match a molecule with specified numbers of double, triple and aromatic bonds

    args: 
        mol : rdkit mol object
        n_double : number of double bonds
        n_triple : number of triple bonds
        n_arom : number of aromatic bonds

    returns:
        bool : True or False
    """
    count_double = 0
    count_triple = 0
    count_arom = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            count_double += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            count_triple += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            count_arom += 1
    return (count_double == n_double) & (count_triple == n_triple) & (count_arom == n_arom)


def count_unsaturated_bonds(mol):
    """
    counts the number of unsaturated (double, triple or aromatic) bonds in a molecule

    args:
        mol : rdkit mol object
    
    returns:
        count : number of unsaturated bonds
    """
    count = 0 
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            count += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            count += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            count += 1
    return count


def match_qm9_molecules(qm9_data, match_func, neg=False):
    """
    filters out molecules that agree with the matching function

    args: 
        qm9_data: list of qm9 molecule data as dicts
        match_func: matching function
        neg(=False) : if True, return also those molecules that don't match

    returns: 
        match_pos : list of qm9 molecules that match
        match_neg (if neg=True): list of qm9 molecules that don't match
    """
    smiles_key = 'SMILES_GDB9'
    match_pos = []
    match_neg = []
    for mol in qm9_data:
        m = Chem.MolFromSmiles(mol[smiles_key])
        if match_func(m):
            match_pos.append(mol)
        else:
            match_neg.append(mol)
    if neg:
        return (match_pos, match_neg)
    else:
        return match_pos


def match_qm7b_molecules(qm7b_data, match_func, neg=False):
    """
    filters out molecules that agree with the matching function for qm7b molecules

    args: 
        qm7b_data: list of qm7b molecule data as dicts
        match_func: matching function
        neg(=False) : if True, return also those molecules that don't match

    returns: 
        match_pos : list of qm7b molecules that match
        match_neg (if neg=True): list of qm7b molecules that don't match
    
    """
    element2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'Cl': 17}
    pos_matches = []
    neg_matches = []
    for mol in qm7b_data:
        atoms = [element2charge[el] for el in mol['elements']]
        coords = mol['coords']
        rdmol = xyz2mol(atoms, coords)
        if match_func(rdmol):
            pos_matches.append(mol)
        else:
            neg_matches.append(mol)
    if neg:
        return (pos_matches, neg_matches)
    else:
        return pos_matches


def count_single_bonds(mol, at1, at2):
    """
    count the number of single bonds between two specified atoms
    
    args: 
        mol : rdkit mol object
        at1 : symbol of atom 1
        at2 : symbol of atom 2

    return: 
       count : number of corresponding bonds as int
    """
    count = 0
    for bond in mol.GetBonds():
        if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE):
            begin_at = bond.GetBeginAtom().GetSymbol()
            end_at = bond.GetEndAtom().GetSymbol()
            if ((begin_at == at1) & (end_at == at2)) | ((begin_at == at2) & (end_at == at1)):
                count += 1
    return count


#=======================================
# Bond counting functions
#=======================================


def count_double_bonds(mol, at1, at2):
    """
    count the number of double bonds between two specified atoms
    
    args: 
        mol : rdkit mol object
        at1 : symbol of atom 1
        at2 : symbol of atom 2

    return: 
       count : number of corresponding bonds as int
    """
    count = 0
    for bond in mol.GetBonds():
        if (bond.GetBondType() == Chem.rdchem.BondType.DOUBLE):
            begin_at = bond.GetBeginAtom().GetSymbol()
            end_at = bond.GetEndAtom().GetSymbol()
            if ((begin_at == at1) & (end_at == at2)) | ((begin_at == at2) & (end_at == at1)):
                count += 1
    return count


def count_triple_bonds(mol, at1, at2):
    """
    count the number of triple bonds between two specified atoms
    
    args: 
        mol : rdkit mol object
        at1 : symbol of atom 1
        at2 : symbol of atom 2

    return: 
       count : number of corresponding bonds as int
    """
    count = 0
    for bond in mol.GetBonds():
        if (bond.GetBondType() == Chem.rdchem.BondType.TRIPLE):
            begin_at = bond.GetBeginAtom().GetSymbol()
            end_at = bond.GetEndAtom().GetSymbol()
            if ((begin_at == at1) & (end_at == at2)) | ((begin_at == at2) & (end_at == at1)):
                count += 1
    return count


def count_aromatic_bonds(mol, at1, at2):
    """
    count the number of aromatic bonds between two specified atoms
    
    args: 
        mol : rdkit mol object
        at1 : symbol of atom 1
        at2 : symbol of atom 2

    return: 
       count : number of corresponding bonds as int
    """
    count = 0
    for bond in mol.GetBonds():
        if (bond.GetBondType() == Chem.rdchem.BondType.AROMATIC):
            begin_at = bond.GetBeginAtom().GetSymbol()
            end_at = bond.GetEndAtom().GetSymbol()
            if ((begin_at == at1) & (end_at == at2)) | ((begin_at == at2) & (end_at == at1)):
                count += 1
    return count


#========================================
# Other useful functions
#========================================


def get_subset_from_qm9_ids(qm9_data, ids):
    """
    gets the molecules that correspond to the given ids
    
    args: 
        qm9_data : list of all qm9 molecules as dicts
        ids : list of qm9 ids
    
    returns: 
        subset : list of qm9 molecules as dicts
    """
    subset = []
    for mol in qm9_data:
        if mol['properties']['index'] in ids:
            subset.append(mol)
    return subset


def get_qm9_property(qm9_data, prop):
    """
    get property for a list of qm9 molecules
    
    args: 
        qm9_data : list of qm9 molecules
        prop : property of interset as astring
    
    returns: 
        array with property of interest of all moelcules in qm9_data
    """
    lumos = []
    for mol in qm9_data:
        lumos.append(mol['properties'][prop])
    return np.asarray(lumos)


def get_qm9_ids(qm9_data):
    """
    get list of qm9 ids from qm9 molecules
    
    args: 
        qm9_data : list of qm9 molecules

    returns: 
        list of all molecules in qm9_data
    """
    ids = []
    for mol in qm9_data:
        ids.append(int(mol['properties']['index']))
    return np.asarray(ids)


def match_carbon_only(x):
    check = True
    for atom in x.GetAtoms():
        if atom.GetSymbol() not in ['C', 'H']:
            check = False
    return check


def match_nitrogen(x):
    check = False
    for atom in x.GetAtoms():
        if atom.GetSymbol() == 'N':
            check = True
    return check

def match_oxygen(x):
    check = False
    for atom in x.GetAtoms():
        if atom.GetSymbol() == 'O':
            check = True
    return check


def get_element_dict(elements):
    element_dict = {}
    for el in elements:
        if el not in element_dict:
            element_dict[el] = 1
        else:
            element_dict[el] += 1
    for el in ['C', 'N', 'O', 'F', 'H']:
        if el not in element_dict:
            element_dict[el] = 0
    return element_dict


def get_degree_of_unsaturation(qm9_mol):
    el_dict = get_element_dict(qm9_mol['elements'])
    dou = (2 * el_dict['C'] + 2 + el_dict['N'] - el_dict['H'] - el_dict['F']) / 2
    return int(dou)


match_carbonyl = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#6]=[#8]'))
match_carbonyl_to_O = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#8][#6]=[#8]'))
match_carbonyl_to_N = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#7][#6]=[#8]'))
match_carbonyl_to_ON = lambda x : match_carbonyl_to_O(x) | match_carbonyl_to_N(x)

match_1_unsat = lambda x : count_unsaturated_bonds(x) == 1
match_2_unsat = lambda x: count_unsaturated_bonds(x) == 2
match_3_unsat = lambda x: count_unsaturated_bonds(x) == 3
match_gt1_unsat = lambda x: count_unsaturated_bonds(x) > 1
match_gt2_unsat = lambda x: count_unsaturated_bonds(x) > 2

match_double_bond = lambda x: match_bond_type(x, 'double')
match_triple_bond = lambda x: match_bond_type(x, 'triple')

match_imine = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#6]=[#7]'))
match_nitrile = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#6]#[#7]'))


def iterate_over_tests(mol, mol_dict):
    """
    args:
        mol

    returns:
        list of tags
    """
    tags = []
    
    if match_fully_saturated(mol):
        tags.append('fullysat')

    if match_1_unsat(mol):
        tags.append('1_unsat')

    if match_2_unsat(mol):
        tags.append('2_unsat')

    if match_3_unsat(mol):
        tags.append('3_unsat')

    if match_gt1_unsat(mol):
        tags.append('gt1_unsat')

    if match_gt2_unsat(mol):
        tags.append('gt2_unsat')

    if match_carbonyl(mol):
        tags.append('C=O')

    if match_carbonyl_to_ON(mol):
        tags.append('(O,N)C=O')

    if match_carbonyl_to_O(mol):
        tags.append('OC=O')

    if match_carbonyl_to_N(mol):
        tags.append('NC=O')

    if match_amino_acid(mol):
        tags.append('aminoacid')

    if match_aromatic(mol):
        tags.append('aromatic')

    if match_aromatic_heterocycle(mol):
        tags.append('hetero_arom')

    if match_double_bond(mol):
        tags.append('double')

    if match_triple_bond(mol):
        tags.append('triple')

    if match_carbon_only(mol):
        tags.append('carbon')

    if match_oxygen(mol):
        tags.append('oxygen')

    if match_nitrogen(mol):
        tags.append('nitrogen')

    if match_imine(mol):
        tags.append('imine')

    if match_halogen(mol):
        tags.append('halogen')

    if get_degree_of_unsaturation(mol_dict) == 0:
        tags.append('dou_0')

    if get_degree_of_unsaturation(mol_dict) == 1:
        tags.append('dou_1')

    if get_degree_of_unsaturation(mol_dict) == 2:
        tags.append('dou_2')

    if get_degree_of_unsaturation(mol_dict) == 3:
        tags.append('dou_3')

    if get_degree_of_unsaturation(mol_dict) > 2:
        tags.append('dou_gt2')

    if match_sulfur(mol):
        tags.append('sulfur')

    if match_nitrile(mol):
        tags.append('nitrile')

    if match_one_cc3_bond(mol):
        tags.append('one_cc3')

    if match_1_unsat_cc3(mol):
        tags.append('1_unsat_cc3')
    
    return tags


def main(datadir, dataname, outputdir, outputname, dataset='qm9', xyz='smiles'):
    """
    args:
        datadir : directory of dataset
        dataname : name of dataset file
        outputdir : directory of the resulting frequency analysis file
        outputname : name of frequency analysis file
        dataset : 'qm9' or 'qm7b'
        xyz : 'smiles' (generate rdkit molecule object from smiles, for qm9)
              or 'xyz' (generate rdkit molecule object from xyz coords with xyz2mol, for qm7b)
    """

    if dataset == 'qm9':
        data = load_qm9_data(datadir, dataname)
    elif dataset == 'qm7b':
        data = read_qm7b_file(f'{datadir}{dataname}')

    element2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'Cl': 17}

    # do the frequency analysis
    tags_dict = {}
    for i, mol_dict in enumerate(data):
        
        if dataset == 'qm9':
            index = mol_dict['properties']['index']
        elif dataset == 'qm7b':
            index = i + 1
            
        if xyz == 'smiles':
            rdmol = Chem.MolFromSmiles(mol_dict['SMILES_GDB9'])
            tags = iterate_over_tests(rdmol, mol_dict)
        elif xyz == 'xyz':
            atoms = [element2charge[el] for el in mol_dict['elements']]
            coords = mol_dict['coords']
            rdmol = xyz2mol(atoms, coords)[0]
            tags = iterate_over_tests(rdmol, mol_dict)
            
        tags_dict[index] = {'tags': tags, 'properties': mol_dict['properties']}

    # write output file

    with open(f'{outputdir}{outputname}', 'wb') as outf:
        pickle.dump(tags_dict, outf)
        
    return

    
if __name__ == '__main__':

    main( * sys.argv[1:])
