##################################################33
#
# Script to generate input data for qml
#
###############################################


import numpy as np
import sys
import os
import tarfile
import pickle

from qml.representations import generate_coulomb_matrix, generate_bob, generate_slatm, get_slatm_mbtypes, generate_acsf, generate_fchl_acsf
from datetime import datetime
from uuid import uuid4
from rdkit import Chem
from frequency_analysis import *
# from pydash.arrays import chunk

from QM9file import *


# some globally useful data
ELEMENT2NUC = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17}


def usage():
    message = """
    Generate represenations data and labels for QML.

    Usage:
    python generate_qml_data.py datadir dataname outputdir outputname rep=rep text=text **kwargs
    
    args: 
        datadir : directory containing the original data with '/' at the end
        dataname : name of the data file
        outputdir : directory for the resulting representation data
        outputname : name of the representations file
        rep : name of the representation (cm, bob, slatm, acsf)
        text (optional) : commentary text to help explain the data further
        kwargs : other parameters of the represntation that need to be specified ('cm', 'bob', 'slatm', 'acsf', 'fchl_acsf')
    """
    print(message)
    return sys.exit(0)


# labelling function
# ==================


match_carbonyl = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#6]=[#8]'))
match_carbonyl_to_O = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#8][#6]=[#8]'))
match_carbonyl_to_N = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#7][#6]=[#8]'))
match_carbonyl_to_ON = lambda x : match_carbonyl_to_O(x) | match_carbonyl_to_N(x)
match_1_unsat = lambda x : count_unsaturated_bonds(x) == 1


def get_labels_qm9_gap(data):
    """
    Generates the labels for classification into 3 classes according to the gaps.

    args:
        data : dict with indices as keys and info as values

    returns : 
        labels : numpy array with labels (0, 1, 2)
    """
    
    labels = []
    
    for index, values in data.items():
        
        tags = values['tags']
        if 'fullysat' in tags:
            labels.append(2)
        elif 'C=O' in tags:
            if 'aminoacid' in tags:
                labels.append(1)
            elif 'aromatic' in tags:
                labels.append(0)
            elif ('1_unsat' in tags) & ('(O,N)C=O' in tags):
                labels.append(1)
            else:
                labels.append(0)
        elif 'aromatic' in tags:
            labels.append(0)
        else:
            labels.append(1)
            
    return np.asarray(labels)       


# get the energies
# ================


def get_homo_lumo_energies(data):
    """
    get HOMO nad LUMO energiess from qm9 data
    
    args:
        data : list of qm9 molecule dicts
   
    returns: 
        homo : HOMO energies
        lumo : LUMO energies
        gaps : HOMO-LUMO gaps
    """
    hartree2ev = 27.211386245988
    
    homos = np.zeros(len(data))
    lumos = np.zeros(len(data))
    gaps = np.zeros(len(data))
    for i, mol in enumerate(data):
        homos[i] = mol['properties']['homo'] * hartree2ev
        lumos[i] = mol['properties']['lumo'] * hartree2ev
        gaps[i] = mol['properties']['gap'] * hartree2ev
    return (homos, lumos, gaps)


# coulomb matrix
# ==============


def get_max_size(qm9_data):
    """
    get size of largest molecule in the dataset
    
    args: 
        qm9_data : list of qm9 molecule dicts

    returns: 
        max_size : size of largest molecule
    """
    max_size = 5
    for mol in qm9_data:
        max_size = max(max_size, len(mol['elements']))
    return max_size


def get_coulomb_matrices_for_data(data):
    """
    generates the coulomb matrix for a dataset
    
    args: 
        data : list of qm9 molecul dicts

    returns: 
        X : array with CM representations fo r all the molecules in the dataset
    """
    max_size = get_max_size(data)
    Z = np.zeros((len(data), max_size))
    C = np.zeros((len(data), max_size, 3))
    X = np.zeros((len(data), int(max_size * (max_size + 1) / 2)))
    for i, mol in enumerate(data):
        nuclear_charges = np.asarray([ELEMENT2NUC[el] for el in mol['elements']])
        nuclear_coords = mol['coords']
        X[i] = generate_coulomb_matrix(nuclear_charges, nuclear_coords, size=max_size)
        Z[i, :nuclear_charges.shape[0]] = nuclear_charges
        C[i, :nuclear_coords.shape[0], :] = nuclear_coords
    return (X, max_size, Z, C)


# Bag of bonds
# ============


def get_asize(qm9_data):
    """
    get max number of occurences of any atom in all the molecules of a dataset
    
    args: 
        qm9_data : list of qm9 molecule dicts
    
    returns: 
        asize: dict with atoms as keys, max occurences as values
    """
    asize = {}
    for mol in qm9_data:
        unique_elements, counts = np.unique(mol['elements'], return_counts=True)
        for el, count in zip(unique_elements, counts):
            if el not in asize:
                asize[el] = count
            else:
                asize[el] = max(asize[el], count)
    return asize


def get_bob_for_data(data):
    """
    generate BoB representation for a dataset
    
    args: 
        qm9_data : list of qm9 molecule dicts

    returns: 
        X : array with BoB representatoion for the entire dataset
    """
    max_size = get_max_size(data)
    asize = get_asize(data)
    atomtypes = sorted(asize.keys())
    nuclear_charges_test = np.asarray([ELEMENT2NUC[el] for el in data[0]['elements']])
    nuclear_coords_test = np.asarray(data[0]['coords'])
    X_test = generate_bob(nuclear_charges_test, nuclear_coords_test, atomtypes, max_size,
                          asize)
    rep_size = X_test.shape[0]
    Z = np.zeros((len(data), max_size))
    C = np.zeros((len(data), max_size, 3))
    X = np.zeros((len(data), rep_size))
    for i, mol in enumerate(data):
        nuclear_charges = np.asarray([ELEMENT2NUC[el] for el in mol['elements']])
        nuclear_coords = mol['coords']
        X[i] = generate_bob(nuclear_charges, nuclear_coords, atomtypes, max_size, asize)
        Z[i, :nuclear_charges.shape[0]] = nuclear_charges
        C[i, :nuclear_coords.shape[0], :] = nuclear_coords
    return (X, max_size, atomtypes, asize, Z, C)


# Slatm
# =====


def get_nuclear_charges_list(data):
    """
    get list of nuclear charges for a dataset  of molecules
    
    args: 
        data : list of qm9 molecule dicts
    
    return: 
        nuclear_charges_list : list with arrays af nuclear charges for each molecule
    """
    nuclear_charges_list = []
    for mol in data:
        nuclear_charges_list.append([ELEMENT2NUC[el] for el in mol['elements']])
    return nuclear_charges_list


def get_slatm_for_data(data, slatm_params_dict):
    """
    generate slatm representation for all molecules in a dataset

    args: 
        data : list of qm9 molecule dicts
        slatm_params_dict : dict with slatm parameters

    returns: 
        X : array with slatm represenation for all the molecules in data
    """
    max_size = get_max_size(data)
    nuclear_charges_list = get_nuclear_charges_list(data)
    if 'pbc' in slatm_params_dict:
        mbtypes = get_slatm_mbtypes(nuclear_charges_list, slatm_params_dict['pbc'])
    else:
        mbtypes = get_slatm_mbtypes(nuclear_charges_list)
    nuclear_charges_test = np.asarray([ELEMENT2NUC[el] for el in data[0]['elements']])
    nuclear_coords_test = np.asarray(data[0]['coords'])
    X_test = generate_slatm(nuclear_coords_test, nuclear_charges_test, mbtypes,
                            ** slatm_params_dict)
    Z = np.zeros((len(data), max_size))
    C = np.zeros((len(data), max_size, 3))
    X = np.zeros((len(data), X_test.shape[0]))
    for i, mol in enumerate(data):
        nuclear_charges = np.asarray([ELEMENT2NUC[el] for el in mol['elements']])
        nuclear_coords = mol['coords']
        X[i] = generate_slatm(nuclear_coords, nuclear_charges, mbtypes, ** slatm_params_dict)
        Z[i, :nuclear_charges.shape[0]] = nuclear_charges
        C[i, :nuclear_coords.shape[0], :] = nuclear_coords
    return (X, max_size, Z, C)



# main code
# =========


def main(datadir, dataname, freq_analysis_name, outputdir, outputname, rep):

    print(f'Fetching data from {datadir}{dataname}...')
    data = load_qm9_data(datadir, dataname)

    # text = kwargs['text']
    uid = str(uuid4().hex)
    # dt = datetime.now()
    # date = dt.strftime('%Y%m%d%M%H%S')

    with open(f'{datadir}{freq_analysis_name}', 'rb') as inf:
        data_freq = pickle.load(inf)

    # get homo and lumo energies
    print('Fetching HOMO and LUMO energies...')
    homos, lumos, gaps = get_homo_lumo_energies(data)

    # for the moment I am not so much interested in these properties
    # properties = np.asarray([mol['properties'] for mol in data])

    labels = get_labels_qm9_gap(data_freq)

    # generate the representations
    if rep == 'cm':
        print(f'Generate Coulomb Matrix representation for molecules ...')
        X, max_size, Z, C = get_coulomb_matrices_for_data(data)
        print(f'Saving to {outputdir}{outputname} ...')
        np.savez(f'{outputdir}{outputname}', X=X, H=homos, L=lumos, G=gaps,
                 Z=Z, C=C, max_size=max_size, D=dataname, U=uid, R='cm',
                 labels=labels)
        print('Done!')
        
    elif rep == 'bob':
        print(f'Generate Bag of Bonds representation for molecules ...')
        X, max_size, atomtypes, asize, Z, C = get_bob_for_data(data)
        print(f'Saving to {outputdir}{outputname} ...')
        np.savez(f'{outputdir}{outputname}', X=X, H=homos, L=lumos, G=gaps,
                 Z=Z, C=C, max_size=max_size, atomtypes=atomtypes, asize=asize, D=dataname,
                 U=uid, R='bob', label=labels)
        print('Done!')
        
    elif rep == 'slatm':
        print(f'Generate SLATM representation for molecules ...')
        slatm_params_dict = {}
        X, max_size, Z, C = get_slatm_for_data(data, slatm_params_dict)
        print(f'Saving to {outputdir}{outputname} ...')
        np.savez(f'{outputdir}{outputname}', X=X, H=homos, L=lumos, G=gaps,
                 Z=Z, C=C, max_size=max_size, D=dataname, U=uid, R='slatm', labels=labels,
                 ** slatm_params_dict)
        print('Done!')
        
    return 


if __name__ == '__main__':

    if len(sys.argv[1:]) < 1:

        usage()

    main( * sys.argv[1:])
