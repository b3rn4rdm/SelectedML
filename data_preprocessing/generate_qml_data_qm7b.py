##################################################33
#
# Script to generate input data for qml from the qm7b dataset
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

from generate_qml_data import get_max_size, get_coulomb_matrices_for_data, get_asize, get_bob_for_data, get_nuclear_charges_list, get_slatm_for_data

from QM7bfile import *

# insert the path to your copy of xyz2mol here
sys.path.insert(1, '../xyz2mol/')

from xyz2mol import xyz2mol


# some globally useful data
ELEMENT2NUC = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'Cl': 17}


def usage():
    message = """
    Generate represenations data and labels for QML.

    Usage:
    python generate_qml_data.py inputdir dataname outputdir outputname rep=rep text=text **kwargs
    
    args: 
        inputdir : directory containing the original data with '/' at the end
        dataname : name of the data file
        outputdir : directory for the resulting representation data
        outputname : name of the representations file
        rep : name of the representation (cm, bob, slatm, acsf)
        text (optional) : commentary text to help explain the data further
        kwargs : other parameters of the represntation that need to be specified ('cm', 'bob', 'slatm', 'acsf', 'fchl_acsf')
    """
    print(message)
    return sys.exit(0)


def get_prop_from_qm7b(qm7b_data, key='lumo_gw'):
    lumos = []
    for mol in qm7b_data:
        lumos.append(mol['properties'][key])
    return np.asarray(lumos)

match_carbonyl = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#6]=[#8]'))
match_nitrile = lambda x : x.HasSubstructMatch(Chem.MolFromSmarts('[#6]#[#7]'))

def get_labels_qm7b_gap_zindo(data):

    labels = []

    for index, values in data.items():

        tags = values['tags']
        if 'fullysat' in tags:
            labels.append(2)
        elif 'aromatic' in tags:
            labels.append(0)
        elif 'C=O' in tags:
            if '1_unsat' in tags:
                labels.append(1)
            else:
                labels.append(0)
        elif '1_unsat' in tags:
            if 'nitrile' in tags:
                labels.append(2)
            else:
                labels.append(1)
        else:
            labels.append(0)

    return np.asarray(labels)



def main(inputdir, inputname, freq_analysis_name, outputdir, outputname, rep):

    print(f'Fetching data from {inputdir}{inputname}...')
    data = read_qm7b_file(f'{inputdir}{inputname}')

    # text = kwargs['text']
    uid = str(uuid4().hex)
    # dt = datetime.now()
    # date = dt.strftime('%Y%m%d%M%H%S')

    with open(f'{inputdir}{freq_analysis_name}', 'rb') as inf:
        data_freq = pickle.load(inf)

    # get homo and lumo energies
    print('Fetching HOMO and LUMO energies...')
    homo_keys = ['homo_zindo', 'homo_pbe0', 'homo_gw']
    lumo_keys = ['lumo_zindo', 'lumo_pbe0', 'lumo_gw']
    gap_keys = ['gap_zindo', 'gap_pbe0', 'gap_gw']
    homos = {}
    lumos = {}
    gaps = {}
    for hk, lk, gk in zip(homo_keys, lumo_keys, gap_keys):
        homos[hk] = get_prop_from_qm7b(data, hk)
        lumos[lk] = get_prop_from_qm7b(data, lk)
        gaps[gk] = lumos[lk] - homos[hk]

    # labels_0 = label_data(data, 0)
    # labels_1 = label_data(data, 1)
    # labels_2 = label_data(data, 2)
    # labels_3 = label_data(data, 3)
    labels = get_labels_qm7b_gap_zindo(data_freq)

    # generate the representations
    if rep == 'cm':
        print(f'Generate Coulomb Matrix representation for molecules ...')
        X, max_size, Z, C = get_coulomb_matrices_for_data(data)
        print(f'Saving to {outputdir}{outputname} ...')
        np.savez(f'{outputdir}{outputname}', X=X,
                 H_ZINDO=homos['homo_zindo'], H_PBE0=homos['homo_pbe0'], H_GW=homos['homo_gw'],
                 L_ZINDO=lumos['lumo_zindo'], L_PBE0=lumos['lumo_pbe0'], L_GW=lumos['lumo_gw'],
                 G_ZINDO=gaps['gap_zindo'], G_PBE0=gaps['gap_pbe0'], G_GW=gaps['gap_gw'],
                 Z=Z, C=C, max_size=max_size, U=uid, D=inputname, R='cm', labels=labels)
        print('Done!')
        
    elif rep == 'bob':
        print(f'Generate Bag of Bonds representation for molecules ...')
        X, max_size, atomtypes, asize, Z, C = get_bob_for_data(data)
        print(f'Saving to {outputdir}{outputname} ...')
        np.savez(f'{outputdir}{outputname}', X=X,
                 H_ZINDO=homos['homo_zindo'], H_PBE0=homos['homo_pbe0'], H_GW=homos['homo_gw'],
                 L_ZINDO=lumos['lumo_zindo'], L_PBE0=lumos['lumo_pbe0'], L_GW=lumos['lumo_gw'],
                 G_ZINDO=gaps['gap_zindo'], G_PBE0=gaps['gap_pbe0'], G_GW=gaps['gap_gw'], 
                 Z=Z, C=C, max_size=max_size, atomtypes=atomtypes, asize=asize, D=inputname,
                 U=uid, R='bob', labels=labels)
        print('Done!')
        
    elif rep == 'slatm':
        print(f'Generate SLATM representation for molecules ...')
        slatm_params_dict = {}
        X, max_size, Z, C = get_slatm_for_data(data, slatm_params_dict)
        print(f'Saving to {outputdir}{outputname} ...')
        np.savez(f'{outputdir}{outputname}', X=X,
                 H_ZINDO=homos['homo_zindo'], H_PBE0=homos['homo_pbe0'], H_GW=homos['homo_gw'],
                 L_ZINDO=lumos['lumo_zindo'], L_PBE0=lumos['lumo_pbe0'], L_GW=lumos['lumo_gw'],
                 G_ZINDO=gaps['gap_zindo'], G_PBE0=gaps['gap_pbe0'], G_GW=gaps['gap_gw'], 
                 Z=Z, C=C, max_size=max_size, D=inputname, U=uid, R='slatm', labels=labels,
                 ** slatm_params_dict)
        print('Done!')
        
    return


if __name__ == '__main__':

    if len(sys.argv[1:]) < 1:

        usage()

    main( * sys.argv[1:])



