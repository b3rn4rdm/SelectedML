import numpy as np
import os


#################################33
#
# Script to handle qm7b dataset
#
###################################3


property_names = ['ae_pbe0', 'p_pbe0', 'p_scs', 'homo_gw', 'homo_pbe0', 'homo_zindo',
              'lumo_gw', 'lumo_pbe0', 'lumo_zindo', 'ip_zindo', 'ea_zindo',
              'e1_zindo', 'emax_zindo', 'imax_zindo']

n_qm7b = 7211


def clean_number_format(num):
    """
    clean some faulty number format in the coordinates
   
    args:
        num : faulty number as a string

    returns : 
        n : corrected number as a float
    """
    if '*^' in num:
        # print(num)
        num = num.replace('*^','e')
    elif '.*^' in num:
        num = num.replace('.*^','e')
        # print(num)
    return float(num)


def read_coords(line):
    """
    read the coordinates from an xyz file
    
    args: 
        line : coordinates of one element as a list of strings

    returns: 
        c : list of coordinates as floats
    """
    c = []
    for x in line:
        c.append(clean_number_format(x))
    return c


def read_qm7b_file(path2file):
    """
    read the qm7b xyz file
    
    args: 
        path2file : path to the qm7b file

    returns: 
        qm7b_data : list of dictionaries, one for each molecule in the dataset
    """
    qm7b_data = []
    with open(path2file, 'r') as inf:
        for n in range(n_qm7b):
            nat = int(inf.readline().split()[0])
            property_values = np.asarray(inf.readline().split()).astype(float)
            elements = []
            coords = []
            for i in range(nat):
                line = inf.readline().split()
                elements.append(line[0])
                coords.append(read_coords(line[1:4]))
            qm7b_data.append({'elements': elements, 'coords': np.asarray(coords), 
                              'properties': {p: val for p, val
                                             in zip(property_names, property_values)}})
            e = inf.readline()
    return qm7b_data
