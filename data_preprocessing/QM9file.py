import numpy as np
import os
import tarfile


# loads the data from qm9 automatically
def load_qm9_data(datadir, dataname):
    """
    loads data of qm9 dataset archive

    args:
        datadir: directory of the data file with "/" at the end
        filename: name of the archive file

    returns :
        qm9_data : list of dictionaries, one for each molecule
    """
    archive = tarfile.open(f'{datadir}{dataname}', 'r:bz2')
    filenames = sorted(archive.getnames())
    archive.extractall(path=datadir)
    qm9data = []
    for f in filenames:
        qm9f = QM9file(f'{datadir}{f}')
        qm9data.append(qm9f.read_data())
        qm9f.close_file()
        os.remove(f'{datadir}{f}')
    archive.close()
    return qm9data


class QM9file:

    
    def __init__(self, path):
        """
        args: 
            path : path to the qm9 file
        """
        self.path = path
        self.f = open(self.path, 'r')
        self.lines = self.f.readlines()
        return


    def get_filename(self):
        """
        get the filename
        
        returns:
            filename : string
        """
        return self.path.rsplit('/', 1)[-1]


    def close_file(self):
        """
        close the file
        """
        self.f.close()
        return


    def clean_number_format(self, num):
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
    

    def read_coords(self, line):
        """
        read the coordinates from an xyz file
        
        args: 
            line : coordinates of one element as a list of strings

        returns: 
             c : list of coordinates as floats
        """
        c = []
        for x in line:
            c.append(self.clean_number_format(x))
        return c


    def read_data(self):
        """
        reads all the information in the qm9 file
        
        returns:
            data : dict with all the properties
        """
        self.properties = {}
        self.properties['nat'] = int(self.lines[0].strip())
        property_keys = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap',
                         'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        for i, key in enumerate(property_keys):
            if i == 0:
                self.properties[key] = self.lines[1].split()[i]
            elif i == 1:
                self.properties[key] = int(self.lines[1].split()[i])
            else:
                self.properties[key] = float(self.lines[1].split()[i])

        nat = self.properties['nat']
        self.elements = np.empty(nat).astype(str)
        self.coords = np.zeros((nat, 3))
        self.mulliken = np.zeros(nat)
        self.frequencies = []
        for i, line in enumerate(self.lines[2:(nat + 2)]):
            l = line.split()
            self.elements[i] = l[0]
            self.coords[i] = self.read_coords(l[1:4])
            self.mulliken[i] = self.clean_number_format(l[4])
        for freq in self.lines[nat + 2].split():
            self.frequencies.append(float(freq))
        smiles_gdb9 = self.lines[nat+3].split()[0]
        smiles_rel = self.lines[nat+3].split()[1]
        inchi_gdb9 = self.lines[nat+4].split()[0]
        inchi_rel = self.lines[nat+4].split()[1]
        return {'properties':self.properties, 'elements':self.elements,
                'coords':self.coords, 'mulliken':self.mulliken,
                'frequencies':self.frequencies, 'SMILES_GDB9': smiles_gdb9,
                'SMILES_rel': smiles_rel, 'InChI_GDB9': inchi_gdb9,
                'InChI_rel': inchi_rel}





    
